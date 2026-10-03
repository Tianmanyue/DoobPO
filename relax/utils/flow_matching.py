"""
Linear Conditional Flow Matching (Conditional Optimal Transport).

Convention
----------
  t = 0 → pure noise (x_0 ~ N(0, I))
  t = 1 → clean data  (x_1 = action)

Interpolation:  x_t = (1 - t) · x_0  +  t · x_1
Target velocity: u_t = x_1 - x_0

Analogous to `GaussianDiffusion` but replaces the discrete denoising
chain with a continuous ODE solved via Euler integration.
"""

from typing import Protocol, Tuple
from dataclasses import dataclass

import jax, jax.numpy as jnp
import optax


class VelocityModel(Protocol):
    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        ...


def _broadcast_sigma(sigma: jax.Array, x: jax.Array) -> jax.Array:
    """Broadcast sigma to match x.shape = (B, D).

    Handles scalar (ndim=0), per-sample (B,), and per-dim (B, D).
    Shape conditionals are resolved at JAX trace time.
    """
    if sigma.ndim == 0:
        return jnp.broadcast_to(sigma, x.shape)
    elif sigma.ndim == 1:
        return jnp.broadcast_to(sigma[:, None], x.shape)
    return sigma


@dataclass(frozen=True)
class FlowMatching:
    num_steps: int  # Euler integration steps for ODE sampling

    def interpolate(self, t: jax.Array, x_start: jax.Array, noise: jax.Array) -> jax.Array:
        """x_t = (1 - t) * noise + t * x_start."""
        if t.ndim == 1:
            t = t[:, None]
        return (1.0 - t) * noise + t * x_start

    def ode_sample(self, key: jax.Array, model: VelocityModel, shape: Tuple[int, ...]) -> jax.Array:
        """Euler ODE integration from t=0 (noise) to t=1 (data)."""
        x = jax.random.normal(key, shape)
        dt = 1.0 / self.num_steps

        def body_fn(x, t_val):
            velocity = model(t_val, x)
            return x + dt * velocity, None

        t_values = jnp.linspace(0.0, 1.0 - dt, self.num_steps)
        x, _ = jax.lax.scan(body_fn, x, t_values)
        return x

    def velocity_loss(self, key: jax.Array, model: VelocityModel, t: jax.Array,
                      x_start: jax.Array) -> jax.Array:
        """Standard (unweighted) velocity matching loss."""
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_t = self.interpolate(t, x_start, noise)
        velocity_pred = model(t, x_t)
        velocity_target = x_start - noise
        return optax.l2_loss(velocity_pred, velocity_target).mean()

    def weighted_velocity_loss(self, key: jax.Array, weights: jax.Array, model: VelocityModel,
                               t: jax.Array, x_start: jax.Array) -> jax.Array:
        """Ratio-weighted velocity matching loss for drift matching."""
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_t = self.interpolate(t, x_start, noise)
        velocity_pred = model(t, x_t)
        velocity_target = x_start - noise
        loss = weights * optax.squared_error(velocity_pred, velocity_target)
        return loss.mean()

    # ------------------------------------------------------------------
    # FPO / ReinFlow / PiRL utilities
    # ------------------------------------------------------------------

    def velocity_loss_per_sample(self, key: jax.Array, model: VelocityModel,
                                 t: jax.Array, x_start: jax.Array) -> jax.Array:
        """Per-sample CFM loss — shape ``(B,)`` instead of scalar."""
        noise = jax.random.normal(key, x_start.shape)
        x_t = self.interpolate(t, x_start, noise)
        velocity_pred = model(t, x_t)
        velocity_target = x_start - noise
        return jnp.mean(optax.squared_error(velocity_pred, velocity_target), axis=-1)

    def sde_sample(
        self,
        key: jax.Array,
        model_fn: VelocityModel,
        sigma_fn,
        shape: Tuple[int, ...],
        randn_clip: float = 0.0,
        use_drift_correction: bool = False,
        clip_intermediate: float = 0.0,
        normalize_logprob: bool = False,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Euler SDE integration from t=0 to t=1 with noise injection.

        Parameters
        ----------
        model_fn   : (t_scalar, x_batch) -> velocity_batch
        sigma_fn   : (t_scalar, x_batch) -> sigma
                     scalar, ``(B,)``, or ``(B, act_dim)``
        shape      : ``(B, act_dim)``
        randn_clip : clip sampled noise to ±randn_clip (0 = no clip)
        use_drift_correction : if True, apply the score-based drift
            correction for Flow-SDE (PiRL).  The corrected mean becomes

                mean = x + v·dt − σ²/(2(1−t)) · (x − v·t)

            which accounts for the noise-induced drift in the
            probability flow.  ReinFlow does NOT use this.
        clip_intermediate : if > 0, clip the deterministic mean to
            ±clip_intermediate at each step before noise injection,
            matching the original ReinFlow ``denoised_clip_value``.
        normalize_logprob : if True, divide the total log-prob by
            ``(num_steps + 1) * act_dim`` to keep values in a
            reasonable range (matching original ReinFlow).

        Returns
        -------
        action     : ``(B, act_dim)`` — final x_K
        trajectory : ``(K+1, B, act_dim)`` — ``[x_0, x_1, …, x_K]``
        logprob    : ``(B,)`` — total log-probability of the trajectory
        """
        init_key, noise_key = jax.random.split(key)
        x_0 = jax.random.normal(init_key, shape)
        dt = 1.0 / self.num_steps
        act_dim = shape[-1]

        init_lp = -0.5 * (jnp.sum(x_0 ** 2, axis=-1) + act_dim * jnp.log(2 * jnp.pi))
        noise_keys = jax.random.split(noise_key, self.num_steps)
        t_values = jnp.linspace(0.0, 1.0 - dt, self.num_steps)

        _clip = randn_clip
        _drift_corr = use_drift_correction
        _clip_interm = clip_intermediate
        _normalize_lp = normalize_logprob

        def body_fn(carry, inputs):
            x, lp = carry
            t_val, step_key = inputs

            vel = model_fn(t_val, x)
            sigma = sigma_fn(t_val, x)
            mean = x + dt * vel

            sigma_bd = _broadcast_sigma(sigma, x)

            if _drift_corr:
                x0_pred = x - vel * t_val
                mean = mean - sigma_bd ** 2 / (2.0 * (1.0 - t_val + 1e-8)) * x0_pred

            if _clip_interm > 0:
                mean = jnp.clip(mean, -_clip_interm, _clip_interm)

            eps = jax.random.normal(step_key, x.shape)
            if _clip > 0:
                eps = jnp.clip(eps, -_clip, _clip)

            x_next = mean + sigma_bd * eps

            step_lp = -0.5 * jnp.sum(
                (x_next - mean) ** 2 / (sigma_bd ** 2 + 1e-16)
                + jnp.log(2 * jnp.pi)
                + 2 * jnp.log(sigma_bd + 1e-8),
                axis=-1,
            )
            return (x_next, lp + step_lp), x

        (x_final, total_lp), x_history = jax.lax.scan(
            body_fn, (x_0, init_lp), (t_values, noise_keys)
        )

        if _normalize_lp:
            total_lp = total_lp / (self.num_steps + 1) / act_dim

        trajectory = jnp.concatenate([x_history, x_final[None]], axis=0)
        return x_final, trajectory, total_lp

    # ------------------------------------------------------------------
    # PiRL: single-step SDE (inject noise at exactly 1 random step)
    # ------------------------------------------------------------------

    def sde_sample_single_step(
        self,
        key: jax.Array,
        model_fn: VelocityModel,
        sigma_fn,
        shape: Tuple[int, ...],
        use_drift_correction: bool = True,
        max_sde_step: int = 0,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """SDE sampling that injects noise at exactly ONE randomly chosen step.

        All other steps are deterministic ODE (Euler) steps.
        Matches the original PiRL/RLinf ``flow_sde`` mode where
        ``denoise_inds = [k] * num_steps`` and only step ``k`` is SDE.

        Parameters
        ----------
        max_sde_step : upper bound (exclusive) for SDE step sampling.
            0 means ``num_steps`` (default); ``num_steps - 1`` implements
            the ``ignore_last`` option from the original RLinf.

        Returns
        -------
        action     : ``(B, act_dim)``
        trajectory : ``(K+1, B, act_dim)``
        logprob    : ``(B,)`` — log-prob at the SDE step only
        sde_step   : ``()``  — which step was chosen for SDE
        """
        k_key, init_key, noise_key = jax.random.split(key, 3)
        upper = max_sde_step if max_sde_step > 0 else self.num_steps
        sde_step = jax.random.randint(k_key, (), 0, upper)

        x_0 = jax.random.normal(init_key, shape)
        dt = 1.0 / self.num_steps

        t_values = jnp.linspace(0.0, 1.0 - dt, self.num_steps)
        step_indices = jnp.arange(self.num_steps)

        _drift_corr = use_drift_correction

        def body_fn(carry, inputs):
            x, lp = carry
            t_val, step_idx = inputs

            vel = model_fn(t_val, x)
            sigma = sigma_fn(t_val, x)
            sigma_bd = _broadcast_sigma(sigma, x)
            mean = x + dt * vel

            if _drift_corr:
                x0_pred = x - vel * t_val
                mean = mean - sigma_bd ** 2 / (2.0 * (1.0 - t_val + 1e-8)) * x0_pred

            eps = jax.random.normal(
                jax.random.fold_in(noise_key, step_idx), x.shape,
            )
            x_next_sde = mean + sigma_bd * eps
            x_next_ode = mean

            step_lp = -0.5 * jnp.sum(
                (x_next_sde - mean) ** 2 / (sigma_bd ** 2 + 1e-16)
                + jnp.log(2 * jnp.pi)
                + 2 * jnp.log(sigma_bd + 1e-8),
                axis=-1,
            )

            is_sde = step_idx == sde_step
            x_next = jnp.where(is_sde, x_next_sde, x_next_ode)
            lp_new = jnp.where(is_sde, lp + step_lp, lp)
            return (x_next, lp_new), x

        init_lp = jnp.zeros(shape[0])
        (x_final, total_lp), x_history = jax.lax.scan(
            body_fn, (x_0, init_lp), (t_values, step_indices),
        )
        trajectory = jnp.concatenate([x_history, x_final[None]], axis=0)
        return x_final, trajectory, total_lp, sde_step

    def compute_single_step_logprob(
        self,
        model_fn: VelocityModel,
        sigma_fn,
        trajectory: jax.Array,
        sde_step_indices: jax.Array,
        use_drift_correction: bool = True,
    ) -> jax.Array:
        """Re-evaluate log-prob at a per-sample SDE step of a stored trajectory.

        Parameters
        ----------
        trajectory       : ``(K+1, B, act_dim)``
        sde_step_indices : ``(B,)`` int — which step to evaluate for each sample
            (also accepts a scalar, applied to all samples).

        The ``model_fn`` must accept ``(t_batch, x_batch)`` where
        ``t_batch`` is ``(B,)`` — **not** a scalar.

        Returns
        -------
        logprob : ``(B,)``
        """
        dt = 1.0 / self.num_steps
        B = trajectory.shape[1]
        batch_idx = jnp.arange(B)

        sde_step_indices = jnp.broadcast_to(
            jnp.asarray(sde_step_indices, dtype=jnp.int32), (B,),
        )

        t_k = sde_step_indices.astype(jnp.float32) * dt   # (B,)
        x_k = trajectory[sde_step_indices, batch_idx]      # (B, act_dim)
        x_k_next = trajectory[sde_step_indices + 1, batch_idx]  # (B, act_dim)

        vel = model_fn(t_k, x_k)
        sigma = sigma_fn(t_k, x_k)
        sigma_bd = _broadcast_sigma(sigma, x_k)
        mean = x_k + dt * vel

        if use_drift_correction:
            x0_pred = x_k - vel * t_k[:, None]
            t_safe = jnp.maximum(t_k, 1e-8)
            mean = mean - sigma_bd ** 2 / (2.0 * (1.0 - t_safe[:, None] + 1e-8)) * x0_pred

        diff = x_k_next - mean
        lp = -0.5 * jnp.sum(
            diff ** 2 / (sigma_bd ** 2 + 1e-16)
            + jnp.log(2 * jnp.pi)
            + 2 * jnp.log(sigma_bd + 1e-8),
            axis=-1,
        )
        return lp

    def compute_trajectory_logprob(
        self,
        model_fn: VelocityModel,
        sigma_fn,
        trajectory: jax.Array,
        use_drift_correction: bool = False,
        clip_intermediate: float = 0.0,
        normalize_logprob: bool = False,
    ) -> jax.Array:
        """Re-evaluate log-prob of a *stored* trajectory under current params.

        Parameters
        ----------
        model_fn   : (t_scalar, x_batch) -> velocity_batch  (current θ)
        sigma_fn   : (t_scalar, x_batch) -> sigma
                     scalar, ``(B,)``, or ``(B, act_dim)``
        trajectory : ``(K+1, B, act_dim)``
        use_drift_correction : same as in ``sde_sample``
        clip_intermediate : same as in ``sde_sample``
        normalize_logprob : same as in ``sde_sample``

        Returns
        -------
        logprob : ``(B,)``
        """
        act_dim = trajectory.shape[-1]
        dt = 1.0 / self.num_steps

        x_0 = trajectory[0]
        init_lp = -0.5 * (jnp.sum(x_0 ** 2, axis=-1) + act_dim * jnp.log(2 * jnp.pi))

        t_values = jnp.linspace(0.0, 1.0 - dt, self.num_steps)

        _drift_corr = use_drift_correction
        _clip_interm = clip_intermediate
        _normalize_lp = normalize_logprob

        def step_fn(lp, inputs):
            x_k, x_k_next, t_k = inputs
            vel = model_fn(t_k, x_k)
            sigma = sigma_fn(t_k, x_k)
            mean = x_k + vel * dt

            sigma_bd = _broadcast_sigma(sigma, x_k)

            if _drift_corr:
                x0_pred = x_k - vel * t_k
                mean = mean - sigma_bd ** 2 / (2.0 * (1.0 - t_k + 1e-8)) * x0_pred

            if _clip_interm > 0:
                mean = jnp.clip(mean, -_clip_interm, _clip_interm)

            diff = x_k_next - mean

            step_lp = -0.5 * jnp.sum(
                diff ** 2 / (sigma_bd ** 2 + 1e-16)
                + jnp.log(2 * jnp.pi)
                + 2 * jnp.log(sigma_bd + 1e-8),
                axis=-1,
            )
            return lp + step_lp, None

        total_lp, _ = jax.lax.scan(
            step_fn, init_lp, (trajectory[:-1], trajectory[1:], t_values)
        )

        if _normalize_lp:
            total_lp = total_lp / (self.num_steps + 1) / act_dim

        return total_lp
