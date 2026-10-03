# Intentionally minimal: do not import off_policy / jax here.
# `python -m relax.trainer.evaluator` must load evaluator.py before JAX initializes;
# eager imports would pull in OffPolicyTrainer and force CUDA in the evaluator child.
