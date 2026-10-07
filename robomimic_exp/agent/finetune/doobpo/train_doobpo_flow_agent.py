"""
DoobPO-Flow-GRPO (off-policy) fine-tuning agent for robomimic.

The off-policy DoobPO training loop is identical for diffusion and flow policies -- it
only touches the model through a shared interface (forward / loss_critic /
compute_advantage / loss_ratio / loss_actor / update_target_critic /
update_target_policy, and .actor / .critic_q / .ratio_net). So this agent simply
reuses TrainDoobPODiffusionAgent; the flow-specific behaviour (velocity matching,
flow ODE sampling, GRPO group advantage) lives in model/flow/ft_doobpo/doobpo_flow.py.
"""

from agent.finetune.doobpo.train_doobpo_diffusion_agent import TrainDoobPODiffusionAgent


class TrainDoobPOFlowAgent(TrainDoobPODiffusionAgent):
    pass
