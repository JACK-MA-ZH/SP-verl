#!/bin/bash
set -x



# --- 路径配置 (使用你的绝对路径) ---
BASE_PATH="/inspire/hdd/global_user/wuyouran-253108540218/llm/verl"
DATA_DIR="$BASE_PATH/drc_data_square"
MODEL_PATH="/inspire/hdd/global_user/wuyouran-253108540218/llm/Qwen2.5-VL-3B-Instruct"

# 配置文件路径
TOOL_CONFIG="verl/tools/config/drc_tools.yaml"
INTERACTION_CONFIG="verl/interactions/config/drc_interaction.yaml"
PROJECT_DIR="$(pwd)"
AGENTLOOP_CONFIG_PATH="$BASE_PATH/recipe/drc/config/agent.yaml"
# 运行训练
python -m verl.trainer.main_ppo \
    trainer.project_name="drc_adversarial_rl" \
    trainer.experiment_name="drc_gen_vs_fix_grpo_v2" \
    trainer.total_epochs=15 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.logger='["console"]' \
    +data._target_=verl.utils.dataset.drc_dataset.DRCDataset \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.prompt_key=generation_prompt \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.train_batch_size=64 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$INTERACTION_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.format=qwen-vl \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=false \
    reward_model.reward_manager=drc \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENTLOOP_CONFIG_PATH" 2>&1 | tee drc_train.log