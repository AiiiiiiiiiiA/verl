set -x

# This script is a test for the search-augmented agent functionality
# using a small model (Qwen1.5-0.5B-Chat) and a dummy dataset.

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=examples/data_preprocess/dummy_search_qa_2lines_formatted.jsonl \
    data.val_files=examples/data_preprocess/dummy_search_qa_2lines_formatted.jsonl \
    data.train_batch_size=2 \
    data.max_prompt_length=128 \
    data.max_response_length=100 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path="Qwen/Qwen1.5-0.5B-Chat" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path="Qwen/Qwen1.5-0.5B-Chat" \
    critic.model.trust_remote_code=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_test_search_agent' \
    trainer.experiment_name='qwen0.5b_search_test' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.max_steps=2 \
    \
    # Search agent specific parameters
    actor_rollout_ref.rollout.agent_search_config.do_search=true \
    actor_rollout_ref.rollout.agent_search_config.max_turns=2 \
    actor_rollout_ref.rollout.agent_search_config.retriever.url="http://127.0.0.1:8000/retrieve" \
    actor_rollout_ref.rollout.agent_search_config.retriever.top_k=1 \
    actor_rollout_ref.rollout.agent_search_config.max_new_tokens_per_turn=50 $@
# Note: tensor_model_parallel_size was removed as it's more relevant for larger models/vLLM.
# FSDP might still be active by default if not explicitly disabled for strategy,
# but with n_gpus_per_node=1, it will run on a single GPU.
# For such a small model, FSDP might be an overhead; strategy could be set to 'ddp' or 'naive' if issues arise.
# However, the original script used FSDP, so retaining that structure for now.
# `actor_rollout_ref.model.trust_remote_code=True` added for Qwen models.
# `actor_rollout_ref.rollout.name=hf` set for simplicity with small models.
# `trainer.save_freq=-1` and `trainer.test_freq=-1` to disable saving and testing for this short run.
# `trainer.max_steps=2` ensures a very short run.
# `data.filter_overlong_prompts=False` as the dataset is tiny.
# `actor_rollout_ref.rollout.agent_search_config.retriever.top_k=1` for minimal retrieval.
# `data.train_batch_size=2` and `actor_rollout_ref.actor.ppo_mini_batch_size=2` to match the dataset size.
# `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1` and `critic.ppo_micro_batch_size_per_gpu=1` for single GPU processing.
# `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1` for rollout batching.
# `trainer.total_epochs=1` to ensure it doesn't try to run many epochs if max_steps isn't hit first with small dataset.
# `data.truncation='error'` kept from original, might need to be 'padding_only' or similar if issues with tiny prompts.
# Removed `actor_rollout_ref.rollout.tensor_model_parallel_size` and `actor_rollout_ref.rollout.gpu_memory_utilization` (vLLM specific)
# `actor_rollout_ref.model.use_remove_padding` is often beneficial and kept.
# `actor_rollout_ref.model.enable_gradient_checkpointing` is kept, might save memory.
# Default FSDP strategy is assumed from the original script. For single GPU, it effectively acts like DDP.
# `data.max_prompt_length=128` and `data.max_response_length=100` are set.
# `actor_rollout_ref.rollout.agent_search_config.max_new_tokens_per_turn=50` is set.
# `trainer.logger=['console']` to simplify output.
# `trainer.project_name` and `trainer.experiment_name` updated.
# `data.val_files` is set to the same as train for simplicity, though test_freq=-1 means it won't run.
# `algorithm.adv_estimator=gae` is kept.
# `actor_rollout_ref.actor.optim.lr` and `critic.optim.lr` are kept.
# `actor_rollout_ref.actor.fsdp_config.param_offload` and `optimizer_offload` are kept as False.
# `actor_rollout_ref.actor.use_kl_loss=False` is kept.
# `algorithm.use_kl_in_reward=False` is kept.
# `trainer.critic_warmup=0` is kept.
# `set -x` at the beginning is kept for command echoing.
# `$@` at the end allows for additional CLI overrides.
