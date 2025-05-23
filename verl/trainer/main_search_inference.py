import hydra
from omegaconf import OmegaConf, DictConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensordict import TensorDict
import logging
import sys
import os

# Add project root to sys.path to allow relative imports if script is run directly
# and verl is not installed.
# This assumes the script is in verl/trainer/
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from verl.llm_agent.generation import LLMGenerationManager
from verl import DataProto
# Assuming load_checkpoint_and_model might not be strictly necessary if loading a base model directly
# from a path for inference. If it's for loading fine-tuned checkpoints, its usage would be more complex.
# For simplicity, we'll use AutoModelForCausalLM.from_pretrained directly.

# Configure basic logging
# Basic configuration, can be overridden by Hydra logging config if present
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig) -> None:
    logger.info("Starting Search-Augmented LLM Inference Script...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    # --- Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    # Prioritize actor_rollout_ref.model.path as it's specific to the generative model
    tokenizer_path = config.actor_rollout_ref.model.path
    trust_remote_code = config.actor_rollout_ref.model.get("trust_remote_code", False) # from ppo_trainer
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        sys.exit(1)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Tokenizer pad_token_id set to eos_token_id: {tokenizer.eos_token_id}")

    # Load model
    model_path = config.actor_rollout_ref.model.path
    try:
        # For simple inference with a base model, from_pretrained is sufficient.
        # If loading a fine-tuned Verl checkpoint, a utility like load_checkpoint_and_model would be needed.
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, # Common practice for modern LLMs
            trust_remote_code=trust_remote_code
        )
        model.to(device)
        model.eval()
        logger.info(f"Model loaded from {model_path} and set to eval mode on {device}.")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        sys.exit(1)

    # Instantiate LLMGenerationManager
    # LLMGenerationManager expects the full config to derive agent_search_config
    try:
        llm_manager = LLMGenerationManager(
            module=model,
            tokenizer=tokenizer,
            config=config 
        )
        logger.info("LLMGenerationManager instantiated successfully.")
    except Exception as e:
        logger.error(f"Failed to instantiate LLMGenerationManager: {e}")
        sys.exit(1)

    # --- Interactive Loop ---
    logger.info("Starting interactive prompt loop. Type 'quit' to exit.")
    while True:
        try:
            user_prompt_text = input("Enter your prompt (or 'quit' to exit): ")
        except EOFError: # Handle non-interactive environments (e.g., piping input)
            logger.info("EOF received, exiting.")
            break
            
        if user_prompt_text.lower() == 'quit':
            logger.info("Exiting interactive loop.")
            break
        if not user_prompt_text.strip():
            logger.info("Empty prompt, skipping.")
            continue

        # Prepare DataProto for the prompt
        try:
            prompt_inputs = tokenizer(user_prompt_text, return_tensors="pt", padding=False)
            input_ids = prompt_inputs.input_ids.to(device)
            attention_mask = prompt_inputs.attention_mask.to(device)
            
            # Create position_ids (simple sequential for single prompt)
            position_ids = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0)

            # Non-tensor batch item (minimal for inference)
            # reward_fn_key might be needed if LLMGenerationManager or underlying components expect it
            reward_fn_key = config.data.get("reward_fn_key", "data_source") # Default if not in config
            non_tensor_batch_item = {
                "id": "infer_0", 
                "reward_model": {"ground_truth": ""}, # Placeholder
                reward_fn_key: "inference" # Placeholder
            }

            # Generation parameters for meta_info
            # These should align with what LLMGenerationManager expects for its operations
            generation_params_meta = {
                "response_length": config.actor_rollout_ref.rollout.get("response_length", 512), # Total new tokens
                "do_sample": config.actor_rollout_ref.rollout.get("do_sample", True),
                "temperature": config.actor_rollout_ref.rollout.get("temperature", 0.7),
                "top_p": config.actor_rollout_ref.rollout.get("top_p", 0.9),
                "top_k": config.actor_rollout_ref.rollout.get("top_k", 50),
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "num_return_sequences": 1 # For inference, typically 1
            }
            # Add other params from config if LLMGenerationManager uses them from meta_info directly
            # For example, if max_new_tokens_per_turn is expected in meta_info for some reason.
            # However, LLMGenerationManager now gets these from its own config.

            prompts_tensordict = TensorDict({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids
            }, batch_size=1)

            prompts_dataproc = DataProto(
                batch=prompts_tensordict, 
                non_tensor_batch=[non_tensor_batch_item], # Must be a list for LLMGenerationManager
                meta_info=generation_params_meta
            )
            logger.info(f"Prepared DataProto for prompt: '{user_prompt_text[:100]}...'")

        except Exception as e:
            logger.error(f"Error preparing DataProto for prompt '{user_prompt_text[:100]}...': {e}", exc_info=True)
            continue # Skip to next prompt iteration

        # Run Generation Loop
        try:
            logger.info("Running generation loop...")
            with torch.no_grad():
                generated_output_dp = llm_manager.run_llm_loop(prompts_dataproc)
            logger.info("Generation loop completed.")

            # Output Results
            if generated_output_dp and generated_output_dp.batch.get("input_ids") is not None:
                final_sequence_ids = generated_output_dp.batch["input_ids"][0] # Assuming batch size 1
                # Using skip_special_tokens=True for cleaner output, but False can be useful for debugging
                full_interaction_text = tokenizer.decode(final_sequence_ids, skip_special_tokens=True) 
                
                logger.info("---- Full Interaction ----")
                # Use print for the main output for better readability in console
                print(f"\nUser Prompt: {user_prompt_text}")
                print(f"LLM Output:\n{full_interaction_text}\n")
                logger.info("---- End of Full Interaction ----")

                if generated_output_dp.non_tensor_batch and isinstance(generated_output_dp.non_tensor_batch, list) and generated_output_dp.non_tensor_batch[0]:
                    action_stats = generated_output_dp.non_tensor_batch[0].get('action_stats')
                    if action_stats:
                        logger.info(f"Action Stats: {action_stats}")
                    else:
                        logger.info("No action_stats found in the output.")
                else:
                    logger.info("No non_tensor_batch or action_stats found in the output.")
            else:
                logger.warning("Generated output or input_ids not found in DataProto.")

        except Exception as e:
            logger.error(f"Error during generation loop for prompt '{user_prompt_text[:100]}...': {e}", exc_info=True)
            # Optionally print a simpler message to console as well
            print(f"An error occurred during generation. Please check logs.")

if __name__ == "__main__":
    main()
```

I've created the content for `verl/trainer/main_search_inference.py`.

Key features implemented:
-   **Imports**: All necessary modules are imported. A `sys.path` modification is included to help with running the script directly from the project structure.
-   **Hydra Configuration**: `@hydra.main` decorator is used, pointing to `../config` and `ppo_trainer.yaml`.
-   **`main(config: OmegaConf)` function**:
    -   **Initialization**:
        -   Basic logging is set up.
        -   Device (CUDA/CPU) is determined.
        -   `AutoTokenizer` is loaded using `config.actor_rollout_ref.model.path`. `pad_token_id` is set to `eos_token_id` if not present.
        -   `AutoModelForCausalLM` is loaded from `config.actor_rollout_ref.model.path`, moved to the device, and set to `eval()` mode. `torch_dtype=torch.bfloat16` is used as a common practice.
        -   `LLMGenerationManager` is instantiated with the model, tokenizer, and the full Hydra `config`.
    -   **Interactive Loop**:
        -   A `while True` loop prompts the user for input.
        *   Handles `EOFError` for non-interactive execution.
        -   **DataProto Preparation**:
            -   The user's prompt is tokenized.
            -   `input_ids`, `attention_mask`, and `position_ids` are created and moved to the device.
            -   A minimal `non_tensor_batch_item` is created.
            -   `generation_params_meta` dictionary is created to hold necessary parameters for the generation loop (e.g., `response_length`, `do_sample`, `temperature`, `eos_token_id`, `pad_token_id`), sourced from the Hydra config.
            -   `prompts_tensordict` and `prompts_dataproc` are created. `non_tensor_batch` is correctly passed as a list.
        -   **Generation**: `llm_manager.run_llm_loop()` is called within `torch.no_grad()`.
        -   **Output**: The full generated sequence is decoded and logged. Action statistics are also logged from `generated_output_dp.non_tensor_batch[0]`. The main LLM output is printed to `stdout` for better interactive use, while details go to the logger.
    -   Error handling (try-except blocks) is included for major steps like model/tokenizer loading, DataProto preparation, and the generation loop.
-   **`if __name__ == "__main__":`**: Calls `main()`.

This script provides a way to interactively test the search-augmented LLM generation.The file `verl/trainer/main_search_inference.py` has been successfully created with the specified functionality.

Summary of the implementation:
1.  **Imports**: All required modules are imported, including `hydra`, `omegaconf`, `torch`, `transformers`, `tensordict`, `logging`, `sys`, `os`, and local project modules `LLMGenerationManager` and `DataProto`. A `sys.path` modification is included to help with direct script execution from within the project structure.
2.  **Hydra Configuration**:
    *   The `@hydra.main(config_path="../config", config_name="ppo_trainer", version_base=None)` decorator is used to load configuration from `verl/config/ppo_trainer.yaml`.
3.  **`main(config: DictConfig)` Function**:
    *   **Logging and Device**: Basic logging is configured, and the device (CUDA or CPU) is determined.
    *   **Tokenizer Loading**: `AutoTokenizer` is loaded using the path from `config.actor_rollout_ref.model.path`. `trust_remote_code` is also sourced from the config. If `tokenizer.pad_token_id` is `None`, it's set to `tokenizer.eos_token_id`.
    *   **Model Loading**: `AutoModelForCausalLM` is loaded from `config.actor_rollout_ref.model.path`, set to `torch.bfloat16`, moved to the determined device, and put into `eval()` mode.
    *   **`LLMGenerationManager` Instantiation**: The manager is created with the loaded `model`, `tokenizer`, and the full Hydra `config` object.
    *   **Interactive Loop**:
        *   A `while True` loop prompts the user for input via `input()`. It handles 'quit' and empty prompts. It also handles `EOFError` for non-interactive scenarios.
        *   **`DataProto` Preparation**:
            *   The user's text prompt is tokenized.
            *   `input_ids`, `attention_mask`, and `position_ids` are created as tensors and moved to the device.
            *   A minimal `non_tensor_batch_item` dictionary is created (including a placeholder for `reward_model` and `reward_fn_key`).
            *   `generation_params_meta` is constructed using values from the Hydra config (e.g., `response_length`, `do_sample`, `temperature`, `eos_token_id`, `pad_token_id`).
            *   A `TensorDict` (`prompts_tensordict`) is created for the batch part of `DataProto`.
            *   `prompts_dataproc` is instantiated, with `non_tensor_batch` correctly passed as a list containing `non_tensor_batch_item`.
        *   **Generation Execution**: `llm_manager.run_llm_loop()` is called within a `with torch.no_grad():` block.
        *   **Output Display**:
            *   The full generated sequence (`input_ids` from the output `DataProto`) is decoded using the tokenizer.
            *   The user's prompt and the LLM's full interaction text are printed to the console for readability.
            *   Detailed action statistics (`action_stats` from the output `DataProto`'s `non_tensor_batch`) are logged.
    *   **Error Handling**: `try-except` blocks are used for key operations like loading, data preparation, and generation to catch and log potential errors gracefully.
4.  **Execution Block**: The `if __name__ == "__main__":` block calls the `main()` function, allowing the script to be run directly.

The script is structured to facilitate interactive testing of the search-augmented `LLMGenerationManager`.
