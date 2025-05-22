import re
import logging
import torch
from tensordict import TensorDict
from transformers import GenerationConfig
from omegaconf import OmegaConf # Added import

from verl import DataProto
from verl.utils.torch_functional import get_response_mask

logger = logging.getLogger(__name__)

class LLMGenerationManager:
    def __init__(self, module, tokenizer, config):
        self.module = module
        self.tokenizer = tokenizer
        self.config = config # Full Hydra config

        # Access nested search configuration
        agent_config = config.actor_rollout_ref.rollout.agent_search_config
        
        self.max_turns = agent_config.max_turns
        self.do_search = agent_config.do_search
        self.retriever_url = agent_config.retriever.url
        self.retriever_top_k = agent_config.retriever.top_k
        self.max_new_tokens_per_turn = agent_config.max_new_tokens_per_turn
        self.tags = OmegaConf.to_container(agent_config.tags, resolve=True) # Convert to Python dict

        # Ensure tokenizer provides these, or error
        # These could also be part of agent_config.tags if they vary per agent setup
        if not hasattr(self.tokenizer, "eos_token_id") or self.tokenizer.eos_token_id is None:
            logger.warning("Tokenizer missing eos_token_id, trying from module.config")
            self.eos_token_id = getattr(self.module.config, "eos_token_id", None)
        else:
            self.eos_token_id = self.tokenizer.eos_token_id

        if not hasattr(self.tokenizer, "pad_token_id") or self.tokenizer.pad_token_id is None:
            logger.warning("Tokenizer missing pad_token_id, trying from module.config, then eos_token_id")
            self.pad_token_id = getattr(self.module.config, "pad_token_id", self.eos_token_id)
        else:
            self.pad_token_id = self.tokenizer.pad_token_id

        if self.pad_token_id is None:
            logger.error("pad_token_id is None after all checks. This may cause issues.")
            # Potentially raise error if padding is strictly required by model or logic
        
        # Fallback for model_max_length if tokenizer doesn't have it
        if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length is not None:
            self.model_max_length = tokenizer.model_max_length
        elif hasattr(self.module.config, "max_position_embeddings") and self.module.config.max_position_embeddings is not None:
            self.model_max_length = self.module.config.max_position_embeddings
        else:
            logger.warning("Could not determine model_max_length from tokenizer or module config, defaulting to 2048.")
            self.model_max_length = 2048


    def _parse_llm_action(self, decoded_text: str):
        # Ensure DOTALL is used if newlines can be within the content
        search_pattern = rf"{re.escape(self.tags['search_tag_open'])}(.*?){re.escape(self.tags['search_tag_close'])}"
        answer_pattern = rf"{re.escape(self.tags['answer_tag_open'])}(.*?){re.escape(self.tags['answer_tag_close'])}"
        think_pattern = rf"{re.escape(self.tags['think_tag_open'])}(.*?){re.escape(self.tags['think_tag_close'])}"

        search_match = re.search(search_pattern, decoded_text, re.DOTALL)
        answer_match = re.search(answer_pattern, decoded_text, re.DOTALL)
        think_match = re.search(think_pattern, decoded_text, re.DOTALL)

        if search_match:
            end_pos = search_match.end()
            return "search", search_match.group(1).strip(), decoded_text[:end_pos]
        elif answer_match:
            end_pos = answer_match.end()
            return "answer", answer_match.group(1).strip(), decoded_text[:end_pos]
        elif think_match:
            end_pos = think_match.end()
            return "think", think_match.group(1).strip(), decoded_text[:end_pos]
        
        return "none", "", decoded_text # Return the full text if no specific action tag

    def run_llm_loop(self, prompts: DataProto) -> DataProto:
        logger.info(f"Running LLM generation loop for max {self.max_turns} turns...")

        original_input_ids = prompts.batch["input_ids"]
        original_attention_mask = prompts.batch.get("attention_mask", torch.ones_like(original_input_ids))
        original_position_ids = prompts.batch.get("position_ids", torch.arange(original_input_ids.shape[1], device=original_input_ids.device).expand_as(original_input_ids))
        
        device = original_input_ids.device

        num_return_sequences = prompts.meta_info.get("num_return_sequences", getattr(self.config, "n", 1))
        
        # Tile initial inputs if num_return_sequences > 1
        if num_return_sequences > 1:
            current_sequences_input_ids = original_input_ids.repeat_interleave(num_return_sequences, dim=0)
            current_attention_mask_template = original_attention_mask.repeat_interleave(num_return_sequences, dim=0)
            current_position_ids_template = original_position_ids.repeat_interleave(num_return_sequences, dim=0)
        else:
            current_sequences_input_ids = original_input_ids
            current_attention_mask_template = original_attention_mask
            current_position_ids_template = original_position_ids
            
        loop_batch_size = current_sequences_input_ids.size(0)
        active_sequences = [seq.clone() for seq in current_sequences_input_ids] 
        is_done = torch.zeros(loop_batch_size, dtype=torch.bool, device=device)
        
        # Assuming all original prompts are same length due to batching, or use original_attention_mask.sum(1)
        # For simplicity, taking .size(1) of the potentially tiled input.
        original_prompt_actual_length = current_sequences_input_ids.size(1) 

        gen_params_base = {
            "do_sample": prompts.meta_info.get("do_sample", getattr(self.config, "do_sample", True)),
            "temperature": prompts.meta_info.get("temperature", getattr(self.config, "temperature", 0.7)),
            "top_p": prompts.meta_info.get("top_p", getattr(self.config, "top_p", 0.9)),
            "top_k": prompts.meta_info.get("top_k", getattr(self.config, "top_k", 50)),
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "num_return_sequences": 1, # Internal loop always generates 1 sequence per input
        }
        max_new_tokens_total = prompts.meta_info.get("response_length", getattr(self.config, "response_length", 512)) # Overall response length
        # max_new_tokens_per_turn is now set in __init__ from agent_search_config
        # Ensure it's at least 1 if self.max_turns > 0
        current_max_new_tokens_per_turn = self.max_new_tokens_per_turn 
        if self.max_turns > 0 and current_max_new_tokens_per_turn == 0 : # Avoid division by zero or no progress
            current_max_new_tokens_per_turn = max(1, max_new_tokens_total // self.max_turns)
            logger.warning(f"max_new_tokens_per_turn was 0, adjusted to {current_max_new_tokens_per_turn} based on total response length and max_turns.")


        self.module.eval()
        for turn_num in range(self.max_turns):
            if is_done.all():
                logger.info("All items done, breaking loop early.")
                break
            
            logger.info(f"Turn {turn_num + 1}/{self.max_turns}")

            input_ids_list_turn = [active_sequences[i] for i, done_flag in enumerate(is_done) if not done_flag]
            if not input_ids_list_turn: break

            max_len_turn = max(seq.size(0) for seq in input_ids_list_turn)
            
            current_batch_input_ids = torch.full((len(input_ids_list_turn), max_len_turn), self.pad_token_id, dtype=torch.long, device=device)
            current_batch_attention_mask = torch.zeros((len(input_ids_list_turn), max_len_turn), dtype=torch.long, device=device)

            for i, seq_tensor in enumerate(input_ids_list_turn):
                current_batch_input_ids[i, :seq_tensor.size(0)] = seq_tensor
                current_batch_attention_mask[i, :seq_tensor.size(0)] = 1
            
            turn_gen_config_dict = {**gen_params_base, "max_new_tokens": current_max_new_tokens_per_turn}
            # Remove None values for GenerationConfig
            turn_gen_config_dict = {k:v for k,v in turn_gen_config_dict.items() if v is not None}
            turn_gen_config = GenerationConfig(**turn_gen_config_dict)

            with torch.no_grad():
                turn_output_sequences = self.module.generate(
                    input_ids=current_batch_input_ids,
                    attention_mask=current_batch_attention_mask,
                    generation_config=turn_gen_config,
                )

            active_indices_map = [i for i, done_flag in enumerate(is_done) if not done_flag]
            
            for i_active, original_batch_idx in enumerate(active_indices_map):
                input_seq_len_for_turn = current_batch_input_ids.size(1)
                generated_part_tokens = turn_output_sequences[i_active, input_seq_len_for_turn:]
                
                decoded_new_text = self.tokenizer.decode(generated_part_tokens, skip_special_tokens=True).strip()
                action_type, content, raw_action_text = self._parse_llm_action(decoded_new_text)
                
                tokens_for_current_segment = generated_part_tokens # Default to all generated tokens this turn
                if action_type != "none" and raw_action_text: # Action tag was found
                    tokens_for_current_segment = self.tokenizer.encode(raw_action_text, add_special_tokens=False, return_tensors="pt")[0].to(device)
                
                tokens_to_append_next = tokens_for_current_segment

                if action_type == "search":
                    if self.do_search:
                        logger.info(f"Batch item {original_batch_idx}, Turn {turn_num+1}: Search triggered for query: '{content}'. Would use URL: {self.retriever_url} with top_k: {self.retriever_top_k}.")
                        # TODO: Actual search client call would be here.
                        # For now, using mock results as specified.
                        info_text = f"{self.tags['information_tag_open']}Mock search results for '{content}'{self.tags['information_tag_close']}"
                    else:
                        logger.info(f"Batch item {original_batch_idx}, Turn {turn_num+1}: Search triggered for query: '{content}', but do_search is False. No search performed.")
                        info_text = f"{self.tags['information_tag_open']}Search is disabled.{self.tags['information_tag_close']}"
                    
                    info_tokens = self.tokenizer.encode(info_text, add_special_tokens=False, return_tensors="pt")[0].to(device)
                    tokens_to_append_next = torch.cat([tokens_for_current_segment, info_tokens], dim=0)
                    if turn_num == self.max_turns - 1: is_done[original_batch_idx] = True
                elif action_type == "answer":
                    logger.info(f"Batch item {original_batch_idx}, Turn {turn_num+1}: Answer found.")
                    is_done[original_batch_idx] = True
                elif action_type == "think" or action_type == "none":
                    if turn_num == self.max_turns - 1: is_done[original_batch_idx] = True
                
                active_sequences[original_batch_idx] = torch.cat([active_sequences[original_batch_idx], tokens_to_append_next], dim=0)
                
                if active_sequences[original_batch_idx].size(0) > self.model_max_length:
                    active_sequences[original_batch_idx] = active_sequences[original_batch_idx][:self.model_max_length]
                    is_done[original_batch_idx] = True

        final_max_len = max(seq.size(0) for seq in active_sequences) if active_sequences else 0
        
        # Pad/truncate to overall configured response length
        # original_prompt_actual_length is the length of the prompt part in `active_sequences`
        expected_total_sequence_length = original_prompt_actual_length + max_new_tokens_total
        
        final_sequences_processed = torch.full((loop_batch_size, expected_total_sequence_length), self.pad_token_id, dtype=torch.long, device=device)

        for i, seq_tensor in enumerate(active_sequences):
            len_to_copy = min(seq_tensor.size(0), expected_total_sequence_length)
            final_sequences_processed[i, :len_to_copy] = seq_tensor[:len_to_copy]

        # Final attention mask and position IDs
        final_attention_mask = torch.zeros_like(final_sequences_processed, device=device)
        final_position_ids = torch.zeros_like(final_sequences_processed, device=device)

        for i in range(loop_batch_size):
            # Use current_attention_mask_template and current_position_ids_template for prompt part
            prompt_part_len_from_template = current_attention_mask_template[i].sum().item()
            
            # Ensure we don't copy more than available in template or target
            copy_prompt_len = min(prompt_part_len_from_template, expected_total_sequence_length)
            
            final_attention_mask[i, :copy_prompt_len] = current_attention_mask_template[i, :copy_prompt_len]
            final_position_ids[i, :copy_prompt_len] = current_position_ids_template[i, :copy_prompt_len]
            
            # For the generated part (response)
            # actual_content_length is length of sequence before padding to expected_total_sequence_length
            actual_content_length = min(active_sequences[i].size(0), expected_total_sequence_length) 
            
            # Start of response part in final_sequences_processed
            response_start_offset = original_prompt_actual_length 
            
            if actual_content_length > response_start_offset:
                # Mask for generated part is 1s
                final_attention_mask[i, response_start_offset:actual_content_length] = 1
                
                # Position IDs for generated part
                last_prompt_pos_id = current_position_ids_template[i, min(copy_prompt_len,original_prompt_actual_length)-1] if copy_prompt_len > 0 else -1

                response_part_model_len = actual_content_length - response_start_offset
                if response_part_model_len > 0:
                    final_position_ids[i, response_start_offset:actual_content_length] = torch.arange(
                        last_prompt_pos_id + 1,
                        last_prompt_pos_id + 1 + response_part_model_len,
                        device=device, dtype=torch.long
                    )
        
        # "prompts" TD field should be the prompt part of final_sequences_processed
        # "responses" TD field is the response part of final_sequences_processed
        final_prompt_tokens_td = final_sequences_processed[:, :original_prompt_actual_length]
        final_response_tokens_td = final_sequences_processed[:, original_prompt_actual_length:]

        output_batch_dict = {
            "prompts": final_prompt_tokens_td,
            "responses": final_response_tokens_td,
            "input_ids": final_sequences_processed, # Full sequence (prompt + response, padded to total length)
            "attention_mask": final_attention_mask,
            "position_ids": final_position_ids,
        }
        output_td = TensorDict(output_batch_dict, batch_size=loop_batch_size)

        output_non_tensor_batch = prompts.non_tensor_batch.copy() if prompts.non_tensor_batch else {}
        output_non_tensor_batch["generation_params"] = {
            **gen_params_base, 
            "max_new_tokens_total": max_new_tokens_total, # The original overall response length from prompts.meta_info or config
            "max_new_tokens_per_turn": self.max_new_tokens_per_turn, # The configured value from agent_search_config
            "max_turns": self.max_turns,
            "input_num_return_sequences": num_return_sequences,
            "do_search_used": self.do_search, # Log whether search was enabled for this run
            "retriever_url_used": self.retriever_url if self.do_search else None,
        }
        
        return DataProto(batch=output_td, non_tensor_batch=output_non_tensor_batch)
