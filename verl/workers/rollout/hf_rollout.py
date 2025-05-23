# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.llm_agent.generation import LLMGenerationManager # Added import
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["HFRollout"]


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module
        # TODO: Ensure tokenizer is correctly provisioned.
        # Using module.tokenizer if available, otherwise None.
        # This might need to be passed explicitly or handled by a central tokenizer registry.
        tokenizer = getattr(module, "tokenizer", None)
        self.llm_generation_manager = LLMGenerationManager(
            module=self.module,
            tokenizer=tokenizer,
            config=self.config
        )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad() # Kept no_grad as generation shouldn't require gradients here
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # All generation logic, including kwargs, GenerationConfig, model.generate call,
        # and output processing (padding, TensorDict creation) is now expected
        # to be handled by self.llm_generation_manager.run_llm_loop.

        # The HFRollout specific contexts like FSDP.summon_full_params and autocast
        # are assumed to be either handled within LLMGenerationManager if they are general enough,
        # or LLMGenerationManager is designed to work correctly without them if they are
        # HFRollout specific optimizations that can be applied before/after the manager call.
        # For now, we assume LLMGenerationManager encapsulates these details or the subtask
        # implies they are part of the "generation logic" moved to the manager.

        # The self.module.eval() and self.module.train() calls:
        # LLMGenerationManager's run_llm_loop uses torch.no_grad() and it's assumed
        # it sets the model to eval mode if necessary.
        # If HFRollout needs to manage the model's train/eval state explicitly
        # around the call to a generic manager, that could be added here.
        # For now, assuming the manager handles its model's state.
        # self.module.eval() # Potentially handled by manager or not needed if manager uses a separate model instance

        output_data_proto = self.llm_generation_manager.run_llm_loop(prompts)

        # self.module.train() # Restore train state if changed by manager or above call

        # empty cache before compute old_log_prob (if this was specific to this point)
        # This might still be relevant if the manager's operations filled the cache.
        torch.cuda.empty_cache()

        return output_data_proto
