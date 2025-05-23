import unittest
from unittest.mock import Mock, patch, call
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf
from collections import defaultdict

# Adjust sys.path if necessary to find verl, assuming tests are run from project root
import sys
import os
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from verl import DataProto
from verl.llm_agent.generation import LLMGenerationManager


class TestLLMGenerationManager(unittest.TestCase):

    def setUp(self):
        # Mock Model
        self.mock_model = Mock()
        # The generate method will be configured per test or with side_effect
        self.mock_model.generate = Mock()
        self.mock_model.eval = Mock() # Ensure eval can be called

        # Mock Tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.eos_token_id = 0
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.model_max_length = 512
        # encode and decode will be configured per test or with side_effect
        self.mock_tokenizer.encode = Mock()
        self.mock_tokenizer.decode = Mock()
        self.mock_tokenizer.batch_decode = Mock(side_effect=lambda x, **kwargs: [self.mock_tokenizer.decode(s, **kwargs) for s in x])


        # Mock Hydra Config
        # Using a more complete config structure based on ppo_trainer.yaml
        # to ensure all paths accessed by LLMGenerationManager are present.
        self.base_config_dict = {
            "actor_rollout_ref": {
                "rollout": {
                    "agent_search_config": {
                        "max_turns": 3,
                        "do_search": True,
                        "retriever": {"url": "http://mockretriever/retrieve", "top_k": 2},
                        "max_new_tokens_per_turn": 50,
                        "tags": {
                            "think_tag_open": "<think>", "think_tag_close": "</think>",
                            "search_tag_open": "<search>", "search_tag_close": "</search>",
                            "answer_tag_open": "<answer>", "answer_tag_close": "</answer>",
                            "information_tag_open": "<information>", "information_tag_close": "</information>",
                        }
                    },
                    # Other rollout params LLMGenerationManager might access via self.config
                    "do_sample": True, 
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "response_length": 150, # Total response length over turns
                },
                "model": { # For things like model_max_length if not on tokenizer
                    "path": "mock/path", # Required by LLMGenerationManager if it tries to read it
                    "eos_token_id": 0, # if LLMGenerationManager accesses self.module.config.eos_token_id
                    "pad_token_id": 0,
                    "max_position_embeddings": 512 # if LLMGenerationManager uses this
                }
            },
            "data": { # For things like num_return_sequences from self.config.n
                "n": 1, # num_return_sequences
                "reward_fn_key": "reward_source" # Just a placeholder
            }
        }
        self.mock_config = OmegaConf.create(self.base_config_dict)
        
        # LLMGenerationManager might access module.config for token ids if tokenizer doesn't have them
        self.mock_model.config = Mock(
            eos_token_id=self.mock_tokenizer.eos_token_id, 
            pad_token_id=self.mock_tokenizer.pad_token_id,
            max_position_embeddings=self.mock_tokenizer.model_max_length
        )


        # Instantiate LLMGenerationManager
        self.manager = LLMGenerationManager(
            module=self.mock_model,
            tokenizer=self.mock_tokenizer,
            config=self.mock_config
        )
        self.manager.model_max_length = 512 # Ensure this is set for tests

    def _create_sample_input_dp(self, prompt_text="Initial prompt"):
        # Tokenize the prompt
        # Mock tokenizer's __call__ or encode behavior for this
        mock_prompt_tokens = torch.tensor([[101, 2000, 2002, 102]], dtype=torch.long) # Example: [CLS] hello world [SEP]
        self.mock_tokenizer.encode.return_value = mock_prompt_tokens[0] # for internal use
        
        input_ids = mock_prompt_tokens
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(0, input_ids.shape[1]).unsqueeze(0)

        non_tensor_batch_item = {"id": "test_0", "reward_model": {"ground_truth": ""}, self.mock_config.data.reward_fn_key: "test_source"}
        
        meta_info = {
            "response_length": self.mock_config.actor_rollout_ref.rollout.response_length,
            "do_sample": self.mock_config.actor_rollout_ref.rollout.do_sample,
            "temperature": self.mock_config.actor_rollout_ref.rollout.temperature,
            "eos_token_id": self.mock_tokenizer.eos_token_id,
            "pad_token_id": self.mock_tokenizer.pad_token_id,
            "num_return_sequences": self.mock_config.data.n
        }

        batch_td = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }, batch_size=1)

        return DataProto(batch=batch_td, non_tensor_batch=[non_tensor_batch_item], meta_info=meta_info)

    def test_parse_search_action(self):
        text = "<search>some query</search> rest of text"
        parsed = self.manager._parse_llm_action(text)
        self.assertEqual(parsed["action_type"], "search")
        self.assertEqual(parsed["query"], "some query")
        self.assertEqual(parsed["raw_text"], "<search>some query</search>")

    def test_parse_answer_action(self):
        text = "<answer>final answer</answer>"
        parsed = self.manager._parse_llm_action(text)
        self.assertEqual(parsed["action_type"], "answer")
        self.assertEqual(parsed["answer"], "final answer")
        self.assertEqual(parsed["raw_text"], "<answer>final answer</answer>")

    def test_parse_think_action(self):
        text = "<think>let me think</think> then something else"
        parsed = self.manager._parse_llm_action(text)
        self.assertEqual(parsed["action_type"], "think")
        self.assertEqual(parsed["thought"], "let me think")
        self.assertEqual(parsed["raw_text"], "<think>let me think</think>")

    def test_parse_malformed_tag_or_content(self):
        # Test with incomplete tag
        text_incomplete = "<search>incomplete"
        parsed_incomplete = self.manager._parse_llm_action(text_incomplete)
        self.assertEqual(parsed_incomplete["action_type"], "content") # Expect "content" as per current impl
        self.assertEqual(parsed_incomplete["text"], text_incomplete)
        self.assertEqual(parsed_incomplete["raw_text"], text_incomplete)

        # Test with plain content
        text_content = "just some text without tags"
        parsed_content = self.manager._parse_llm_action(text_content)
        self.assertEqual(parsed_content["action_type"], "content")
        self.assertEqual(parsed_content["text"], text_content)

    def test_run_llm_loop_max_turns(self):
        sample_input_dp = self._create_sample_input_dp()
        
        # Mock model.generate to always return "thinking" tokens
        # Assume input prompt is [101, 2000, 2002, 102] (length 4)
        # Each turn generates, e.g., 3 new tokens for "<think>...</think>"
        # Let's say "<think>...</think>" encodes to [300, 301, 302]
        
        # Turn 1: input_len=4, output_len=4+3=7. Generated: [300,301,302]
        # Turn 2: input_len=7, output_len=7+3=10. Generated: [300,301,302]
        # Turn 3: input_len=10, output_len=10+3=13. Generated: [300,301,302]

        def generate_side_effect(input_ids, attention_mask, generation_config):
            # input_ids is the current full sequence for the batch
            # generation_config.max_new_tokens is crucial here
            new_think_tokens = torch.tensor([[300, 301, 302]], dtype=torch.long) # Mock tokens for "<think>...</think>"
            # Append new_think_tokens to each sequence in input_ids
            # We need to respect max_new_tokens for this turn. Let's assume it's <= 3
            num_new = min(new_think_tokens.size(1), generation_config.max_new_tokens)
            return torch.cat([input_ids, new_think_tokens[:, :num_new]], dim=1)

        self.mock_model.generate.side_effect = generate_side_effect
        self.mock_tokenizer.decode.return_value = "<think>thinking</think>" # What decode returns for [300,301,302]
        self.mock_tokenizer.encode.return_value = torch.tensor([300,301,302], dtype=torch.long) # For re-encoding parsed action

        result_dp = self.manager.run_llm_loop(sample_input_dp)
        
        action_stats = result_dp.non_tensor_batch[0]["action_stats"]
        self.assertEqual(action_stats["completed_turns"], self.manager.max_turns)
        self.assertEqual(action_stats["num_think_actions"], self.manager.max_turns)
        self.assertEqual(action_stats["num_answer_actions"], 0)

    def test_run_llm_loop_finds_answer(self):
        sample_input_dp = self._create_sample_input_dp()

        # Turn 1: Generate think tokens
        # Turn 2: Generate answer tokens
        think_tokens = torch.tensor([[300, 301, 302]], dtype=torch.long) # <think>...</think>
        answer_tokens = torch.tensor([[400, 401, 402]], dtype=torch.long) # <answer>...</answer>

        # Mock responses for self.mock_model.generate
        # First call (turn 1): appends think_tokens
        # Second call (turn 2): appends answer_tokens
        def generate_side_effect_answer(input_ids, attention_mask, generation_config):
            if input_ids.size(1) == 4: # Initial prompt length
                 return torch.cat([input_ids, think_tokens], dim=1)
            elif input_ids.size(1) == 4 + think_tokens.size(1): # After first think
                 return torch.cat([input_ids, answer_tokens], dim=1)
            return input_ids # Should not happen in this test

        self.mock_model.generate.side_effect = generate_side_effect_answer
        
        # Mock tokenizer.decode for different token sequences
        def decode_side_effect(token_ids, skip_special_tokens=True):
            if torch.equal(token_ids, think_tokens[0]): return "<think>thinking</think>"
            if torch.equal(token_ids, answer_tokens[0]): return "<answer>found it</answer>"
            return "unknown tokens" # Fallback for other parts if any

        self.mock_tokenizer.decode.side_effect = decode_side_effect
        
        # Mock tokenizer.encode for re-encoding the parsed action string
        self.mock_tokenizer.encode.side_effect = lambda text, **kwargs: {
            "<think>thinking</think>": think_tokens[0],
            "<answer>found it</answer>": answer_tokens[0],
        }.get(text, torch.tensor([999])) # 999 for unknown

        result_dp = self.manager.run_llm_loop(sample_input_dp)
        
        action_stats = result_dp.non_tensor_batch[0]["action_stats"]
        self.assertEqual(action_stats["num_answer_actions"], 1)
        self.assertEqual(action_stats["completed_turns"], 2) # Answer found on turn 2
        self.assertEqual(action_stats["num_think_actions"], 1)


    @patch('requests.post')
    def test_run_llm_loop_with_search(self, mock_post):
        sample_input_dp = self._create_sample_input_dp()

        # Configure mock_post for search result
        mock_search_response = Mock()
        mock_search_response.json.return_value = {"results": [{"title": "Test Doc", "text": "Content of test doc"}]}
        mock_search_response.raise_for_status = Mock()
        mock_post.return_value = mock_search_response

        # Define tokens for each part of the interaction
        search_action_tokens = torch.tensor([[500, 501, 502]], dtype=torch.long) # <search>query</search>
        # "<information>Document 1 (Title: Test Doc): Content of test doc</information>"
        # This will be long, let's mock its tokenization simply
        info_tokens = torch.tensor([[600, 601, 602, 603, 604]], dtype=torch.long) 
        answer_action_tokens = torch.tensor([[400, 401, 402]], dtype=torch.long) # <answer>...</answer>
        
        # Mock model.generate side effect
        # 1. Input initial prompt -> Output search_action_tokens
        # 2. Input initial_prompt + search_action_tokens + info_tokens -> Output answer_action_tokens
        def generate_side_effect_search(input_ids, attention_mask, generation_config):
            if input_ids.size(1) == 4: # Initial prompt
                return torch.cat([input_ids, search_action_tokens], dim=1)
            # Check if input_ids contains the info_tokens (approximate check)
            # A more robust check would be to verify the exact sequence if possible
            elif 600 in input_ids[0].tolist(): # If info_tokens (mocked as containing 600) are present
                return torch.cat([input_ids, answer_action_tokens], dim=1)
            else: # Should not be reached if logic is correct
                return input_ids 

        self.mock_model.generate.side_effect = generate_side_effect_search

        # Mock tokenizer.decode
        def decode_side_effect(token_ids, skip_special_tokens=True):
            # This decode is for the *newly generated part only* by the agent logic
            if torch.equal(token_ids, search_action_tokens[0]): return "<search>test search</search>"
            if torch.equal(token_ids, answer_action_tokens[0]): return "<answer>answered after search</answer>"
            # This decode is also used by the test itself to check full output, so handle that too
            if len(token_ids) > 10 : return "<search>test search</search><information>...</information><answer>answered after search</answer>"
            return "unknown_sequence_for_decode"

        self.mock_tokenizer.decode.side_effect = decode_side_effect
        
        # Mock tokenizer.encode (for re-encoding parsed text and info_text)
        def encode_side_effect(text, add_special_tokens=False, return_tensors=None):
            if text == "<search>test search</search>": return search_action_tokens[0]
            if text == f"{self.manager.tags['information_tag_open']}\nDocument 1 (Title: Test Doc): Content of test doc\n{self.manager.tags['information_tag_close']}": return info_tokens[0]
            if text == "<answer>answered after search</answer>": return answer_action_tokens[0]
            return torch.tensor([999], dtype=torch.long) # Unknown text

        self.mock_tokenizer.encode.side_effect = encode_side_effect

        result_dp = self.manager.run_llm_loop(sample_input_dp)
        
        action_stats = result_dp.non_tensor_batch[0]["action_stats"]
        
        # Assert requests.post was called
        mock_post.assert_called_once_with(
            self.manager.retriever_url,
            json={"query": "test search", "top_k": self.manager.retriever_top_k},
            timeout=10
        )
        
        self.assertEqual(action_stats["num_search_actions"], 1)
        self.assertEqual(action_stats["num_answer_actions"], 1)
        self.assertEqual(action_stats["completed_turns"], 2) # Search (turn 1) + Answer (turn 2)

        # Verify the information block was added for the second LLM call
        # The second call to generate should have input_ids = initial_prompt + search_action_tokens + info_tokens
        # Expected sequence for second call:
        # sample_input_dp.batch["input_ids"] (len 4)
        # search_action_tokens (len 3)
        # info_tokens (len 5)
        # Total expected length = 4 + 3 + 5 = 12
        
        # Check the arguments of the second call to mock_model.generate
        # The first call is with input_ids of length 4
        # The second call is with input_ids of length 4 + search_action_tokens_len + info_tokens_len
        # generate_args_list = self.mock_model.generate.call_args_list
        # self.assertEqual(len(generate_args_list), 2) # Should be called twice
        
        # First call's input_ids
        # first_call_args = generate_args_list[0][1] # [0] for args, [1] for kwargs
        # self.assertEqual(first_call_args['input_ids'].size(1), 4) 
        
        # Second call's input_ids
        # second_call_args = generate_args_list[1][1]
        # expected_input_for_second_call_len = sample_input_dp.batch["input_ids"].size(1) + \
        #                                    search_action_tokens.size(1) + \
        #                                    info_tokens.size(1)
        # self.assertEqual(second_call_args['input_ids'].size(1), expected_input_for_second_call_len)
        # A more robust check would be to ensure the info_tokens are actually part of the input_ids for the second call.
        # This depends on how precisely we mock tokenization of the info_text.
        # For now, checking call count and stats is primary.


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

```
I've created the content for `tests/llm_agent/test_generation.py`.

Key aspects of this implementation:
-   **Imports**: All necessary modules are imported.
-   **`setUp` Method**:
    -   Mocks for `model`, `tokenizer`, and `config` (using `OmegaConf`) are initialized.
    -   `LLMGenerationManager` is instantiated with these mocks.
    -   A helper `_create_sample_input_dp` is defined to generate basic `DataProto` inputs for tests.
-   **Test Methods for `_parse_llm_action`**:
    -   `test_parse_search_action`, `test_parse_answer_action`, `test_parse_think_action` check correct parsing of valid tags.
    -   `test_parse_malformed_tag_or_content` checks that incomplete tags or plain text are handled as "content" (as per current `_parse_llm_action` logic).
-   **Test Methods for `run_llm_loop`**:
    -   `test_run_llm_loop_max_turns`: Mocks `generate` to always return "think" actions, asserting that the loop runs for `max_turns` and logs think actions.
    -   `test_run_llm_loop_finds_answer`: Mocks `generate` to return an answer on the second turn, asserting correct turn count and answer action logging.
    -   `test_run_llm_loop_with_search`: This is the most complex test.
        -   It uses `@patch('requests.post')` to mock the HTTP search call.
        -   It configures `mock_model.generate` with a `side_effect` to simulate a multi-turn interaction: first generating a search query, then, after the (mocked) search results are incorporated, generating an answer.
        -   It configures `mock_tokenizer.decode` and `mock_tokenizer.encode` with `side_effect` to handle different token sequences and text segments that appear during this multi-turn process.
        -   It asserts that `requests.post` was called correctly.
        -   It asserts that search and answer actions were logged in stats.
        -   Comments are included for how one might verify the content of arguments to `mock_model.generate` more deeply, which can be tricky with complex token sequences.
-   **Execution**: `if __name__ == '__main__': unittest.main(...)` is included.

This test suite provides a good starting point for verifying the core functionality of `LLMGenerationManager`, especially its action parsing and multi-turn loop logic with search integration.The file `tests/llm_agent/test_generation.py` has been successfully created with the specified unit tests for `verl.llm_agent.generation.LLMGenerationManager`.

Summary of the implementation:
1.  **Imports**: Includes `unittest`, `unittest.mock.Mock`, `unittest.mock.patch`, `torch`, `tensordict.TensorDict`, `verl.DataProto`, `verl.llm_agent.generation.LLMGenerationManager`, `omegaconf.OmegaConf`, and `collections.defaultdict`. `sys.path` is adjusted to allow finding the `verl` package.
2.  **`TestLLMGenerationManager(unittest.TestCase)` Class**:
    *   **`setUp` Method**:
        *   Initializes `self.mock_model` with a mocked `generate` method and `eval` method.
        *   Initializes `self.mock_tokenizer` with mocked `encode`, `decode`, and `batch_decode` methods, and sets `eos_token_id`, `pad_token_id`, and `model_max_length`.
        *   Creates `self.mock_config` using `OmegaConf.create` to simulate the Hydra configuration structure, including `actor_rollout_ref.rollout.agent_search_config` with all specified sub-fields (`max_turns`, `do_search`, `retriever` details, `max_new_tokens_per_turn`, and `tags`). It also includes minimal fields under `model` and `data` that `LLMGenerationManager` might access.
        *   The `mock_model.config` attribute is also mocked to provide token IDs and max length if the manager tries to access them there.
        *   Instantiates `self.manager = LLMGenerationManager(...)` with these mocks.
        *   A helper method `_create_sample_input_dp` is defined to create `DataProto` objects for test inputs, including tokenized prompts and relevant `meta_info`.
    *   **Test Methods for `_parse_llm_action`**:
        *   `test_parse_search_action`, `test_parse_answer_action`, `test_parse_think_action`: These test the correct parsing of well-formed search, answer, and think tags, asserting the returned dictionary's `action_type`, content (`query`, `answer`, `thought`), and `raw_text`.
        *   `test_parse_malformed_tag_or_content`: Tests that incomplete tags or plain text without action tags are parsed as `{"action_type": "content", ...}`.
    *   **Test Methods for `run_llm_loop`**:
        *   `test_run_llm_loop_max_turns`: Configures `mock_model.generate` and `mock_tokenizer.decode` (using `side_effect`) to simulate an agent that only "thinks". It asserts that the loop runs for the configured `max_turns` and that the action statistics reflect this.
        *   `test_run_llm_loop_finds_answer`: Configures mocks to simulate finding an answer on the second turn. Asserts that `num_answer_actions` is 1 and `completed_turns` is 2.
        *   `test_run_llm_loop_with_search`:
            *   Uses `@patch('requests.post')` to mock the external HTTP call for search.
            *   The `mock_model.generate` and `mock_tokenizer.decode`/`encode` methods are configured with `side_effect` to simulate a multi-step process: (1) LLM generates a search query, (2) mock `requests.post` returns dummy search results, (3) LLM incorporates these results and generates an answer.
            *   Asserts that `mock_post` was called with the correct URL and payload.
            *   Asserts that the action statistics correctly record one search and one answer action.
            *   Includes comments on how further assertions could be made about the arguments to `mock_model.generate` to verify the information flow.
3.  **Execution Block**: `if __name__ == '__main__': unittest.main(argv=['first-arg-is-ignored'], exit=False)` is included to allow running the tests.

This test suite covers various aspects of the `LLMGenerationManager`, including action parsing, turn limits, answer detection, and the search integration (mocked).
