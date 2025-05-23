# Search-Augmented LLM Agent in veRL

This document describes the functionality and usage of the search-augmented Large Language Model (LLM) agent integrated into the veRL framework. This feature allows LLMs to perform interleaved reasoning and search operations to answer questions or complete tasks.

## 1. Overview

The core idea is to train an LLM to act as an agent that can decide when it lacks information and needs to call an external search tool. The agent communicates its internal thoughts, search queries, and final answers using special XML-like tags. Search results are then fed back into the agent's context, allowing it to continue its reasoning process.

## 2. Key Components

### Special Tags
The agent uses the following tags in its generation:
-   `<think>...</think>`: For its internal reasoning process.
-   `<search>query</search>`: To issue a search query to an external tool.
-   `<answer>...</answer>`: To provide the final answer.
-   `<information>...</information>`: This tag is used by the system to provide search results back to the LLM.

### LLMGenerationManager
-   Located in `verl.llm_agent.generation.LLMGenerationManager`.
-   Orchestrates the multi-turn reasoning and searching process.
-   Parses the special tags from the LLM's output.
-   Calls the retrieval server when a `<search>` tag is detected.
-   Formats search results into `<information>` blocks and updates the LLM's context.
-   Tracks action statistics (e.g., number of searches, answers).

### Retrieval Server
-   Located in `verl.search.retrieval_server.py`.
-   A FastAPI application that serves search requests from the `LLMGenerationManager`.
-   Currently supports BM25 retrieval (via `verl.search.retrieval.BM25Retriever`).
-   Listens for POST requests on the `/retrieve` endpoint.

## 3. Setup

### Dependencies
Ensure the following Python packages are installed for search functionality:
-   `fastapi`: For the retrieval server.
-   `uvicorn`: To run the FastAPI server.
-   `httpx`: (Often used by TestClient, good to have if running client-side tests or a more advanced client)
-   `requests`: Used by `LLMGenerationManager` to call the search server.
-   `python-Levenshtein`: (Often a dependency for fuzzywuzzy, which can be used with BM25)
-   `rank-bm25`: For BM25 retrieval (optional, a mock is used if not found).
  ```bash
  pip install fastapi uvicorn requests rank-bm25
  ```

### Running the Retrieval Server
1.  **Build an Index (Optional but Recommended for BM25):**
    Use `verl/search/index_builder.py` to create an index from your corpus.
    ```bash
    python verl/search/index_builder.py --corpus_path /path/to/your/corpus.jsonl --index_path /path/to/save/bm25_index.pkl --retriever_type bm25
    ```
    The corpus file should be in JSONL format, with each line being a JSON object containing "title" and "text" fields.

2.  **Launch the Server:**
    ```bash
    python verl/search/retrieval_server.py --retriever_type bm25 --corpus_path /path/to/your/corpus.jsonl
    ```
    (If you built an index, the server would ideally load it; current implementation re-indexes from corpus or uses dummy data. This can be enhanced.)
    The server will start (by default on `http://127.0.0.1:8000`).

## 4. Configuration

The search agent's behavior is controlled by parameters in your Hydra configuration file (e.g., `verl/trainer/config/ppo_trainer.yaml`), under the `actor_rollout_ref.rollout.agent_search_config` section:

-   `do_search` (bool, default: `false`): Set to `true` to enable search functionality.
-   `max_turns` (int, default: `3`): Maximum number of search/reasoning turns per episode.
-   `max_new_tokens_per_turn` (int, default: `256`): Maximum tokens the LLM can generate in a single reasoning/action step.
-   `retriever`:
    -   `url` (str, default: `"http://127.0.0.1:8000/retrieve"`): URL of the retrieval server.
    -   `top_k` (int, default: `3`): Number of documents to retrieve.
-   `tags`: Defines the open/close tags for `think`, `search`, `answer`, and `information`. Defaults are provided.

## 5. Data Preparation

To train a search-augmented LLM, your input prompts need to instruct the model on how to use the special tags.

-   Use the `examples/data_preprocess/format_for_search_agent.py` script to convert your existing datasets:
    ```bash
    python examples/data_preprocess/format_for_search_agent.py \
        --input_file /path/to/your_original_data.jsonl \
        --output_file /path/to/formatted_for_search.jsonl \
        --prompt_instruction_template "Your custom instructions..."
    ```
-   The script will prepend the provided instruction template (which should explain the use of `<think>`, `<search>`, `<answer>`) to each question in your dataset.

**Example Formatted Prompt Snippet:**
```
You are a helpful assistant. Your task is to answer the given question. You can use the <search>tool_query</search> tool to find information. Reason step-by-step using <think>your thoughts</think>. When you have the final answer, present it using <answer>your_answer</answer>. Here is the question:

Question: What is the capital of France?

Assistant:
```

## 6. Training

To enable search during PPO training, modify your training script (e.g., `examples/ppo_trainer/run_deepseek7b_llm.sh`) to override the Hydra parameters:

```bash
python -m verl.trainer.main_ppo \
    # ... other parameters ...
    actor_rollout_ref.rollout.agent_search_config.do_search=true \
    actor_rollout_ref.rollout.agent_search_config.max_turns=3 \
    actor_rollout_ref.rollout.agent_search_config.retriever.url="http://127.0.0.1:8000/retrieve" \
    # ... other search params if needed ...
```
Ensure the retrieval server is running and accessible at the configured URL.

## 7. Inference

Use the `verl/trainer/main_search_inference.py` script for interactive testing:
```bash
python verl/trainer/main_search_inference.py actor_rollout_ref.model.path=/path/to/your/trained_model_checkpoint
```
This script will load the model and allow you to enter prompts. It will use the `LLMGenerationManager` to perform search-augmented generation and print the full interaction trace. Ensure your retrieval server is running.

## 8. Customization

### Implementing New Retrievers
1.  Create a new class inheriting from `verl.search.retrieval.BaseRetriever`.
2.  Implement the `search` and `index_documents` methods.
3.  Update `verl.search.retrieval_server.py` to support initializing and using your new retriever.

### Modifying Reward Functions
-   The `verl.workers.reward_manager.NaiveRewardManager` has been updated to parse the content of the `<answer>...</answer>` tag.
-   The `compute_score` function (see `verl.trainer.ppo.reward.py`) is where you would customize scoring based on the extracted answer and ground truth.

```
