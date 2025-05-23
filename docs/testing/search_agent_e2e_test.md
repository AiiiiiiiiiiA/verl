## End-to-End Test Procedure for Search Agent

This document outlines the steps to manually perform a basic end-to-end test of the search-augmented LLM agent in the veRL framework. This test uses the Qwen1.5-0.5B-Chat model and a dummy dataset.

**Goal**: To verify that the PPO training script can run with search enabled, that the agent makes calls to the retrieval server, and that basic metrics are logged. This is not a test for convergence or answer quality.

### Prerequisites

1.  **Dependencies**: Ensure all necessary Python packages are installed (including `torch`, `transformers`, `fastapi`, `uvicorn`, `requests`, `hydra-core`, `omegaconf`, and `rank-bm25` if using its full capabilities, though the mock works without it).
    ```bash
    # Make sure your environment has verl's dependencies installed
    # pip install -r requirements.txt 
    # (and any new ones like fastapi, uvicorn, requests, rank-bm25)
    ```
2.  **Codebase**: You should have the latest version of the `verl` codebase with the search agent features implemented.
3.  **Model Access**: The Qwen1.5-0.5B-Chat model (or the small model configured in the test script) should be accessible from Hugging Face Hub or a local path. You might need to be logged in via `huggingface-cli login` if it's a gated model (though Qwen1.5-0.5B-Chat is typically open).
4.  **Dummy Data**: The dummy dataset `examples/data_preprocess/dummy_search_qa_2lines.jsonl` and its formatted version `dummy_search_qa_2lines_formatted.jsonl` should be present, as created by a previous step. The test script `examples/ppo_trainer/run_test_search_agent_qwen0.5b.sh` should also be present.

### Test Steps

**Step 1: Start the Retrieval Server**

1.  Open a terminal in the root directory of the `verl` project.
2.  Run the retrieval server. For this test, it will use the `dummy_search_qa_2lines.jsonl` as its corpus (which it loads and indexes, or uses its internal dummy docs if the path is wrong).
    ```bash
    python verl/search/retrieval_server.py --corpus_path examples/data_preprocess/dummy_search_qa_2lines.jsonl --retriever_type bm25
    ```
3.  **Expected Output (Retrieval Server)**:
    *   You should see logs indicating the server has started, e.g., `INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`.
    *   Logs indicating the BM25Retriever initialized and indexed documents (either from the dummy corpus file or the internal dummy_docs). For example:
        `INFO:     Application startup complete.`
        `INFO:     verl.search.retrieval_server: Initializing retriever with type: bm25`
        `INFO:     verl.search.retrieval_server: Loaded 2 documents from examples/data_preprocess/dummy_search_qa_2lines.jsonl.`
        `INFO:     verl.search.retrieval.BM25Retriever: BM25Retriever indexed 2 documents...`
    *   Keep this terminal window open. The server needs to be running for the next step.

**Step 2: Run the Test PPO Training Script**

1.  Open a **new** terminal in the root directory of the `verl` project.
2.  Ensure your Python environment is activated if you are using one.
3.  Execute the test training script:
    ```bash
    bash examples/ppo_trainer/run_test_search_agent_qwen0.5b.sh
    ```

**Step 3: Observe Logs and Outputs**

*   **Retrieval Server Terminal**:
    *   As the training script runs, if the LLM agent generates `<search>query</search>` tags, you should see POST requests to the `/retrieve` endpoint in the retrieval server's logs. For example:
        `INFO:     127.0.0.1:XXXXX - "POST /retrieve HTTP/1.1" 200 OK`
    *   You might also see logs from the retriever itself about the queries it's processing if you added more verbose logging there.

*   **Training Script Terminal**:
    *   **Initialization**: Logs indicating the PPO trainer is initializing, loading the model (`Qwen/Qwen1.5-0.5B-Chat`), tokenizer, and dataset (`dummy_search_qa_2lines_formatted.jsonl`).
    *   **LLMGenerationManager Logs**: You should see logs from `LLMGenerationManager` if it's configured for high verbosity (e.g., "Running LLM generation loop...", "Search triggered for query: ...").
    *   **Action Statistics**: The PPO trainer logs should include metrics with the prefix `action_stats/` after each training step/iteration. For example:
        `action_stats/completed_turns_avg: X.X`
        `action_stats/num_search_actions_avg: Y.Y`
        `action_stats/num_answer_actions_avg: Z.Z`
        (The exact values will depend on the behavior of the small Qwen model with the dummy prompts).
    *   **Training Progress**: The script is configured to run for only `trainer.max_steps=2`. You should see it complete these steps and exit without crashing.
    *   Look for any error messages. The absence of crashes and the presence of search-related logs are key indicators for this basic test.

### Successful Test Indication

A successful basic end-to-end test is indicated by:
1.  Both the retrieval server and the training script start and run without critical errors.
2.  The retrieval server logs incoming requests to `/retrieve` when the agent decides to search.
3.  The training script logs `action_stats` metrics.
4.  The training script completes its configured 2 steps.

This procedure does not validate the quality of the LLM's reasoning or the relevance of search results, but it checks that the main components of the search agent framework are wired together and operational.
