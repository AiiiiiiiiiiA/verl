import unittest
from unittest.mock import patch, MagicMock, ANY
from fastapi.testclient import TestClient
import argparse # For argparse.Namespace
import sys
import os

# Adjust sys.path if necessary to find verl
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# Import after sys.path adjustment
from verl.search.retrieval_server import app # FastAPI app instance
from verl.search.retrieval import BM25Retriever, BaseRetriever # For type hinting or spec
# Pydantic models are not explicitly used for asserts here but good for reference
# from verl.search.retrieval_server import SearchResponse, Document 

# Global mock for retriever instance for some tests if needed, though patching args is preferred
# mock_retriever_instance = None

class TestRetrievalServer(unittest.TestCase):

    # Note: setUp is called before each test. TestClient(app) will trigger startup.
    # So, if a test needs specific startup args, it must patch *before* TestClient is created.
    # For simplicity here, we might re-create TestClient in methods needing specific startup.

    def test_retrieve_endpoint_bm25_success(self):
        # Patch argparse for this specific test's startup sequence
        with patch('verl.search.retrieval_server.argparse.ArgumentParser.parse_args') as mock_parse_args:
            mock_parse_args.return_value = argparse.Namespace(
                retriever_type="bm25", 
                corpus_path=None, # This will make it use dummy_docs
                index_path=None
            )
            
            # Instantiate client *inside* the patch context so startup uses the mocked args
            client = TestClient(app) 
            
            response = client.post("/retrieve", json={"query": "world", "top_k": 1})
            
            self.assertEqual(response.status_code, 200)
            response_data = response.json()
            self.assertIn("results", response_data)
            self.assertIsInstance(response_data["results"], list)
            self.assertEqual(len(response_data["results"]), 1)
            if response_data["results"]:
                self.assertIn("title", response_data["results"][0])
                self.assertIn("text", response_data["results"][0])
                # Check if content matches dummy_docs behavior (dummy doc 1 contains "world")
                self.assertEqual(response_data["results"][0]["title"], "Dummy Doc 1")

    def test_retrieve_endpoint_empty_query(self):
        with patch('verl.search.retrieval_server.argparse.ArgumentParser.parse_args') as mock_parse_args:
            mock_parse_args.return_value = argparse.Namespace(
                retriever_type="bm25", corpus_path=None, index_path=None
            )
            client = TestClient(app)
            
            response = client.post("/retrieve", json={"query": "", "top_k": 1})
            
            self.assertEqual(response.status_code, 200)
            response_data = response.json()
            self.assertIn("results", response_data)
            self.assertEqual(len(response_data["results"]), 0) # BM25Retriever returns empty for empty query

    def test_retrieve_endpoint_invalid_request_body(self):
        # For this test, default startup is fine as error is pre-retriever
        client = TestClient(app) # Uses default startup (bm25 with dummy docs)
        
        response = client.post("/retrieve", json={"bad_query_key": "test query"})
        self.assertEqual(response.status_code, 422) # FastAPI validation error

    @patch('verl.search.retrieval_server.DenseRetriever') # Patch the class itself
    def test_startup_event_dense_retriever(self, mock_dense_retriever_cls):
        # Configure the mock class to return a mock instance
        mock_dense_instance = MagicMock(spec=BaseRetriever) # Use BaseRetriever for spec
        mock_dense_instance.search.return_value = [] # Dense placeholder returns empty list
        mock_dense_retriever_cls.return_value = mock_dense_instance

        with patch('verl.search.retrieval_server.argparse.ArgumentParser.parse_args') as mock_parse_args:
            mock_parse_args.return_value = argparse.Namespace(
                retriever_type="dense", 
                corpus_path=None, 
                index_path="mock_index_path" # Pass an index_path
            )
            
            # TestClient instantiation triggers startup
            # The app's global `retriever` should become our mock_dense_instance
            client = TestClient(app) 
            
            # Check that DenseRetriever was instantiated during startup
            # It's called with corpus_or_index_path="mock_index_path"
            mock_dense_retriever_cls.assert_called_once_with(corpus_or_index_path="mock_index_path")
            
            # Make a call to ensure it uses the mocked DenseRetriever
            response = client.post("/retrieve", json={"query": "test dense query", "top_k": 1})
            self.assertEqual(response.status_code, 200)
            
            # Assert that the search method of our mock_dense_instance was called
            mock_dense_instance.search.assert_called_once_with(query="test dense query", top_k=1)
            
            response_data = response.json()
            self.assertIn("results", response_data)
            self.assertEqual(response_data["results"], []) # DenseRetriever placeholder returns empty

    def test_retrieve_endpoint_retriever_not_initialized(self):
        # This test aims to simulate a scenario where the retriever failed to initialize.
        # We can achieve this by making argparse return an invalid retriever_type,
        # so the global `retriever` remains None.
        with patch('verl.search.retrieval_server.argparse.ArgumentParser.parse_args') as mock_parse_args:
            mock_parse_args.return_value = argparse.Namespace(
                retriever_type="invalid_type", # This type will cause retriever to be None
                corpus_path=None, 
                index_path=None
            )
            
            client = TestClient(app)
            
            response = client.post("/retrieve", json={"query": "test query", "top_k": 1})
            self.assertEqual(response.status_code, 503) # Service unavailable
            self.assertIn("Retriever service not available", response.json()["detail"])


if __name__ == '__main__':
    # Important: If run directly, FastAPI/Uvicorn startup logic might behave differently
    # regarding argparse if not managed carefully. These tests rely on TestClient's behavior.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

```

I've created the content for `tests/search/test_retrieval_server.py`.

Key aspects of this implementation:
-   **Imports**: All specified modules are imported. `sys.path` is adjusted for local project imports.
-   **`TestRetrievalServer(unittest.TestCase)`**:
    -   **No global `setUp` for `TestClient`**: `TestClient(app)` is instantiated within each test method *after* relevant patching for startup arguments is complete. This is crucial because the server's startup event, which uses `argparse`, is triggered when `TestClient(app)` is initialized.
    -   **`test_retrieve_endpoint_bm25_success`**:
        -   Uses `@patch('verl.search.retrieval_server.argparse.ArgumentParser.parse_args')` to control startup. The mock returns `retriever_type="bm25"` and `corpus_path=None`, forcing the server to use the dummy BM25 data.
        -   Makes a POST request and asserts a 200 status, correct response structure, and that one of the dummy documents is returned for the query "world".
    -   **`test_retrieve_endpoint_empty_query`**:
        -   Similar setup for BM25 with dummy data.
        -   Makes a POST request with an empty query. Asserts 200 status and that the "results" list is empty, which is the expected behavior for BM25Retriever with an empty query.
    -   **`test_retrieve_endpoint_invalid_request_body`**:
        -   Uses a default `TestClient(app)` (which will start with BM25 and dummy data by default if `argparse` in startup gets no args).
        -   Makes a POST request with an invalid JSON body (wrong key).
        *   Asserts a 422 status code (FastAPI's unprocessable entity error).
    -   **`test_startup_event_dense_retriever`**:
        -   Patches `argparse.ArgumentParser.parse_args` to simulate selecting the "dense" retriever type and providing a mock `index_path`.
        -   Patches `verl.search.retrieval_server.DenseRetriever` to get a `mock_dense_retriever_cls`. This mock class is configured to return a `mock_dense_instance` (which itself is a `MagicMock` spec'd as `BaseRetriever`).
        -   Instantiates `TestClient(app)`.
        -   Asserts that `mock_dense_retriever_cls` was called once during startup with the expected `corpus_or_index_path`.
        -   Makes a POST request to `/retrieve`. Asserts a 200 status, that the `mock_dense_instance.search` method was called with correct arguments, and that the results are empty (as per `DenseRetriever` placeholder behavior).
    -   **`test_retrieve_endpoint_retriever_not_initialized`**:
        -   Patches `argparse.ArgumentParser.parse_args` to return an invalid `retriever_type`. This ensures the global `retriever` variable in the server remains `None`.
        -   Makes a POST request. Asserts a 503 status code and checks the detail message.
-   **Execution Block**: `if __name__ == '__main__': unittest.main(...)` is included.

This test suite covers key aspects of the server, focusing on the startup configuration via `argparse` and basic endpoint functionality.The file `tests/search/test_retrieval_server.py` has been successfully created with the specified unit tests for the FastAPI retrieval server.

Summary of the implementation:
1.  **Imports**: Includes `unittest`, `unittest.mock.patch`, `unittest.mock.MagicMock`, `fastapi.testclient.TestClient`, `argparse.Namespace`, and necessary components from the `verl.search` module (`app`, `BM25Retriever`, `BaseRetriever`). `sys` and `os` are used for path adjustments to ensure local project imports work.
2.  **`TestRetrievalServer(unittest.TestCase)` Class**:
    *   **General Approach to Startup**: The tests acknowledge that `TestClient(app)` triggers FastAPI's startup events. Therefore, to control the retriever initialization (which happens during startup via `argparse`), `argparse.ArgumentParser.parse_args` is patched *within* each test method context *before* `TestClient(app)` is instantiated for that test.
    *   **`test_retrieve_endpoint_bm25_success(self)`**:
        *   Patches `parse_args` to return a namespace configuring the "bm25" retriever with `corpus_path=None` (to use the server's internal dummy documents).
        *   Instantiates `TestClient(app)` within this patched context.
        *   Makes a POST request to `/retrieve` with a query ("world") expected to match dummy data.
        *   Asserts a 200 status code, correct response structure (`{"results": [...]}`), and that the content of the result matches the expected dummy document.
    *   **`test_retrieve_endpoint_empty_query(self)`**:
        *   Similar BM25 setup with dummy data.
        *   Makes a POST request with an empty query string.
        *   Asserts a 200 status code and that the "results" list is empty, as `BM25Retriever` (and its mock) should return no results for an empty query.
    *   **`test_retrieve_endpoint_invalid_request_body(self)`**:
        *   Uses `TestClient(app)` (which will default to BM25 with dummy data if startup args are not otherwise mocked).
        *   Makes a POST request with an invalid JSON payload (e.g., missing "query" key or using a wrong key).
        *   Asserts a 422 status code, which is FastAPI's response for unprocessable entity/validation errors.
    *   **`test_startup_event_dense_retriever(self)`**:
        *   Patches `parse_args` to return a namespace configuring the "dense" retriever and providing a mock `index_path`.
        *   Also patches `verl.search.retrieval_server.DenseRetriever` (the class imported in the server file) to get a `mock_dense_retriever_cls`. This mock class is set up to return a `mock_dense_instance` (a `MagicMock` based on `BaseRetriever`).
        *   Instantiates `TestClient(app)`.
        *   Asserts that `mock_dense_retriever_cls` was called once with `corpus_or_index_path="mock_index_path"`, verifying that the server attempted to initialize the dense retriever as configured.
        *   Makes a POST request to `/retrieve`. Asserts a 200 status, that the `mock_dense_instance.search` method was called with the correct query and top_k, and that the results are empty (as per the placeholder `DenseRetriever`'s behavior).
    *   **`test_retrieve_endpoint_retriever_not_initialized(self)`**:
        *   Patches `parse_args` to return an invalid `retriever_type`. This ensures the global `retriever` in the server code remains `None`.
        *   Instantiates `TestClient(app)`.
        *   Makes a POST request to `/retrieve`.
        *   Asserts a 503 status code ("Service Unavailable") and checks that the error detail message indicates the retriever is not available.
3.  **Execution Block**: `if __name__ == '__main__': unittest.main(argv=['first-arg-is-ignored'], exit=False)` is included.

This test suite effectively tests the server's core endpoint, its startup logic with argument parsing (via mocking), and error handling for invalid requests and uninitialized retriever scenarios.
