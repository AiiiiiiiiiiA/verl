import argparse
import json
import logging
import os
import pickle

# To handle running this script directly for a project not yet "installed"
# or when the `verl` package is not in the default Python path.
try:
    from verl.search.retrieval import BM25Retriever, DenseRetriever, FAISS_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE
except ModuleNotFoundError:
    # This allows the script to be run with `python verl/search/index_builder.py`
    # from the project root, by temporarily adding the project root to sys.path.
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from verl.search.retrieval import BM25Retriever, DenseRetriever, FAISS_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Build an index for a retriever from a corpus.")
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help="Path to the input JSONL corpus file (e.g., data.jsonl)."
    )
    parser.add_argument(
        "--index_path",
        type=str,
        required=True,
        help="Path to save the built BM25 index (e.g., bm25_index.pkl)."
    )
    parser.add_argument(
        "--retriever_type",
        type=str,
        default="bm25",
        choices=["bm25", "dense_e5_faiss"], 
        help="Type of retriever to build an index for (default: bm25). 'bm25' saves a single .pkl file. 'dense_e5_faiss' saves index components into the specified directory."
    )
    args = parser.parse_args()

    logger.info(f"Starting index building process for retriever type: {args.retriever_type}")

    # --- Load Documents ---
    if not os.path.exists(args.corpus_path):
        logger.error(f"Corpus path does not exist: {args.corpus_path}")
        return # Exit if corpus not found

    loaded_documents = []
    try:
        with open(args.corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    doc = json.loads(line)
                    if "title" in doc and "text" in doc:
                        loaded_documents.append(doc)
                    else:
                        logger.warning(f"Skipping line {i+1} due to missing 'title' or 'text': {line.strip()}")
                except json.JSONDecodeError:
                    logger.warning(f"Skipping line {i+1} due to JSON decode error: {line.strip()}")
        
        if not loaded_documents:
            logger.error(f"No valid documents loaded from {args.corpus_path}. Exiting.")
            return
        logger.info(f"Successfully loaded {len(loaded_documents)} documents from {args.corpus_path}.")

    except Exception as e:
        logger.error(f"Failed to read or parse corpus file {args.corpus_path}: {e}")
        return

    # --- Index Building ---
    if args.retriever_type == "bm25":
        logger.info("Initializing BM25Retriever...")
        # Corpus not passed here, will be indexed via index_documents method call
        bm25_retriever = BM25Retriever()

        logger.info("Starting document indexing for BM25...")
        bm25_retriever.index_documents(loaded_documents)
        logger.info("BM25 document indexing complete.")

        # Ensure the BM25 model object exists after indexing.
        # bm25_retriever.bm25 is the BM25Okapi (or mock) object.
        if bm25_retriever.bm25 is None:
            logger.error("BM25 model (bm25_retriever.bm25) is None after indexing. Cannot save index.")
            return

        # Data to be pickled: the BM25Okapi object and the list of original documents.
        index_data_to_save = {
            "bm25_object": bm25_retriever.bm25,
            "documents": bm25_retriever.documents
        }

        try:
            # Ensure the directory for the index_path exists if it's a file path
            index_dir = os.path.dirname(args.index_path)
            if index_dir and not os.path.exists(index_dir): # Check if index_dir is not empty string
                os.makedirs(index_dir, exist_ok=True)
                logger.info(f"Created directory for BM25 index file: {index_dir}")

            with open(args.index_path, "wb") as f_out:
                pickle.dump(index_data_to_save, f_out)
            logger.info(f"BM25 index successfully saved to: {args.index_path}")
        except Exception as e:
            logger.error(f"Failed to save BM25 index to {args.index_path}: {e}")

    elif args.retriever_type == "dense_e5_faiss":
        logger.info("Selected retriever type: dense_e5_faiss.")
        if not FAISS_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error(
                "DenseRetriever dependencies (FAISS or sentence-transformers) are not installed. "
                "Please install them to use the 'dense_e5_faiss' retriever type. "
                "E.g., pip install faiss-cpu sentence-transformers"
            )
            return # Or sys.exit(1)

        logger.info("Initializing DenseRetriever (default model: intfloat/e5-large-v2)...")
        try:
            dense_retriever = DenseRetriever() # Uses default model
        except ImportError as e: # Catch import error during instantiation if deps were somehow missed by flags
            logger.error(f"Failed to initialize DenseRetriever: {e}")
            return

        logger.info("Starting document indexing for dense retrieval...")
        dense_retriever.index_documents(loaded_documents)
        logger.info("Dense document indexing complete.")

        # args.index_path is treated as a directory for DenseRetriever
        # The save_index method of DenseRetriever handles directory creation.
        logger.info(f"Saving dense index components to directory: {args.index_path}...")
        try:
            dense_retriever.save_index(args.index_path)
            logger.info(f"Dense index components (FAISS index, documents, config) successfully saved to directory: {args.index_path}")
        except Exception as e:
            logger.error(f"Failed to save DenseRetriever index to directory {args.index_path}: {e}")
            
    else:
        # This case should not be reached if choices are correctly set in argparse.
        logger.error(f"Unsupported retriever type: {args.retriever_type}. This should not happen.")

if __name__ == "__main__":
    main()
```

The main change is adding a `try-except` block for the import of `BM25Retriever` to make the script more runnable in different execution contexts by modifying `sys.path` if the direct import fails. This is a common pattern for scripts within a larger project that might not be formally "installed" during development. I also added `exist_ok=True` to `os.makedirs` and slightly improved logging for skipped lines during document loading.

The core logic remains the same as described in the previous turn.
