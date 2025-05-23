import abc
import logging
from typing import List, Dict, Any, Optional

# Attempt to import rank_bm25, otherwise use a mock
try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False
    # Standard library logger, get it at the module level
    # but configure/use it where needed, e.g., inside classes or functions.
    # logger = logging.getLogger(__name__) # This is fine, or get it inside classes
    # print("rank_bm25 library not found. Using a mock BM25Okapi implementation.") # Print is not ideal for libraries

    class BM25Okapi:
        """
        A mock implementation of BM25Okapi for when rank_bm25 is not available.
        Simulates basic indexing and searching.
        """
        def __init__(self, tokenized_corpus: List[List[str]]):
            self.tokenized_corpus = tokenized_corpus
            self.indexed_documents: List[Dict[str, Any]] = [] 
            self._logger = logging.getLogger(__name__ + ".MockBM25Okapi") # Class-specific logger
            if not RANK_BM25_AVAILABLE: # Log only if we are actually using the mock due to import failure
                 self._logger.warning("rank_bm25 library not found. Using a mock BM25Okapi implementation.")


        def get_top_n(self, tokenized_query: List[str], documents: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
            """
            Mock search method. Returns a subset of the provided documents.
            This mock doesn't actually use the query or corpus for ranking.
            """
            documents_to_search = self.indexed_documents # Use documents stored during index_documents

            results = []
            if not documents_to_search:
                self._logger.warning("MockBM25Okapi: No documents available to search.")
                return []

            for doc in documents_to_search:
                doc_text_tokens = doc.get("text", "").lower().split()
                found = False
                for query_token in tokenized_query:
                    if query_token.lower() in doc_text_tokens:
                        found = True
                        break
                if found:
                    results.append(doc)
                if len(results) >= n: # Use >= to handle n=0 or if somehow more found
                    break
            
            if len(results) < n:
                additional_needed = n - len(results)
                non_result_docs = [doc for doc in documents_to_search if doc not in results] # Expensive for large lists
                results.extend(non_result_docs[:additional_needed])
                
            return results[:n]

        def _store_documents_for_mock(self, documents: List[Dict[str, str]]):
            """Helper for the mock to store the original documents."""
            self.indexed_documents = documents


class BaseRetriever(abc.ABC):
    """
    Abstract base class for retriever models.
    """
    def __init__(self):
        """Initializes the retriever."""
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    @abc.abstractmethod
    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Search for relevant documents given a query.

        Args:
            query: The search query string.
            top_k: The number of top documents to return.

        Returns:
            A list of dictionaries, where each dictionary represents a document
            (e.g., {"title": "...", "text": "..."}).
        """
        pass

    @abc.abstractmethod
    def index_documents(self, documents: List[Dict[str, str]]):
        """
        Index a list of documents.

        Args:
            documents: A list of dictionaries, where each dictionary has
                       'title' and 'text' keys.
        """
        pass


class BM25Retriever(BaseRetriever):
    """
    A retriever using the BM25 algorithm.
    """
    def __init__(self, corpus: Optional[List[Dict[str, str]]] = None):
        """
        Initializes the BM25Retriever.

        Args:
            corpus: An optional list of documents to index immediately.
                    Each document is a dict with 'title' and 'text'.
        """
        super().__init__()
        self.documents: List[Dict[str, str]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        if corpus:
            self.index_documents(corpus)

    def _tokenize(self, text: str) -> List[str]:
        """Simple space-based tokenizer."""
        return text.lower().split()

    def index_documents(self, documents: List[Dict[str, str]]):
        """
        Indexes the provided documents using BM25.

        Args:
            documents: A list of document dictionaries with 'title' and 'text'.
        """
        if not documents:
            self._logger.warning("No documents provided for indexing.")
            self.documents = []
            self.tokenized_corpus = []
            self.bm25 = None
            return

        self.documents = documents
        self.tokenized_corpus = [self._tokenize(doc.get("text","")) for doc in self.documents]
        
        if RANK_BM25_AVAILABLE:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self._logger.info(f"BM25Retriever indexed {len(documents)} documents using rank_bm25.")
        else:
            # This means BM25Okapi is our mock class (already instantiated by virtue of the import logic)
            self.bm25 = BM25Okapi(self.tokenized_corpus) 
            # For mock, explicitly store original documents as it doesn't inherently keep them via tokenized corpus alone for search.
            if hasattr(self.bm25, '_store_documents_for_mock'):
                 self.bm25._store_documents_for_mock(self.documents)
            self._logger.info(f"BM25Retriever indexed {len(documents)} documents using Mock BM25Okapi.")
        
        # TODO: Add placeholder for saving/loading the index
        # self.save_index(path) / self.load_index(path)

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Searches the indexed documents for the given query.

        Args:
            query: The search query string.
            top_k: The number of top documents to return.

        Returns:
            A list of document dictionaries.
        """
        if not self.bm25 or not self.documents: # Check self.documents as well
            self._logger.warning("BM25Retriever has not been indexed or no documents are available. Returning empty list.")
            return []
        if top_k <= 0:
            return []

        tokenized_query = self._tokenize(query)
        
        try:
            # The rank_bm25 library's get_top_n expects the original documents list to map results back.
            # Our mock also uses this pattern (it searches self.indexed_documents which is set from self.documents).
            top_docs = self.bm25.get_top_n(tokenized_query, self.documents, n=top_k)
        except Exception as e:
            self._logger.error(f"Error during BM25 search: {e}")
            return []
            
        return top_docs


class DenseRetriever(BaseRetriever):
    """
    A placeholder for a dense retriever model.
    """
    def __init__(self, model_name_or_path: Optional[str] = None, corpus_or_index_path: Optional[str] = None):
        """
        Initializes the DenseRetriever.

        Args:
            model_name_or_path: Name or path of the dense retrieval model.
            corpus_or_index_path: Path to the corpus or a pre-built index.
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.corpus_or_index_path = corpus_or_index_path
        self._logger.info(f"DenseRetriever initialized with model: {model_name_or_path}, corpus/index: {corpus_or_index_path}")
        # TODO: Initialize model, tokenizer, index

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Placeholder for dense search.
        """
        if top_k <= 0:
            return []
        self._logger.info(f"Dense search called for query: '{query}', top_k: {top_k}. Not yet implemented.")
        # TODO: Implement dense search logic:
        # 1. Encode query using the sentence transformer model.
        # 2. Search the vector index (e.g., FAISS) for top_k closest document embeddings.
        # 3. Retrieve original document content for the top_k results.
        return []

    def index_documents(self, documents: List[Dict[str, str]]):
        """
        Placeholder for indexing documents for dense retrieval.
        """
        self._logger.info(f"Dense retriever indexing called for {len(documents)} documents. Not yet implemented.")
        # TODO: Implement dense indexing logic:
        # 1. For each document, generate its embedding using the sentence transformer model.
        # 2. Store these embeddings in a searchable index (e.g., FAISS).
        # 3. Store the original documents or a mapping to retrieve them.
        pass

# Example Usage (optional, for testing)
if __name__ == '__main__':
    # Setup basic logging for example
    logging.basicConfig(level=logging.INFO)
    
    sample_corpus = [
        {"title": "Doc1", "text": "The quick brown fox jumps over the lazy dog"},
        {"title": "Doc2", "text": "A fast red fox"},
        {"title": "Doc3", "text": "Exploring the universe and its wonders"},
        {"title": "Doc4", "text": "Artificial intelligence in modern times"}
    ]

    # Test BM25Retriever
    print("\n--- Testing BM25Retriever ---")
    # Note: The mock BM25Okapi will log a warning if rank_bm25 is not installed.
    bm25_retriever = BM25Retriever() 
    bm25_retriever.index_documents(sample_corpus)
    
    search_query_bm25 = "quick fox"
    results_bm25 = bm25_retriever.search(search_query_bm25, top_k=2)
    print(f"BM25 results for '{search_query_bm25}':")
    for res in results_bm25:
        print(f"  Title: {res['title']}, Text: {res['text'][:50]}...")

    search_query_bm25_2 = "artificial intelligence"
    results_bm25_2 = bm25_retriever.search(search_query_bm25_2, top_k=1)
    print(f"BM25 results for '{search_query_bm25_2}':")
    for res in results_bm25_2:
        print(f"  Title: {res['title']}, Text: {res['text'][:50]}...")

    # Test DenseRetriever (placeholder)
    print("\n--- Testing DenseRetriever ---")
    dense_retriever = DenseRetriever(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")
    dense_retriever.index_documents(sample_corpus)
    results_dense = dense_retriever.search("modern AI", top_k=2)
    print(f"Dense results: {results_dense}")
