"""Semantic similarity metrics using embeddings

Uses sentence transformers to compute semantic similarity between:
- Expected vs actual outputs
- Task descriptions vs responses
- Trace steps for coherence analysis
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None


class SemanticMetric:
    """Semantic similarity metrics using transformer embeddings
    
    Uses pre-trained sentence transformers (e.g., all-MiniLM-L6-v2) to compute
    semantic embeddings and cosine similarity between texts.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize semantic metric
        
        Args:
            model_name: Name of sentence transformer model
            device: Device to run model on ('cpu', 'cuda', or None for auto)
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for SemanticMetric. "
                "Install with: pip install sentence-transformers>=2.2.0"
            )
        
        self.model_name = model_name
        self.device = device
        
        # Lazy loading of model
        self._model: Optional[SentenceTransformer] = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load model"""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of text strings
            
        Returns:
            Embeddings array [num_texts, embedding_dim]
        """
        if not texts:
            return np.array([])
        
        # Handle single string
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    def cosine_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between embeddings
        
        Args:
            embeddings1: First set of embeddings [N, D]
            embeddings2: Second set of embeddings [M, D]
            
        Returns:
            Similarity matrix [N, M] or scalar if both are 1D
        """
        # Normalize embeddings
        def normalize(x):
            norms = np.linalg.norm(x, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return x / norms
        
        emb1_norm = normalize(embeddings1)
        emb2_norm = normalize(embeddings2)
        
        # Compute cosine similarity
        if embeddings1.ndim == 1 and embeddings2.ndim == 1:
            return np.dot(emb1_norm, emb2_norm)
        else:
            return np.dot(emb1_norm, emb2_norm.T)
    
    def semantic_accuracy(
        self,
        expected: Any,
        actual: Any
    ) -> float:
        """
        Compute semantic accuracy using embeddings
        
        Args:
            expected: Expected output (will be converted to string)
            actual: Actual output (will be converted to string)
            
        Returns:
            Semantic similarity score [0, 1]
        """
        expected_str = str(expected)
        actual_str = str(actual)
        
        if not expected_str or not actual_str:
            return 0.0
        
        # Encode both texts
        embeddings = self.encode([expected_str, actual_str])
        
        # Compute cosine similarity
        similarity = self.cosine_similarity(embeddings[0], embeddings[1])
        
        # Convert from [-1, 1] to [0, 1]
        return (similarity + 1.0) / 2.0
    
    def semantic_coherence(
        self,
        texts: List[str]
    ) -> float:
        """
        Compute semantic coherence across a sequence of texts
        
        Measures how semantically consistent a sequence is by computing
        pairwise similarities between consecutive elements.
        
        Args:
            texts: List of text strings (e.g., trace steps)
            
        Returns:
            Coherence score [0, 1]
        """
        if len(texts) < 2:
            return 1.0
        
        # Encode all texts
        embeddings = self.encode(texts)
        
        # Compute pairwise similarities between consecutive texts
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self.cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append((sim + 1.0) / 2.0)  # Normalize to [0, 1]
        
        # Average coherence
        return np.mean(similarities) if similarities else 0.0
    
    def topic_consistency(
        self,
        task_description: str,
        response: str
    ) -> float:
        """
        Compute topic consistency between task and response
        
        Args:
            task_description: Task description text
            response: Agent response text
            
        Returns:
            Consistency score [0, 1]
        """
        if not task_description or not response:
            return 0.0
        
        embeddings = self.encode([task_description, response])
        similarity = self.cosine_similarity(embeddings[0], embeddings[1])
        
        return (similarity + 1.0) / 2.0
    
    def collect(
        self,
        agent: Any,
        task: Any,
        result: Any,
        trace: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Collect semantic metrics
        
        Args:
            agent: Agent instance
            task: Task instance
            result: Task result
            trace: Execution trace
            
        Returns:
            Dictionary of semantic metrics
        """
        metrics = {}
        
        # Semantic accuracy (if expected output exists)
        expected = getattr(task, "expected_output", None)
        if expected is not None:
            result_str = str(result)
            metrics["semantic_accuracy"] = self.semantic_accuracy(expected, result_str)
        
        # Semantic coherence from trace
        trace_texts = []
        for step in trace:
            step_text = str(step.get("output_data", ""))
            if step_text:
                trace_texts.append(step_text)
        
        if trace_texts:
            metrics["semantic_coherence"] = self.semantic_coherence(trace_texts)
        
        # Topic consistency
        task_desc = str(getattr(task, "description", ""))
        result_str = str(result)
        if task_desc and result_str:
            metrics["topic_consistency"] = self.topic_consistency(task_desc, result_str)
        
        return metrics


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_order: List[str] = []
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        if text in self.cache:
            # Update access order
            if text in self.access_order:
                self.access_order.remove(text)
            self.access_order.append(text)
            return self.cache[text]
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """Cache embedding"""
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[text] = embedding
        self.access_order.append(text)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()
