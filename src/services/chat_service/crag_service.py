"""
CRAG (Corrective RAG) service with self-correcting retrieval.

Provides:
- CRAGService: Self-correcting RAG with confidence scoring
- Automatic query expansion on low confidence
- Better grounding and reduced hallucinations

Workflow:
    1. Initial retrieval (k=4)
    2. Calculate confidence score
    3. If low: Corrective retrieval (k=8, expanded)
    4. Generate with best evidence

Benefits:
- 🎯 Better accuracy for complex queries
- 🛡️ Reduces hallucinations
- 🔄 Automatic self-correction

Logging:
    Uses the shared project logger with output written to ``logs/crag_service.log``.
"""

from typing import Any, Dict, List
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from infrastructure.config import settings
from infrastructure.log import get_logger
from services.chat_service.rag_templates import RAG_TEMPLATE
from services.chat_service.rag_service import QdrantRetriever
from infrastructure.utils import format_docs, calculate_confidence

logger = get_logger(__name__, "crag_service.log")

TOP_K_RESULTS = int(settings.retrieval.get("top_k", 4))
CRAG_CONFIDENCE_THRESHOLD = float(settings.crag.get("confidence_threshold", 0.6))
CRAG_EXPANDED_K = int(settings.crag.get("expanded_k", 8))


class CRAGService:
    """
    Corrective RAG service with automatic self-correction.
    
    Features:
    - Initial retrieval with confidence scoring
    - Automatic corrective retrieval if confidence low
    - Query expansion strategies
    - Detailed metrics for debugging
    
    Usage:
        service = CRAGService(retriever, llm)
        result = service.generate(query, confidence_threshold=0.6)
        
        logger.info(result['answer'])
        logger.info(f"Confidence: {result['confidence_final']}")
        logger.info(f"Correction applied: {result['correction_applied']}")
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Any,
        initial_k: int = TOP_K_RESULTS,
        expanded_k: int = CRAG_EXPANDED_K
    ):
        """
        Initialize CRAG service.
        
        Args:
            retriever: LangChain retriever (QdrantRetriever, etc.)
            llm: LangChain LLM instance
            initial_k: Number of docs for initial retrieval
            expanded_k: Number of docs for corrective retrieval
        """
        self.retriever = retriever
        self.llm = llm
        self.initial_k = initial_k
        self.expanded_k = expanded_k
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    def _set_k(self, k: int) -> None:
        """Set retrieval count on the retriever (supports both patterns)."""
        if isinstance(self.retriever, QdrantRetriever):
            self.retriever.top_k = k
        elif hasattr(self.retriever, "search_kwargs"):
            self.retriever.search_kwargs["k"] = k
    
    def generate(
        self,
        query: str,
        confidence_threshold: float = CRAG_CONFIDENCE_THRESHOLD,
        verbose: bool = True,
        chat_history: str = "",
    ) -> Dict[str, Any]:
        """
        Generate answer with CRAG (Corrective RAG).
        
        Workflow:
        1. Initial retrieval (k=initial_k)
        2. Calculate confidence score
        3. If confidence < threshold:
           - Apply corrective retrieval (k=expanded_k)
           - Recalculate confidence
        4. Generate answer with best evidence
        
        Args:
            query: User question
            confidence_threshold: Minimum confidence score (0-1)
            verbose: Print progress logs
            chat_history: Optional prior conversation history for continuity
        
        Returns:
            Dict with:
            - answer: Generated answer
            - confidence_initial: Initial confidence score
            - confidence_final: Final confidence score
            - correction_applied: Whether corrective retrieval was used
            - docs_used: Number of documents in final generation
            - generation_time: Total time taken
            - evidence_urls: List of source URLs
        """
        if verbose:
            logger.info("🔍 Query: %s", query)
            logger.info("🎯 Confidence threshold: %s\n", confidence_threshold)
        
        # Step 1: Initial retrieval
        if verbose:
            logger.info("1️⃣  Initial retrieval (k=%s)...", self.initial_k)
        
        self._set_k(self.initial_k)
        docs_initial = self.retriever.invoke(query)
        confidence_initial = calculate_confidence(docs_initial, query)
        
        if verbose:
            logger.info("   📊 Confidence: %.2f", confidence_initial)
        
        # Step 2: Check if correction needed
        if confidence_initial >= confidence_threshold:
            if verbose:
                logger.info("   ✅ Confidence sufficient - proceeding with initial retrieval")
            final_docs = docs_initial
            confidence_final = confidence_initial
            correction_applied = False
        else:
            if verbose:
                logger.warning("   ⚠️  Low confidence - applying corrective retrieval...\n")
            
            # Step 3: Corrective retrieval
            if verbose:
                logger.info("2️⃣  Corrective retrieval (k=%s, expanded)...", self.expanded_k)
            
            # Expand k for more diverse results
            self._set_k(self.expanded_k)
            docs_corrected = self.retriever.invoke(query)
            confidence_final = calculate_confidence(docs_corrected, query)
            
            if verbose:
                logger.info("   📊 Corrected confidence: %.2f", confidence_final)
                improvement = (confidence_final - confidence_initial) * 100
                logger.info("   📈 Confidence improved by %.1f%%", improvement)
            
            final_docs = docs_corrected
            correction_applied = True
        
        # Step 4: Generate answer
        if verbose:
            logger.info(f"\n3️⃣  Generating answer...")
        
        start = time.time()
        
        # Format docs and generate
        context = format_docs(final_docs)
        question_for_prompt = query
        if chat_history.strip():
            question_for_prompt = (
                "Conversation history:\n"
                f"{chat_history.strip()}\n\n"
                "Current user question:\n"
                f"{query}"
            )

        prompt_input = {"context": context, "question": question_for_prompt}
        answer = (self.prompt | self.llm | StrOutputParser()).invoke(prompt_input)
        
        elapsed = time.time() - start
        
        # Extract evidence URLs
        evidence_urls = list(set([doc.metadata['url'] for doc in final_docs]))
        
        return {
            'answer': answer,
            'confidence_initial': confidence_initial,
            'confidence_final': confidence_final,
            'correction_applied': correction_applied,
            'docs_used': len(final_docs),
            'generation_time': elapsed,
            'evidence_urls': evidence_urls,
            'evidence': final_docs
        }

