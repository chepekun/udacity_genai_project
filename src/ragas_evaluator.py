from typing import Dict, List, Optional

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationResult
    from ragas.metrics import (
        BleuScore,  #  measures answer vs. ground truth
        Faithfulness,  # measures answer vs. context.
        NonLLMContextPrecisionWithReference,  # measures retrieved contexts vs reference contexts
        ResponseRelevancy,  # measures answer vs. question.
        RougeScore,  #  measures answer vs. ground truth
    )

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    reference_answer: Optional[str] = None,
    reference_contexts: Optional[List[str]] = None,
    use_llm_metrics: bool = True,  # for unit testing
) -> Dict[str, float | str]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "Ragas is not available"}

    # TODO->DONE: Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = None
    if use_llm_metrics:
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                base_url="https://openai.vocareum.com/v1",
                temperature=0.1,
            )
        )

    # TODO->DONE: Create evaluator_embeddings with model test-embedding-3-small
    evaluator_embeddings = None
    if use_llm_metrics:
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                base_url="https://openai.vocareum.com/v1",
            )
        )

    # TODO->DONE: Define an instance for each metric to evaluate
    dataset = {
        "question": [question],
        "contexts": [contexts],
        "answer": [answer],
    }
    metrics = []
    if use_llm_metrics:
        metrics.extend([Faithfulness(), ResponseRelevancy()])

    # with reference answer
    if reference_answer is not None:
        dataset["reference"] = [reference_answer]
        metrics.extend([BleuScore(), RougeScore()])

    # with reference context
    if reference_contexts is not None:
        dataset["reference_contexts"] = [reference_contexts]
        metrics.extend([NonLLMContextPrecisionWithReference()])

    if not metrics:
        return {"error": "No metrics selected"}

    # TODO->DONE: Evaluate the response using the metrics
    try:
        result: EvaluationResult = evaluate(
            dataset=Dataset.from_dict(dataset),
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            return_executor=False,
        )

        # TODO->DONE: Return the evaluation results
        return {k: float(v) for k, v in result.scores[0].items()}

    except Exception as e:
        return {"error": f"Error: {e}"}
