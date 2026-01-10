import json
import os
from pathlib import Path
from typing import Dict

import llm_client
import rag_client
import ragas_evaluator

CHROMA_DIR = str(Path(__file__).parents[1] / "vector_db" / "voice_transcripts")
COLLECTION_NAME = "nasa_voice_transcripts"

INPUT_QUESTIONS = Path(__file__).parent / "evaluation_dataset.txt"
OUTPUT_FILE = Path(__file__).parent / "evaluation_results.json"


def run_pipeline(question: str) -> Dict[str, str | Dict[str, float | str]]:
    """End-to-end pipeline"""

    collection, success, _ = rag_client.initialize_rag_system(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    assert success

    results = rag_client.retrieve_documents(collection, question, n_results=5)
    assert results is not None

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    context = rag_client.format_context(documents, metadatas)

    document_ids = [
        {
            "mission": str(md["mission"]),
            "chunk_index": str(md["chunk_index"]),
        }
        for md in metadatas
    ]

    answer = llm_client.generate_response(
        openai_key="",
        user_message=question,
        context=context,
        conversation_history=[],  # no history
        model="gpt-3.5-turbo",
    )
    assert not answer.startswith("An error occurred")

    metrics = ragas_evaluator.evaluate_response_quality(
        question,
        answer,
        documents,
    )

    assert "error" not in metrics.keys()

    return {
        "question": question,
        "documents": document_ids,  # type: ignore
        "answer": answer,
        "metrics": metrics,
    }


if __name__ == "__main__":
    # NOTE: PLEASE SET THE API KEY TO RUN!
    os.environ["OPENAI_API_KEY"] = "voc-..."

    # read questions
    with open(INPUT_QUESTIONS) as f:
        questions = f.readlines()

    # get output
    output = []
    for question in questions:
        output.append(run_pipeline(question))

    # export to json
    with open(OUTPUT_FILE, "w") as fp:
        json.dump(output, fp, indent=4)
