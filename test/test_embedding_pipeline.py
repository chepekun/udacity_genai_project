from pathlib import Path

import embedding_pipeline

DATA_PATH = Path(__file__).parents[1] / "data"


def test_embedding_pipeline():
    """Test that the files are found and embedded"""
    pipeline = embedding_pipeline.ChromaEmbeddingPipelineTextOnly(
        openai_api_key="",
        embedding_model="",
        chroma_persist_directory="unit_test",
        collection_name="unit_test_collection",
        debug_mode=True,
    )

    # add documents: replace, update and skip
    stats = pipeline.process_all_text_data(str(DATA_PATH), update_mode="replace")

    assert stats["files_processed"] == 3
    assert stats["total_chunks"] == 3
    assert stats["documents_added"] == 3

    stats = pipeline.process_all_text_data(str(DATA_PATH), update_mode="update")
    assert stats["documents_updated"] == 3

    stats = pipeline.process_all_text_data(str(DATA_PATH), update_mode="skip")
    assert stats["documents_skipped"] == 3

    # test query
    test_query = "Houston, Roger that."
    results = pipeline.query_collection(test_query, n_results=2)
    assert len(results["documents"][0]) == 2
