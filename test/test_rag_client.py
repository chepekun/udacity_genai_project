from chromadb.api.types import DefaultEmbeddingFunction

import rag_client


def test_rag_client():
    """Test that the unit_test database is accessible"""

    # unit test backend available
    backends = rag_client.discover_chroma_backends()
    assert "unit_test__unit_test_collection" in backends
    backend = backends["unit_test__unit_test_collection"]
    assert backend["document_count"] == "3"

    # collection can be recreated
    collection, success, _ = rag_client.initialize_rag_system(
        chroma_dir=backend["directory"],
        collection_name=backend["collection_name"],
    )
    assert success

    # original embedding function is recreated
    assert isinstance(collection._embedding_function, DefaultEmbeddingFunction)

    # query with mission filter
    test_query = "Houston, Roger that."
    results = rag_client.retrieve_documents(collection, test_query, n_results=2, mission_filter="apollo_11")
    assert results is not None
    assert len(results["documents"][0]) == 1
    assert results["ids"][0][0].startswith("apollo_11")

    # query without filter
    results = rag_client.retrieve_documents(collection, test_query, n_results=2)
    assert results is not None
    assert len(results["documents"][0]) == 2

    # context formatting
    context = rag_client.format_context(documents=results["documents"][0], metadatas=results["metadatas"][0])

    lines = context.split("\n")
    source_1 = "**Source 1** — Mission: Apollo 11 | Source: Transcript | Category: General"
    assert source_1 in lines

    source_2 = "**Source 2** — Mission: Apollo 13 | Source: Transcript | Category: General"
    assert source_2 in lines
