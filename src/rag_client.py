import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction, QueryResult
from chromadb.config import Settings

from embedding_pipeline import create_embedding_function

# Configure logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,  # Disable telemetry for privacy
    allow_reset=True,  # Allow database reset for development
)


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}

    # TODO->DONE: Create list of directories that match specific criteria (directory type and name pattern)
    database_root = Path(__file__).parents[1] / "vector_db"
    database_dirs = list(database_root.iterdir())

    # TODO->DONE: Loop through each discovered directory
    for db in database_dirs:
        # TODO->DONE: Wrap connection attempt in try-except block for error handling
        try:
            # TODO->DONE: Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=db,
                settings=CHROMA_SETTINGS,
            )

            # TODO->DONE: Retrieve list of available collections from the database
            collections = client.list_collections()

            # TODO->DONE: Loop through each collection found
            for collection in collections:
                # TODO->DONE: Create unique identifier key combining directory and collection names
                key = db.name + "__" + collection.name

                # TODO->DONE: Build information dictionary containing:
                collection_info: Dict[str, str] = {
                    # TODO->DONE: Store directory path as string
                    "directory": str(db.resolve()),
                    # TODO->DONE: Store collection name
                    "collection_name": collection.name,
                    # TODO->DONE: Create user-friendly display name
                    "display_name": f"{db.name}-{collection.name}",
                    # TODO->DONE: Get document count with fallback for unsupported operations
                    "document_count": str(collection.count()),
                }
                # TODO->DONE: Add collection information to backends dictionary
                backends[key] = collection_info

        except Exception as e:
            # TODO->DONE: Handle connection or access errors gracefully
            # TODO->DONE: Create fallback entry for inaccessible directories
            # TODO->DONE: Include error information in display name with truncation
            # TODO->DONE: Set appropriate fallback values for missing information
            log.exception("Unexpected error connecting to Chroma persistent DB at '%s': %s", db.name, e)
            collection_info: Dict[str, str] = {
                "directory": str(db.resolve()),
                "collection_name": "UNAVAILABLE",
                "display_name": f"Error: {str(e)[:10]}",
                "document_count": "0",
            }
            backends[db.name] = collection_info

    # TODO->DONE: Return complete backends dictionary with all discovered collections
    return backends


def recreate_embedding_function(client, collection_name: str) -> EmbeddingFunction:
    for col in client.list_collections():
        if col.name == collection_name:
            return create_embedding_function(
                embedding_model=col.metadata["embedding_model"],
                openai_api_key=col.metadata["openai_api_key"],
            )

    raise NameError(f"Embedding function could not be recreated for collection {collection_name}")


def initialize_rag_system(
    chroma_dir: str,
    collection_name: str,
) -> Tuple[Collection, bool, str]:
    """Initialize the RAG system with specified backend (cached for performance)"""

    # TODO->DONE: Create a chomadb persistentclient
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=CHROMA_SETTINGS,
    )

    # Get embedding function
    embedding_function = recreate_embedding_function(client, collection_name)

    # TODO->DONE: Return the collection with the collection_name
    collection: Collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )

    return collection, True, ""


def retrieve_documents(
    collection: Collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # TODO->DONE: Initialize filter variable to None (represents no filtering)
    where_filter: Optional[Dict] = None

    # TODO->DONE: Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter is not None:
        mf = mission_filter.strip().lower()
        if mf and mf not in {"all", "*", "any"}:
            # TODO->DONE: If filter conditions are met, create filter dictionary with appropriate field-value pairs
            where_filter = {"mission": mission_filter}

    try:
        # TODO->DONE: Execute database query with the following parameters:
        results: QueryResult = collection.query(
            # TODO->DONE: Pass search query in the required format
            query_texts=[query],
            # TODO->DONE: Set maximum number of results to return
            n_results=n_results,
            # TODO->DONE: Apply conditional filter (None for no filtering, dictionary for specific filtering)
            where=where_filter,
        )
    except Exception as e:
        log.exception(f"Unexpected error retrieving from {collection}: {e}")
        return None

    # TODO->DONE: Return query results
    return {k: v for k, v in results.items()}


def get_indices_unique_documents(metadatas: List[Dict]):
    """Used to filter out repeated documents"""
    seen = set()
    indices = []
    for i, d in enumerate(metadatas):
        key = tuple(sorted(d.items()))
        if key not in seen:
            seen.add(key)
            indices.append(i)
    return indices


def format_context(documents: List[str], metadatas: List[Dict], max_charts: Optional[int] = 1500) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    # Filter repeated documents
    valid_indices = get_indices_unique_documents(metadatas)
    documents = [documents[i] for i in valid_indices]
    metadatas = [metadatas[i] for i in valid_indices]

    # TODO->DONE: Initialize list with header text for context section
    parts: List[str] = ["DOCUMENTS", ""]

    # TODO->DONE: Loop through paired documents and their metadata using enumeration
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        # TODO->DONE: Extract mission information from metadata with fallback value
        # TODO->DONE: Clean up mission name formatting (replace underscores, capitalize)
        mission = str(meta.get("mission", "unknown mission")).strip().replace("_", " ").title()

        # TODO->DONE: Extract source information from metadata with fallback value
        source = str(meta.get("source", "unknown source")).strip().replace("_", " ").title()

        # TODO->DONE: Extract category information from metadata with fallback value
        # TODO->DONE: Clean up category name formatting (replace underscores, capitalize)
        category = str(meta.get("category", "general")).strip().replace("_", " ").title()

        # TODO->DONE: Create formatted source header with index number and extracted information
        # TODO->DONE: Add source header to context parts list
        header = f"**Source {idx}** — Mission: {mission} | Source: {source} | Category: {category}"
        parts.append(header)

        # TODO->DONE: Check document length and truncate if necessary
        if max_charts is not None and len(doc) > max_charts:
            doc = doc[:max_charts].rstrip() + "…"

        # TODO->DONE: Add truncated or full document content to context parts list
        parts.append(doc)
        parts.append("")

    # TODO->DONE: Join all context parts with newlines and return formatted string
    return "\n".join(parts).strip()
