#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders and creates
a permanent ChromaDB collection with OpenAI embeddings for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Apollo 11 Textract extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple

import chromadb
import dateutil
import numpy as np
import pandas as pd
from chromadb.api.types import DefaultEmbeddingFunction, EmbeddingFunction
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


class VoiceTranscript:
    """Helper class to read and handle voice transcript text files"""

    def __init__(
        self,
        content: str,
        mission: str,
        file_path: str,
        max_interruption_minutes: float = 60,
    ):
        self.mission = mission
        self.max_interruption_minutes = max_interruption_minutes

        # load transcript metadata
        metadata_path = Path(file_path).parent / "transcript_metadata.json"
        with open(metadata_path) as f:
            self.transcript_metadata = json.load(f)

        # read file as dataframes
        self.data: List[pd.DataFrame] = self.parse_transcript(content)

    def iterate_chunks(self, chunk_size: int, chunk_overlap: int) -> Iterator[str]:
        """Splits the transcript in chunks"""
        for df_index, df in enumerate(self.data):
            message_length: np.ndarray = df.text.map(lambda t: len(t)).to_numpy()

            num_lines = len(message_length)

            # cumulative sum of paragraph length
            cum_length = np.empty(num_lines + 1, dtype=np.int64)
            cum_length[0] = 0
            np.cumsum(message_length, out=cum_length[1:])

            start_idx = 0
            while start_idx < num_lines:
                # Find next chunk
                idx = int(np.searchsorted(cum_length, cum_length[start_idx] + chunk_size, side="right"))
                if idx > num_lines:
                    yield self.create_chunk(df_index, start_idx, num_lines - 1)
                    break

                end_idx = idx - 1
                yield self.create_chunk(df_index, start_idx, end_idx)

                # Include overlap
                next_start_chars = max(0, cum_length[end_idx + 1] - chunk_overlap)
                next_start_idx = int(np.searchsorted(cum_length, next_start_chars, side="right"))

                # Ensure we make progress to avoid infinite loop
                if next_start_idx <= start_idx:
                    next_start_idx = end_idx

                start_idx = next_start_idx

    def create_chunk(self, df_index: int, start_idx: int, end_idx: int) -> str:
        """Extract a piece of the transcript and bracket it with the timestamp."""
        df = self.data[df_index]
        start_idx = max(0, min(start_idx, len(df) - 1))
        end_idx = min(end_idx, len(df) - 1)

        text = [f"[{self.mission}: {df.iloc[start_idx].time_stamp}]"]
        for idx in range(start_idx, end_idx + 1):
            entry = df.iloc[idx]
            text.append(f"{entry.speaker}: {entry.text}")

        return "\n".join(text)

    def parse_transcript(self, content: str) -> List[pd.DataFrame]:
        """
        Parses a mission transcript into a pandas dataframe
        For each paragraph:
        - first line must match "<DDD:HH:MM> <SPEAKER>"
        - remaining lines are the message text (joined with newlines)
        The file is split into multiple dataframes if the communication interrupts for more than x minutes.
        """
        # read transcript metadata
        mission_start: datetime = self.get_mission_start()
        speaker_names: Dict[str, str] = self.get_speaker_names()
        text_processor: Callable[[str], str] = self.build_text_processor()

        # Header format: GET timestamp + speaker code on the same line
        header_re = re.compile(r"^(\d{3}:\d{2}:\d{2})\s+([A-Z0-9\-]+)\s*$")

        dfs: List[pd.DataFrame] = []
        timestamps: List[str] = []
        speakers: List[str] = []
        texts: List[str] = []

        previous_timestamp: datetime = mission_start

        # Break the content into paragraphs
        paragraphs = re.split(r"(?:\r?\n){2,}", content.strip())
        for paragraph in paragraphs:
            para_lines = paragraph.splitlines()
            if not para_lines:
                continue

            head = para_lines[0].strip()
            m = header_re.match(head)
            if not m:
                logger.info(f"Warning: skipped chunk without header: {head!r}")
                continue

            mission_time = m.group(1)
            speaker = speaker_names[m.group(2)]
            text = text_processor("\n".join(line.rstrip("\n") for line in para_lines[1:]).strip())

            hours, minutes, seconds = map(int, mission_time.split(":"))
            times_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
            timestamp = mission_start + times_delta

            if (timestamp - previous_timestamp).total_seconds() / 60 >= self.max_interruption_minutes:
                # flush data
                dfs.append(
                    pd.DataFrame(
                        {
                            "time_stamp": timestamps,
                            "speaker": speakers,
                            "text": texts,
                        }
                    )
                )
                timestamps.clear()
                speakers.clear()
                texts.clear()

            timestamps.append(str(timestamp))
            speakers.append(speaker)
            texts.append(text)

            previous_timestamp = timestamp

        if texts:
            dfs.append(
                pd.DataFrame(
                    {
                        "time_stamp": timestamps,
                        "speaker": speakers,
                        "text": texts,
                    }
                )
            )

        return dfs

    def get_mission_start(self) -> datetime:
        """Reads the start of the mission"""
        return dateutil.parser.parse(self.transcript_metadata["mission_start"])

    def get_speaker_names(self) -> Dict[str, str]:
        """Reads a map of the speaker names used in the transcript"""
        return self.transcript_metadata["speaker_name"]

    def build_text_processor(self) -> Callable[[str], str]:
        """Returns a function that replaces whole words in text."""
        technical_acronyms: Dict[str, str] = self.transcript_metadata["technical_acronyms"]

        keys = sorted(technical_acronyms.keys(), key=len, reverse=True)
        pattern = re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, keys)))

        def map_text(text: str) -> str:
            return pattern.sub(lambda m: technical_acronyms[m.group(0)], text)

        return map_text


def create_embedding_function(
    embedding_model: str = "text-embedding-3-small",
    openai_api_key: str = "",
) -> EmbeddingFunction:
    """Encapsulate embedding function creation to reuse in the rag_client"""

    if not embedding_model:
        logger.info("Using Default embedding function")
        return DefaultEmbeddingFunction()

    if not openai_api_key:
        logger.info("Using OpenAI embedding function with environment keys")
        return embedding_functions.OpenAIEmbeddingFunction(
            model_name=embedding_model,
            api_base="https://openai.vocareum.com/v1",
        )

    logger.info("Using OpenAI embedding function with input keys")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=embedding_model,
        api_base="https://openai.vocareum.com/v1",
    )


class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - Text files only"""

    def __init__(
        self,
        openai_api_key: str,
        embedding_model: str,
        chroma_persist_directory: str,
        collection_name: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        debug_mode: bool = False,
    ):
        """
        Initialize the embedding pipeline

        Args:
            openai_api_key: OpenAI API key
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        # TODO->DONE: Initialize OpenAI client
        # NOTE: I will pass the model to automatically embed within ChromaDB.
        # Otherwise I can get an inconsistent dimension in the RAG client where a different model
        # can be used
        # self.openai_client = OpenAI(
        #     base_url="https://openai.vocareum.com/v1",
        #     api_key=openai_api_key,
        # )

        embedding_function = create_embedding_function(embedding_model, openai_api_key)

        # TODO->DONE: Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=Path(__file__).parents[1] / "vector_db" / chroma_persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry for privacy
                allow_reset=True,  # Allow database reset for development
            ),
        )

        # TODO->DONE: Create or get collection
        try:
            # Create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={
                    "description": "NASA Mission Transcripts",
                    "embedding_model": embedding_model,
                    "openai_api_key": openai_api_key,
                },
            )

        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

        # TODO->DONE: Store configuration parameters
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.debug_mode = debug_mode

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks with metadata

        Args:
            text: Text to chunk
            metadata: Base metadata for the text

        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """

        if metadata["data_type"] == "transcript":
            voice_transcript = VoiceTranscript(
                content=text,
                mission=metadata["mission"],
                file_path=metadata["file_path"],
            )

            num_paragraphs = sum([len(d) for d in voice_transcript.data])
            logger.info(f"   Voice transcript read with {num_paragraphs} paragraphs.")

            output: List[Tuple[str, Dict[str, Any]]] = []

            # TODO->DONE: Handle short texts that don't need chunking
            # TODO->DONE: Implement chunking logic with overlap
            # TODO->DONE: Try to break at sentence boundaries
            chunk_meta_base = {
                "mission": metadata["mission"],
                "source": metadata["source"],
                "file_path": metadata["file_path"],
            }

            for chunk_index, chunk_text in enumerate(
                voice_transcript.iterate_chunks(self.chunk_size, self.chunk_overlap)
            ):
                # TODO->DONE: Create metadata for each chunk
                chunk_meta = chunk_meta_base.copy()
                chunk_meta["chunk_index"] = chunk_index
                output.append((chunk_text, chunk_meta))

                if self.debug_mode:
                    # only 1 embedding
                    break

            return output

        else:
            raise NotImplementedError(f"Reader for data tpe {metadata['data_type']} is not implemented")

    def check_document_exists(self, doc_id: str) -> bool:
        """
        Check if a document with the given ID already exists in the collection

        Args:
            doc_id: Document ID to check

        Returns:
            True if document exists, False otherwise
        """
        try:
            # TODO->DONE: Query collection for document ID
            response = self.collection.get(ids=[doc_id])
            doc_ids = response.get("ids", [])

            # TODO->DONE: Return True if exists, False otherwise
            return doc_id in doc_ids

        except Exception as e:
            logger.error(f"Error checking existance of document {doc_id}: {e}")
            return False

    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Update an existing document in the collection

        Args:
            doc_id: Document ID to update
            text: New text content
            metadata: New metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get new embedding
            # NOTE: handeled by Chroma DB
            # embedding = self.get_embedding(text)

            # Update the document
            self.collection.update(ids=[doc_id], documents=[text], metadatas=[metadata])
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:
        """
        Delete all documents from a specific source (useful for re-processing files)

        Args:
            source_pattern: Pattern to match source names

        Returns:
            Number of documents deleted
        """
        try:
            # Get all documents
            all_docs = self.collection.get()

            # Find documents matching the source pattern
            ids_to_delete = []
            for i, metadata in enumerate(all_docs["metadatas"]):  # type: ignore
                if source_pattern in metadata.get("source", ""):  # type: ignore
                    ids_to_delete.append(all_docs["ids"][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}")
                return len(ids_to_delete)
            else:
                logger.info(f"No documents found matching source pattern: {source_pattern}")
                return 0

        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0

    def get_file_documents(self, file_path: Path) -> List[str]:
        """
        Get all document IDs for a specific file

        Args:
            file_path: Path to the file

        Returns:
            List of document IDs for the file
        """
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)

            # Get all documents
            all_docs = self.collection.get()

            # Find documents from this file
            file_doc_ids = []
            for i, metadata in enumerate(all_docs["metadatas"]):  # type: ignore
                if metadata.get("source") == source and metadata.get("mission") == mission:
                    file_doc_ids.append(all_docs["ids"][i])

            return file_doc_ids

        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []

    def get_embedding(self, text: str) -> List[float]:
        """
        Get OpenAI embedding for text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # NOTE: NOT USED.
        # The embedding is done directly by ChromaDB
        # try:
        #     # TODO->DONE: Call OpenAI embeddings API
        #     response = self.openai_client.embeddings.create(model=self.embedding_model, input=text)
        #     # TODO->DONE: Return embedding vector
        #     return response.data[0].embedding

        # # TODO->DONE: Add error handling
        # except Exception as e:
        #     logger.info(f"Error generating embeddings: {str(e)}")
        #     raise
        return []

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """
        Generate stable document ID based on file path and chunk position
        This allows for document updates without changing IDs
        """
        # TODO->DONE: Create consistent ID format
        # TODO->DONE: Use mission, source, and chunk_index
        # Format: mission_source_chunk_0001
        return f"{metadata['mission']}_{file_path.name}_chunk_{metadata['chunk_index']:04}"

    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process plain text files with enhanced metadata extraction

        Args:
            file_path: Path to text file

        Returns:
            List of (text, metadata) tuples
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                return []

            # Enhanced metadata extraction
            metadata = {
                "source": file_path.stem,
                "file_path": str(file_path),
                "file_type": "text",
                "content_type": "full_text",
                "mission": self.extract_mission_from_path(file_path),
                "data_type": self.extract_data_type_from_path(file_path),
                "document_category": self.extract_document_category_from_filename(file_path.name),
                "file_size": len(content),
                "processed_timestamp": datetime.now().isoformat(),
            }

            return self.chunk_text(content, metadata)

        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []

    def extract_mission_from_path(self, file_path: Path) -> str:
        """Extract mission name from file path"""
        path_str = str(file_path).lower()
        if "apollo11" in path_str or "apollo_11" in path_str:
            return "apollo_11"
        elif "apollo13" in path_str or "apollo_13" in path_str:
            return "apollo_13"
        elif "challenger" in path_str:
            return "challenger"
        else:
            return "unknown"

    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Extract data type from file path"""
        path_str = file_path.name.lower()
        if "transcript" in path_str:
            return "transcript"
        elif "textract" in path_str:
            return "textract_extracted"
        elif "audio" in path_str:
            return "audio_transcript"
        elif "flight_plan" in path_str:
            return "flight_plan"
        else:
            return "document"

    def extract_document_category_from_filename(self, filename: str) -> str:
        """Extract document category from filename for better organization"""
        filename_lower = filename.lower()

        # voice transcript
        if "transcript" in filename_lower:
            return "voice_transcript"

        # Apollo transcript types
        elif "pao" in filename_lower:
            return "public_affairs_officer"
        elif "cm" in filename_lower:
            return "command_module"
        elif "tec" in filename_lower:
            return "technical"
        elif "flight_plan" in filename_lower:
            return "flight_plan"

        # Challenger audio segments
        elif "mission_audio" in filename_lower:
            return "mission_audio"

        # NASA archive documents
        elif "ntrs" in filename_lower:
            return "nasa_archive"
        elif "19900066485" in filename_lower:
            return "technical_report"
        elif "19710015566" in filename_lower:
            return "mission_report"

        # General categories
        elif "full_text" in filename_lower:
            return "complete_document"
        else:
            return "general_document"

    def scan_text_files_only(self, base_path_str: str) -> List[Path]:
        """
        Scan data directories for text files only (avoiding JSON duplicates)

        Args:
            base_path: Base directory path

        Returns:
            List of text file paths to process
        """
        base_path = Path(base_path_str)
        files_to_process = []

        # Define directories to scan
        data_dirs = ["apollo11", "apollo13", "challenger"]

        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")

                # Find only text files
                text_files = list(dir_path.glob("**/*.txt"))
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")

        # Filter out unwanted files
        filtered_files = []
        for file_path in files_to_process:
            # Skip system files and summaries
            if (
                file_path.name.startswith(".")
                or "summary" in file_path.name.lower()
                or file_path.suffix.lower() != ".txt"
            ):
                continue
            filtered_files.append(file_path)

        logger.info(f"Total text files to process: {len(filtered_files)}")

        # Log file breakdown by mission
        mission_counts = {}
        for file_path in filtered_files:
            mission = self.extract_mission_from_path(file_path)
            mission_counts[mission] = mission_counts.get(mission, 0) + 1

        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info(f"  {mission}: {count} files")

        return filtered_files

    def add_documents_to_collection(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = "skip",
    ) -> Dict[str, int]:
        """
        Add documents to ChromaDB collection in batches with update handling

        Args:
            documents: List of (text, metadata) tuples
            file_path: Path to the source file
            batch_size: Number of documents to process in each batch
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add

        Returns:
            Dictionary with counts of added, updated, and skipped documents
        """
        if not documents:
            return {"added": 0, "updated": 0, "skipped": 0}

        stats = {"added": 0, "updated": 0, "skipped": 0}

        # TODO->DONE: Handle different update modes (skip, update, replace)
        if update_mode == "replace":
            # delete the whole file
            try:
                self.collection.delete(where={"file_path": str(file_path)})
            except Exception as e:
                logger.info(f"Could not delete collection: {e}")

        # TODO->DONE: Process documents in batches
        total = len(documents)
        for batch_start in range(0, total, batch_size):
            batch = documents[batch_start : batch_start + batch_size]

            # collect the data for the batch
            add_ids, add_docs, add_metas = [], [], []
            upd_ids, upd_docs, upd_metas = [], [], []
            for text, meta in batch:
                # TODO->DONE: Generate document ID
                doc_id = self.generate_document_id(file_path, meta)

                # TODO->DONE: Check if exists
                exists = False
                try:
                    res = self.collection.get(ids=[doc_id])
                    ids = res.get("ids", [])
                    exists = doc_id in ids
                except Exception:
                    exists = False

                if exists and update_mode == "skip":
                    stats["skipped"] += 1
                    continue

                # TODO->DONE: Get embedding
                # NOTE: Embedding done within ChromaDB
                # try:
                #     embedding = self.get_embedding(text)
                # except Exception as e:
                #     logger.warning(f"Embedding failed for {doc_id}: {e}")
                #     stats["skipped"] += 1
                #     continue

                # TODO->DONE: Add to add or update lists
                if exists and update_mode == "update":
                    upd_ids.append(doc_id)
                    upd_docs.append(text)
                    upd_metas.append(meta)
                    stats["updated"] += 1
                else:
                    add_ids.append(doc_id)
                    add_docs.append(text)
                    add_metas.append(meta)
                    stats["added"] += 1

            # TODO->DONE: Add or update in collection for the batch
            if add_ids:
                try:
                    self.collection.add(
                        ids=add_ids,
                        documents=add_docs,
                        metadatas=add_metas,
                    )
                except Exception as e:
                    logging.error(f"Batch add failed: {e}")
                    num_failed = len(add_ids)
                    stats["added"] -= num_failed
                    stats["skipped"] += num_failed

            if upd_ids:
                try:
                    self.collection.update(
                        ids=upd_ids,
                        documents=upd_docs,
                        metadatas=upd_metas,
                    )
                except Exception as e:
                    logging.error(f"Batch update failed: {e}")
                    num_failed = len(upd_ids)
                    stats["updated"] -= num_failed
                    stats["skipped"] += num_failed

        return stats

    def process_all_text_data(self, base_path: str, update_mode: str = "skip") -> Dict[str, int]:
        """
        Process all text files and add to ChromaDB

        Args:
            base_path: Base directory containing data folders
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents (default)
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add

        Returns:
            Statistics about processed files
        """
        stats = {
            "files_processed": 0,
            "documents_added": 0,
            "documents_updated": 0,
            "documents_skipped": 0,
            "errors": 0,
            "total_chunks": 0,
            "missions": {},
        }

        # TODO->DONE: Get files to process
        files_to_process: List[Path] = self.scan_text_files_only(base_path)

        # TODO->DONE: Loop through each file
        for file_path in files_to_process:
            stats["files_processed"] += 1

            try:
                logger.info(
                    f"Processing file {file_path.name} from mission {self.extract_mission_from_path(file_path)}:"
                )

                # TODO->DONE: Process file and add to collection
                documents: List[Tuple[str, Dict[str, Any]]] = self.process_text_file(file_path)
                if not documents:
                    stats["errors"] += 1
                    continue
                logger.info(f"   Created {len(documents)} chunks")

                file_stats = self.add_documents_to_collection(documents, file_path, update_mode=update_mode)
                logger.info(f"   Chunks added to the collection {self.collection.name}")

                # TODO->DONE: Update statistics
                stats["documents_added"] += file_stats["added"]
                stats["documents_updated"] += file_stats["updated"]
                stats["documents_skipped"] += file_stats["skipped"]
                stats["total_chunks"] += len(documents)

                # mission breakdown
                mission = self.extract_mission_from_path(file_path)
                if mission not in stats["missions"]:
                    stats["missions"][mission] = {"files": 0, "chunks": 0, "added": 0, "updated": 0, "skipped": 0}
                stats["missions"][mission]["files"] += 1
                stats["missions"][mission]["chunks"] += len(documents)
                stats["missions"][mission]["added"] += file_stats["added"]
                stats["missions"][mission]["updated"] += file_stats["updated"]
                stats["missions"][mission]["skipped"] += file_stats["skipped"]

            # TODO->DONE: Handle errors gracefully
            except Exception as e:
                logger.error(f"Problem reading file {file_path.name}: {e}")
                stats["errors"] += 1

        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection"""
        # TODO->DONE: Return collection name, document count, metadata
        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count(),
            "metadata": self.collection.metadata or {},
        }

    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the collection for testing

        Args:
            query_text: Query text
            n_results: Number of results to return

        Returns:
            Query results
        """
        # TODO->DONE: Perform test query and return results
        try:
            # NOTE: Embedding done by ChromaDB
            # query_embeddings = self.get_embedding(query_text)
            # if not query_embeddings:
            #     logger.info("Failed to generate query embedding")
            #     return {}

            querry_result = self.collection.query(
                query_texts=query_text,
                n_results=n_results,
                where=None,
                include=["documents"],
            )

            return {"documents": querry_result["documents"]}

        except Exception as e:
            logger.error(f"Could not querry the collection: {e}")
            return {}

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection"""
        try:
            # Get all documents to analyze
            all_docs = self.collection.get()

            if not all_docs["metadatas"]:
                return {"error": "No documents in collection"}

            stats = {
                "total_documents": len(all_docs["metadatas"]),
                "missions": {},
                "data_types": {},
                "document_categories": {},
                "file_types": {},
            }

            # Analyze metadata
            for metadata in all_docs["metadatas"]:
                mission = metadata.get("mission", "unknown")
                data_type = metadata.get("data_type", "unknown")
                doc_category = metadata.get("document_category", "unknown")
                file_type = metadata.get("file_type", "unknown")

                # Count by mission
                stats["missions"][mission] = stats["missions"].get(mission, 0) + 1

                # Count by data type
                stats["data_types"][data_type] = stats["data_types"].get(data_type, 0) + 1

                # Count by document category
                stats["document_categories"][doc_category] = stats["document_categories"].get(doc_category, 0) + 1

                # Count by file type
                stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ChromaDB Embedding Pipeline for NASA Data")
    parser.add_argument("--data-path", default=str(Path(__file__).parents[1] / "data"), help="Path to data directories")
    parser.add_argument("--openai-key", default="", help="OpenAI API key")
    parser.add_argument("--chroma-dir", default="./voice_transcripts", help="ChromaDB persist directory")
    parser.add_argument("--collection-name", default="nasa_voice_transcripts", help="Collection name")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Text chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap size")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument(
        "--update-mode",
        choices=["skip", "update", "replace"],
        default="skip",
        help="How to handle existing documents: skip, update, or replace",
    )
    parser.add_argument("--test-query", help="Test query after processing")
    parser.add_argument("--stats-only", action="store_true", help="Only show collection statistics")
    parser.add_argument("--delete-source", help="Delete all documents from a specific source pattern")

    args = parser.parse_args()

    # Initialize pipeline
    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Handle delete source operation
    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(f"Deleted {deleted_count} documents matching source pattern: {args.delete_source}")
        return

    # If stats only, show collection statistics and exit
    if args.stats_only:
        logger.info("Collection Statistics:")
        stats = pipeline.get_collection_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        return

    # Process all data
    logger.info(f"Starting text data processing with update mode: {args.update_mode}")
    start_time = time.time()

    stats = pipeline.process_all_text_data(args.data_path, update_mode=args.update_mode)

    end_time = time.time()
    processing_time = end_time - start_time

    # Print results
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(f"Documents updated in collection: {stats['documents_updated']}")
    logger.info(f"Documents skipped (already exist): {stats['documents_skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")

    # Mission breakdown
    logger.info("\nMission breakdown:")
    for mission, mission_stats in stats["missions"].items():  # type: ignore
        logger.info(f"  {mission}: {mission_stats['files']} files, {mission_stats['chunks']} chunks")
        logger.info(
            f"    Added: {mission_stats['added']}, \
                Updated: {mission_stats['updated']}, \
                    Skipped: {mission_stats['skipped']}"
        )

    # Collection info
    collection_info = pipeline.get_collection_info()
    logger.info(f"\nCollection: {collection_info.get('collection_name', 'N/A')}")
    logger.info(f"Total documents in collection: {collection_info.get('document_count', 'N/A')}")

    # Test query if provided
    if args.test_query:
        logger.info(f"\nTesting query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        if results and "documents" in results:
            logger.info(f"Found {len(results['documents'][0])} results:")
            for i, doc in enumerate(results["documents"][0][:3]):  # Show top 3
                logger.info(f"Result {i + 1}: {doc[:200]}...")

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
