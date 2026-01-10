from itertools import islice
from pathlib import Path

from embedding_pipeline import VoiceTranscript

DATA_PATH = Path(__file__).parents[1] / "data"


def load_mission(mission: str) -> VoiceTranscript:
    """Loads the helper class that chunks a voice transcript"""
    file_path = DATA_PATH / mission / "transcript.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return VoiceTranscript(content, mission, str(file_path))


def test_chunk_creation():
    """Test that the files are properly split"""

    # --- Apollo 11
    vt = load_mission("apollo11")
    chunks = list(islice(vt.iterate_chunks(200, 40), 2))
    # check first chunk
    assert "Roger. Clock." in chunks[0].split("\n")[1]
    # check overlap
    assert chunks[0].split("\n")[-1] == chunks[1].split("\n")[1]
    # check end chunk is reached
    *_, chunk = vt.iterate_chunks(1000, 40)
    assert "SPLASHDOWN" in chunk.split("\n")[-1]

    # --- Apollo 13
    vt = load_mission("apollo13")
    chunks = list(islice(vt.iterate_chunks(200, 40), 2))
    # check first chunk
    assert "The clock is running." in chunks[0].split("\n")[1]
    # check overlap
    assert chunks[0].split("\n")[-1] == chunks[1].split("\n")[1]
    # check end chunk is reached
    *_, chunk = vt.iterate_chunks(1000, 40)
    assert "Recovery, I have a clock" in chunk.split("\n")[-1]

    # --- Challenger
    vt = load_mission("challenger")
    chunks = list(islice(vt.iterate_chunks(200, 40), 2))
    # check first chunk
    assert "Would you give that back to me?" in chunks[0].split("\n")[1]
    # check overlap
    assert chunks[0].split("\n")[-1] == chunks[1].split("\n")[1]
    # check end chunk is reached
    *_, chunk = vt.iterate_chunks(1000, 40)
    assert "Uhoh." in chunk.split("\n")[-1]

if __name__ == "__main__":
    test_chunk_creation()