# Project: NASA Mission Intelligence

This project implements a Q&A system that can answer questions about some of NASA's most historic space missions, working with actual mission transcripts and technical documents from Apollo 11, Apollo 13, and the Challenger missions.

For this it uses a complete Retrieval-Augmented Generation (RAG) pipeline.

## Files
- data/ 
    - apollo11/: Contains the text files with data about the Apollo 11 mission and the corresponding metadata.
    - apollo13/: Contains the text files with data about the Apollo 13 mission and the corresponding metadata.
    - challenger/: - Contains the text files with data about the Challanger mission and the corresponding metadata.

- src/
    - chat.py: Streamlit app to run the chat with the LLM.
    - embedding_pipeline.py: Creates the vector embedding of the mission documents.
    - llm_client.py: Collects user input and RAG documents to generate a response.
    - rag_client.py: Searches the collection for documents matching the prompt.
    - ragas_evaluator.py: Evaluates the performance of the pipeline.

- test/
    - text_xx.py: Contains the unit tests for each elements of the pipeline (designed to not require LLM access).

- evaluation.
    - **evaluate_pipeline.py**: Performs an end-to-end test of the entire pipeline.
        - Must be triggered manually setting the the environment variable os.environ["OPENAI_API_KEY"] = "voc-..."
    - **evaluation_dataset.txt**: List of questions to evaluate. 
    - **evaluation_results.json**: Output of the evaluation.        

- vector_db/
    - unit_test/: Small collections created by the unit tests.    
    - voice_transcripts/: Collection created with the full dataset.

## Dependencies

Packages required for the main pipline:
```
openai
chromadb
numpy
pandas
```

Packages required for the RAGAS evaluation:
```
ragas
langchain-openai
rapidfuzz
rouge-score
sacrebleu
```

Packages required for the chat application:
```
streamlit
```

Packages required for the IDE
```
ruff
pyright
pytest
```

## Installation

There are two ways to install this project:
* Use uv (https://docs.astral.sh/uv/) and sync with pyproject.toml
* Run pip install -r /requirements.txt

## License

[License](LICENSE.txt)
