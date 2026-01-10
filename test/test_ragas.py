import ragas_evaluator

RAG_DOCUMENTS = [
    """**Source 1** — Mission: Apollo 11 | Source: Transcript | Category: General
[apollo_11: 1969-07-16 13:32:04]
Neil: Roger. Clock.
Neil: Roger. We got a roll program.
Mike: Roger. Roll.
Neil: Roll's complete and the pitch is programed.
Neil: One Bravo.
Neil: Roger.
Houston: Stand by for mode 1 Charlie.
Houston: MARK.""",
    """**Source 2** — Mission: Apollo 13 | Source: Transcript | Category: General
[apollo_13: 1970-04-11 19:13:02]
Jim: The clock is running.
Joe: Okay. P11, Jim.
Jim: Yaw program.
Joe: Clear the tower.
Jim: Yaw complete. Roll program.
Houston: Houston, Roger. Roll.
Houston: 13, Houston. GO at 30 seconds.
Jim: Roll complete, and we are pitching.
Houston: Roger that. Stand by for mode I Bravo.
Houston: MARK.""",
]


def test_ragas():
    """Test that RAGAS runs correctly"""

    question = "Who said: 'Houston, Roger that'?"
    answer = "Houston says Roger in the Apollo 13 mission, but int he Apollo 11 mission it is Neil"
    reference_answer = "Houston and Neil say Roger in the missions"

    metrics = ragas_evaluator.evaluate_response_quality(
        question,
        answer,
        RAG_DOCUMENTS,
        reference_answer=reference_answer,
        reference_contexts=[RAG_DOCUMENTS[0]],
        use_llm_metrics=False,
    )

    assert "error" not in metrics.keys()
