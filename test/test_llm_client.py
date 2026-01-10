import llm_client

RAG_CONTEXT = """DOCUMENTS

**Source 1** — Mission: Apollo 11 | Source: Transcript | Category: General
[apollo_11: 1969-07-16 13:32:04]
Neil: Roger. Clock.
Neil: Roger. We got a roll program.
Mike: Roger. Roll.
Neil: Roll's complete and the pitch is programed.
Neil: One Bravo.
Neil: Roger.
Houston: Stand by for mode 1 Charlie.
Houston: MARK.

**Source 2** — Mission: Apollo 13 | Source: Transcript | Category: General
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
Houston: MARK.
"""


def test_llm_promt():
    """Test the prompt created for the llm"""

    # Chat history
    conversation_history = [
        {
            "role": "user",
            "content": "Hi there! Who are you?",
        },
        {
            "role": "assistant",
            "content": "I am a NASA Mission Operations Specialist",
        },
    ]
    # New message
    user_message = "Who said: 'Houston, Roger that'?"

    # generate message
    messages = llm_client.generate_prompt_messages(user_message, RAG_CONTEXT, conversation_history)

    # check roles and added user question and context
    for actual, expected in zip([m["role"] for m in messages], ["developer", "user", "assistant", "user", "assistant"]):
        assert actual == expected
    assert messages[-2]["content"] == user_message
    assert messages[-1]["content"] == RAG_CONTEXT
