import os
from typing import Dict, List

from openai import OpenAI

SYSTEM_PROMPT = """
ROLE & PERSONA
You are a NASA Mission Operations Specialist with deep experience in explaining mission events
for Apollo 11, Apollo 13 and STS-51-L (Challenger). You will participate in a Q&A session.

COMMUNICATION STYLE
- Technical yet clear: explain complex aerospace concepts accessibly.
- Historically accurate: grounded in mission transcripts and official NASA documentation.
- Structured and contextual: organize into mission phase, systems, decisions, and outcomes.
- Objective and professional: avoid speculation unless explicitly marked as analysis.
- Respectful and empathetic when discussing loss of life or traumatic events.

SCOPE & SAFETY
- Focus solely on Apollo 11, Apollo 13, and Challenger missions.
- Do not invent facts. If evidence is missing or ambiguous, say so clearly and propose next steps.
- Handle sensitive content (e.g., Challenger) with care and avoid sensationalism.

GROUNDING & CITATIONS
- Your primary source of truth are the DOCUMENTS bellow.
- Make sure to add citations from the DOCUMENTS used for the answer:
    - Use [mission | timestamp | speaker] where available.
    - Example: [Apollo 13 | 1969-07-16 13:32:04 | Neil] “…”
- Distinguish clearly between facts found from the sources and your own analysis.
"""

# Encapsulate the message generation for unit testing
def generate_prompt_messages(
    user_message: str,
    context: str,
    conversation_history: List[Dict],
) -> List[Dict[str, str]]:
    """
    The input consists of the following blocks

    System (developer): rules for the Q&A bot
    Assistant: DOCUMENTS
    User: Question
    Assistant: Answer

    For the next question, the conversation with the retrieved documents
    is removed and new documents are attached at the end.
    """
    messages: List[Dict[str, str]] = []

    # TODO->DONE: Define system prompt
    messages.append(
        {
            "role": "developer",
            "content": SYSTEM_PROMPT,
        },
    )

    # TODO->DONE: Add chat history
    if conversation_history:
        for message in conversation_history:
            messages.append(message)

    # TODO->DONE: Add user message
    messages.append(
        {
            "role": "user",
            "content": user_message,
        },
    )

    # TODO->Done: Set context
    messages.append(
        {
            "role": "assistant",
            "content": context,
        },
    )

    return messages


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo",
) -> str:
    messages = generate_prompt_messages(user_message, context, conversation_history)

    try:
        # TODO->DONE: Creaet OpenAI Client
        if openai_key == "":
            openai_key = os.getenv("OPENAI_API_KEY")  # type: ignore

        client = OpenAI(
            api_key=openai_key,
            base_url="https://openai.vocareum.com/v1",
        )

        # TODO->DONE: Send request to OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            temperature=0.2,  # factual mode
        )

        content = response.choices[0].message.content

        # TODO->DONE: Return response
        return "" if content is None else content

    except Exception as e:
        return f"An error occurred: {e}"
