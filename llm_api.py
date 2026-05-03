# llm_api.py

import os
import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=os.environ.get("GROQ_API_KEY", api_key))

def classify_with_api(user_message: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a movie query parser. "
                    "Reply with JSON only.\n\n"
                    "Schema:\n"
                    "{\n"
                    '  "intent": "recommend" | "refine" | "chitchat",\n'
                    '  "genres": ["action", "sci-fi", ...],\n'
                    '  "min_year": int or null,\n'
                    '  "max_year": int or null,\n'
                    '  "min_rating": float or null,\n'
                    '  "max_rating": float or null\n'
                    "}\n\n"
                    "Rules:\n"
                    '- intent is "recommend" if user wants movie suggestions\n'
                    '- intent is "refine" if user wants different/more results\n'
                    '- intent is "chitchat" for anything else\n'
                    "- genres: extract any film genres mentioned\n"
                    "- extract year ranges and rating filters from the message\n"
                    "- use null for unspecified filters"
                )
            },
            {"role": "user", "content": user_message}
        ]
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    return {
        "intent":     parsed.get("intent", "chitchat"),
        "genres":     parsed.get("genres", []),
        "min_year":   parsed.get("min_year"),
        "max_year":   parsed.get("max_year"),
        "min_rating": parsed.get("min_rating"),
        "max_rating": parsed.get("max_rating"),
    }


def generate_recommendation_with_api(
    user_query: str,
    context: str,
    liked_genres: list[str],
    conversation_history: list[dict]
) -> str:
    """
    Send movie context + user query to Groq API.
    Returns a conversational recommendation response.
    """
    pref_note = (
        f"The user likes: {', '.join(liked_genres)}. "
        if liked_genres else ""
    )

    # Build conversation for the API
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly movie recommendation assistant.\n"
                f"{pref_note}\n"
                "RULES:\n"
                "- Use ONLY the movies from the Context below\n"
                "- List EVERY movie from the context, numbered\n"
                "- Do NOT skip any movie\n"
                "- Do NOT repeat any movie\n"
                "- Do NOT invent or hallucinate movies\n"
                "- For each movie: title, year, genre, rating, "
                "and one sentence about the plot\n"
                "- Keep descriptions concise\n"
                "- If the user asks a follow-up question about a movie, "
                "answer it using the context\n\n"
                f"Context:\n{context}"
            )
        }
    ]

    # Add conversation history (last 6 messages for context)
    for msg in conversation_history[-6:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append(msg)

    # Add current query
    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=2000,
        messages=messages
    )

    return response.choices[0].message.content


# Quick test
if __name__ == "__main__":
    # Test classify
    test_queries = [
        "list movies from action genre from year 2008 to 2015 and rating greater than 7",
        "suggest some good horror films",
        "what is the meaning of life",
    ]

    print("=" * 60)
    print("CLASSIFY TEST")
    print("=" * 60)
    for q in test_queries:
        result = classify_with_api(q)
        print(f"\nQuery: {q}")
        print(f"Result: {json.dumps(result, indent=2)}")

    # Test generate
    print("\n" + "=" * 60)
    print("GENERATE TEST")
    print("=" * 60)

    fake_context = """[1] I Saw the Devil (2010)
    Genre: Action, Crime, Drama
    Rating: 7.8/10
    Plot: A secret agent exacts revenge on a serial killer.

[2] Furious 7 (2015)
    Genre: Action, Crime, Thriller
    Rating: 7.1/10
    Plot: Deckard Shaw seeks revenge against Dominic Toretto."""

    reply = generate_recommendation_with_api(
        user_query="list action movies from 2008 to 2015 rated above 7",
        context=fake_context,
        liked_genres=["action"],
        conversation_history=[]
    )
    print(f"\nResponse:\n{reply}")
