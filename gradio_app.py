import gradio as gr
from langchain_core.messages import HumanMessage
from chatbot import build_graph

graph = build_graph()

SESSION_CONFIG = {"configurable": {"thread_id": "gradio-session-1"}}
first_run = True

def respond(user_message, history):
    global first_run

    if first_run:
        input_state = {
            "intent":       "",
            "context":      "",
            "liked_genres": [],
            "seen_titles":  [],
            "last_query":   "",
            "messages":     [HumanMessage(content=user_message)],
        }
        first_run = False
    else:
        input_state = {"messages": [HumanMessage(content=user_message)]}

    graph.invoke(input_state, config=SESSION_CONFIG)
    result = graph.get_state(SESSION_CONFIG).values
    reply  = result["messages"][-1].content
    liked  = result.get("liked_genres", [])

    footer = f"\n\n*Preferences learned: {', '.join(liked)}*" if liked else ""
    return reply + footer

demo = gr.ChatInterface(
    fn=respond,
    title="🎬 Movie Recommendation Chatbot",
    description="Powered by LangGraph + RAG + phi3:mini — fully local, no API needed",
    examples=[
        "Recommend some thriller movies",
        "I love sci-fi and action films, what should I watch?",
        "Show me something different",
        "What makes Christopher Nolan films special?",
    ],
)

if __name__ == "__main__":
    demo.launch()