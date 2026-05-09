import asyncio
from functools import partial
from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse
from core.graph_manager import GraphManager

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    mgr = GraphManager.get_instance()
    if mgr is None:
        raise HTTPException(status_code=503, detail="Index not ready. Complete setup first.")
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            partial(mgr.invoke, req.thread_id, req.message, req.is_first_message),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    messages = result.get("messages", [])
    reply = messages[-1].content if messages else "No response generated."

    return ChatResponse(
        reply=reply,
        liked_genres=result.get("liked_genres", []),
        seen_count=len(result.get("seen_titles", [])),
        intent=result.get("intent", ""),
    )
