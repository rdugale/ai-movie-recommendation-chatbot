import uuid
from fastapi import APIRouter
from models.schemas import NewSessionResponse, SessionStateResponse
from core.graph_manager import GraphManager

router = APIRouter()


@router.post("/session/new", response_model=NewSessionResponse)
async def new_session():
    return NewSessionResponse(thread_id=str(uuid.uuid4()))


@router.get("/session/{thread_id}/state", response_model=SessionStateResponse)
async def session_state(thread_id: str):
    mgr = GraphManager.get_instance()
    values = mgr.get_state(thread_id) if mgr else {}
    return SessionStateResponse(
        thread_id=thread_id,
        liked_genres=values.get("liked_genres", []),
        seen_count=len(values.get("seen_titles", [])),
        message_count=len(values.get("messages", [])),
    )
