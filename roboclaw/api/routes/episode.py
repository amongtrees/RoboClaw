"""Episode history routes: CRUD, search, export."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

# In-memory episode store for Phase 1
_episodes: dict[str, dict] = {}


@router.get("/episodes/{episode_id}")
async def get_episode(episode_id: str):
    """Get a full episode with all events."""
    episode = _episodes.get(episode_id)
    if not episode:
        return {"error": f"Episode '{episode_id}' not found"}, 404
    return episode


@router.get("/episodes/{episode_id}/events")
async def get_episode_events(episode_id: str):
    """Get the timeline of events for an episode."""
    episode = _episodes.get(episode_id)
    if not episode:
        return {"error": f"Episode '{episode_id}' not found"}, 404
    return episode.get("events", [])


@router.get("/episodes/{episode_id}/artifacts")
async def get_episode_artifacts(episode_id: str):
    """List artifacts associated with an episode."""
    episode = _episodes.get(episode_id)
    if not episode:
        return {"error": f"Episode '{episode_id}' not found"}, 404
    return episode.get("artifacts", [])


@router.get("/episodes")
async def list_episodes(
    robot_id: str | None = None,
    task_type: str | None = None,
    outcome: str | None = None,
    limit: int = 20,
    offset: int = 0,
):
    """Search episodes with optional filters."""
    results = list(_episodes.values())
    if robot_id:
        results = [e for e in results if e.get("robot_id") == robot_id]
    if task_type:
        results = [e for e in results if e.get("task_type") == task_type]
    if outcome:
        results = [e for e in results if e.get("outcome") == outcome]

    results.sort(key=lambda e: e.get("started_at", 0), reverse=True)
    return results[offset : offset + limit]


@router.post("/episodes/search")
async def semantic_search_episodes(body: dict):
    """Semantic search over episodes (uses Milvus embeddings).

    Body:
        query: str — natural language search query
        top_k: int — number of results (default 10)
    """
    query = body.get("query", "")
    top_k = body.get("top_k", 10)

    # Phase 1 stub: simple text search over stored episodes
    results = []
    for ep in _episodes.values():
        text = f"{ep.get('goal', '')} {ep.get('summary', '')}"
        if query.lower() in text.lower():
            results.append(ep)

    return results[:top_k]
