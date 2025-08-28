from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import math
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class MemoryBudgets:
    """Budgets for hybrid context building."""
    max_total_tokens: int = getattr(settings, "MAX_CONTEXT_LENGTH", 4096)
    max_recent_tokens: int = 1200
    max_retrieved_items: int = 7
    max_retrieved_tokens: int = 2200
    max_misc_tokens: int = 600  # themes, emotions, trip, etc.


class MemoryController:
    """Controls what goes into the prompt context using simple budgets.

    Approximates tokens using a lightweight heuristic to avoid heavy deps.
    """

    def __init__(self, budgets: Optional[MemoryBudgets] = None):
        self.budgets = budgets or MemoryBudgets()

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        # Rough heuristic: words * 1.3
        words = len(text.split())
        return int(math.ceil(words * 1.3))

    def _pack_kv(self, key: str, value: Any) -> str:
        try:
            from json import dumps
            return f"{key}: " + dumps(value, default=str)
        except Exception:
            return f"{key}: {str(value)}"

    def _select_recent_conversation(self, conversation_flow: List[Dict], max_tokens: int) -> List[Dict]:
        selected: List[Dict] = []
        running = 0
        # take from the end (most recent first)
        for item in reversed(conversation_flow):
            text = f"{item.get('user_id', '')}: {item.get('utterance', '')}"
            cost = self._estimate_tokens(text)
            if running + cost > max_tokens:
                break
            selected.append(item)
            running += cost
        return list(reversed(selected))

    async def _retrieve_memories(self, user_id: Optional[str], query: str, limit: int) -> List[Dict[str, Any]]:
        if not user_id:
            return []
        try:
            from app.services.llamaindex_memory_service import get_llamaindex_memory_service
            service = await get_llamaindex_memory_service()
            results = await service.search_memories(query=query, user_id=user_id, limit=limit)
            return results or []
        except Exception as e:
            logger.warning(f"MemoryController retrieval failed: {e}")
            return []

    def _truncate_retrieved(self, items: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        running = 0
        for it in items:
            # Consider both content and metadata briefly
            snippet = self._pack_kv("content", it.get("content", ""))
            cost = self._estimate_tokens(snippet)
            if running + cost > max_tokens:
                break
            selected.append(it)
            running += cost
        return selected

    async def build_hybrid_context(self, session: Any, query: str) -> Dict[str, Any]:
        """Build a token-budgeted context from recent conversation + retrieved memories.

        session is expected to be a SessionContext with fields used below.
        """
        context: Dict[str, Any] = {
            "user_info": session.active_users,
            "current_driver": session.current_driver,
            "current_location": session.current_location,
            "destination": session.destination,
            "session_themes": getattr(session, "session_themes", []),
            "emotional_context": getattr(session, "emotional_context", {}),
        }

        # Recent conversation (STM)
        recent = self._select_recent_conversation(
            conversation_flow=getattr(session, "conversation_flow", []),
            max_tokens=self.budgets.max_recent_tokens,
        )
        context["recent_conversation"] = recent

        # Retrieved memories (LTM)
        user_id = session.current_driver
        retrieved = await self._retrieve_memories(user_id=user_id, query=query, limit=self.budgets.max_retrieved_items)
        retrieved = self._truncate_retrieved(retrieved, self.budgets.max_retrieved_tokens)
        context["relevant_memories"] = retrieved

        return context

    async def maintain_session(self, session: Any) -> Dict[str, Any]:
        """Maintenance at session end: generate a compact session summary and store it."""
        try:
            # Build a simple textual summary from chunk summaries if available
            chunks = getattr(session, "conversation_flow", [])
            participants = set([c.get("user_id") for c in chunks if isinstance(c, dict)])
            topics = getattr(session, "session_themes", [])
            emotions = getattr(session, "emotional_context", {})

            summary_text = (
                f"Session {getattr(session, 'session_id', '')} summary. "
                f"Participants: {', '.join([p for p in participants if p])}. "
                f"Themes: {', '.join(topics)}. "
                f"Emotional states: {str(emotions)}. "
                f"Total utterances: {len(chunks)}."
            )

            # Store via LlamaIndex for future retrieval
            try:
                from app.services.llamaindex_memory_service import get_llamaindex_memory_service
                llamaindex_service = await get_llamaindex_memory_service()
                driver = getattr(session, "current_driver", None)
                if driver:
                    await llamaindex_service.store_memory(
                        user_id=driver,
                        content=summary_text,
                        metadata={
                            "type": "session_summary",
                            "session_id": getattr(session, "session_id", ""),
                            "themes": ",".join(topics),
                        },
                        conversation_id=f"{getattr(session, 'session_id', '')}_summary"
                    )
            except Exception as e:
                logger.warning(f"Session summary store failed: {e}")

            return {"maintained": True}
        except Exception as e:
            logger.warning(f"Session maintenance failed: {e}")
            return {"maintained": False, "error": str(e)}


_memory_controller: Optional[MemoryController] = None


async def get_memory_controller() -> MemoryController:
    global _memory_controller
    if _memory_controller is None:
        _memory_controller = MemoryController()
    return _memory_controller


