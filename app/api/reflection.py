"""
Reflection API - Monitor Queen's Self-Improvement Journey
"""

from fastapi import APIRouter
from app.modules.ai_wrapper.reflection_manager import get_reflection_manager
from typing import Dict, Any

router = APIRouter()

@router.get("/reflection/summary")
async def get_reflection_summary() -> Dict[str, Any]:
    """Get Queen's learning and improvement summary"""
    reflection_manager = get_reflection_manager()
    summary = reflection_manager.get_reflection_summary()
    
    return {
        "status": "success",
        "queen_learning_progress": summary,
        "description": "Queen's self-improvement metrics and learning insights"
    }

@router.get("/reflection/insights")
async def get_learning_insights() -> Dict[str, Any]:
    """Get recent learning insights from Queen's interactions"""  
    reflection_manager = get_reflection_manager()
    
    # Get recent interactions for insights
    recent_interactions = reflection_manager.interaction_history[-5:] if reflection_manager.interaction_history else []
    
    insights = []
    for interaction in recent_interactions:
        if interaction.get("reflection_score", 0) > 0.7:
            insights.append({
                "timestamp": interaction["timestamp"],
                "user_query": interaction["user_input"][:100] + "..." if len(interaction["user_input"]) > 100 else interaction["user_input"],
                "performance_score": interaction["reflection_score"],
                "learning_note": "High-quality interaction - patterns captured"
            })
    
    return {
        "status": "success", 
        "recent_learning_insights": insights,
        "improvement_areas": [
            "Context understanding",
            "Personalized responses", 
            "Safety awareness",
            "Tool usage optimization"
        ]
    }

@router.post("/reflection/feedback")
async def provide_feedback(feedback_data: Dict[str, Any]) -> Dict[str, Any]:
    """Allow users to provide feedback on Queen's responses for learning"""
    reflection_manager = get_reflection_manager()
    
    user_input = feedback_data.get("user_input", "")
    ai_response = feedback_data.get("ai_response", "")
    user_feedback = feedback_data.get("feedback", "")
    rating = feedback_data.get("rating", 3)  # 1-5 scale
    
    # Process feedback for learning
    await reflection_manager.post_response_reflection(
        user_input, ai_response, user_feedback
    )
    
    return {
        "status": "success",
        "message": "Thank you! Queen has learned from your feedback.",
        "feedback_processed": True
    }

@router.get("/reflection/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get Queen's performance metrics over time"""
    reflection_manager = get_reflection_manager()
    
    if not reflection_manager.interaction_history:
        return {
            "status": "success",
            "message": "No interaction data available yet",
            "metrics": {"total_interactions": 0}
        }
    
    # Calculate performance metrics
    total_interactions = len(reflection_manager.interaction_history)
    scores = [i.get("reflection_score", 0.5) for i in reflection_manager.interaction_history]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    recent_scores = scores[-10:] if len(scores) >= 10 else scores
    recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0
    
    return {
        "status": "success",
        "performance_metrics": {
            "total_interactions": total_interactions,
            "overall_avg_score": round(avg_score, 3),
            "recent_avg_score": round(recent_avg, 3),
            "improvement_trend": reflection_manager._calculate_improvement_trend(),
            "performance_grade": "A" if recent_avg > 0.8 else "B" if recent_avg > 0.6 else "C"
        }
    } 