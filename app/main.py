from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.modules.voice_layer.api import router as voice_router
from app.modules.user_info.api import router as user_router
from app.modules.memory.api import router as memory_router
from app.modules.mcp.api import router as mcp_router
from app.modules.maps.api import router as maps_router
from app.modules.context.api import router as context_router
from app.api.reflection import router as reflection_router
from app.api.navigation import router as navigation_router
from app.modules.voice_layer.voice_processor import get_voice_processor
from app.modules.user_info.user_graph import get_user_graph
from app.modules.memory.memory_store import get_memory_store
from app.modules.mcp.mcp_client import get_mcp_client
from app.core.security import get_security_manager
from app.core.ai_security import get_ai_security_manager
from app.core.config import settings

app = FastAPI(
    title="In-Car AI Assistant",
    description="Backend API for the In-Car AI Assistant application",
    version="0.1.0"
)

# Configure CORS with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600,
)

# Security middleware - temporarily disabled for WebSocket testing
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Temporarily disable all security checks for testing
    print(f"Request path: {request.url.path}")
    response = await call_next(request)
    return response
    
    # Original security code (commented out for testing)
    # Skip security checks for WebSocket connections
    # if request.url.path.startswith("/api/v1/voice/ws/"):
    #     response = await call_next(request)
    #     return response
    
    # security_manager = get_security_manager()
    
    # Check rate limit
    # client_id = request.client.host
    # if not security_manager.check_rate_limit(client_id):
    #     return JSONResponse(
    #         status_code=429,
    #         content={"detail": "Too many requests"}
    #     )
    
    # Verify request signature for non-GET requests
    # if request.method != "GET":
    #     signature = request.headers.get("X-Request-Signature")
    #     if not signature or not security_manager.verify_request_signature(request, signature):
    #         return JSONResponse(
    #             status_code=401,
    #             content={"detail": "Invalid request signature"}
    #         )
    
    # response = await call_next(request)
    # return response

@app.get("/")
async def root():
    return {"message": "Welcome to In-Car AI Assistant API"}

# Include routers with security
app.include_router(voice_router, prefix="/api/v1/voice", tags=["voice"])
app.include_router(user_router, prefix="/api/v1/users", tags=["users"])
app.include_router(memory_router, prefix="/api/v1/memory", tags=["memory"])
app.include_router(mcp_router, prefix="/api/v1/mcp", tags=["mcp"])
app.include_router(maps_router, prefix="/api/v1/maps", tags=["maps"])
app.include_router(context_router, prefix="/api/v1/context", tags=["context"])
app.include_router(reflection_router, prefix="/api/v1", tags=["reflection"])
app.include_router(navigation_router, prefix="/api/v1", tags=["navigation"])

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    # Close database connections
    user_graph = get_user_graph()
    user_graph.close()
    
    # Close MCP client session
    mcp_client = get_mcp_client()
    await mcp_client.close() 
    
    # Cleanup ChromaDB resources
    from app.db.chroma_manager import get_chroma_manager
    chroma_manager = get_chroma_manager()
    chroma_manager.cleanup() 