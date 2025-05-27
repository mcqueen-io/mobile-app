from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict
from app.core.config import settings
import time
from functools import wraps
import hashlib
import hmac

security = HTTPBearer()

class SecurityManager:
    def __init__(self):
        self.rate_limit_window = 60  # 1 minute
        self.max_requests = 100
        self.request_counts: Dict[str, list] = {}
        
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")

    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            return payload
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    def check_rate_limit(self, client_id: str) -> bool:
        current_time = time.time()
        if client_id not in self.request_counts:
            self.request_counts[client_id] = []
        
        # Remove old requests
        self.request_counts[client_id] = [
            t for t in self.request_counts[client_id]
            if current_time - t < self.rate_limit_window
        ]
        
        if len(self.request_counts[client_id]) >= self.max_requests:
            return False
            
        self.request_counts[client_id].append(current_time)
        return True

    async def verify_request_signature(self, request: Request, signature: str) -> bool:
        # Get request body
        body = await request.body()
        
        # Create HMAC signature
        expected_signature = hmac.new(
            settings.SECRET_KEY.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)

def require_auth():
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if not request:
                raise HTTPException(status_code=400, detail="Request object not found")
            
            # Check rate limit
            client_id = request.client.host
            if not security_manager.check_rate_limit(client_id):
                raise HTTPException(status_code=429, detail="Too many requests")
            
            # Verify authentication
            try:
                auth = await security(request)
                payload = security_manager.verify_token(auth.credentials)
                request.state.user = payload
            except Exception as e:
                raise HTTPException(status_code=401, detail=str(e))
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Create singleton instance
security_manager = SecurityManager()

def get_security_manager():
    return security_manager 