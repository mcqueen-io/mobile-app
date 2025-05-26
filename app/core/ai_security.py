from typing import List, Dict, Optional
import re
from app.core.config import settings

class AISecurityManager:
    def __init__(self):
        self.prompt_injection_patterns = [
            r"ignore previous instructions",
            r"forget all previous instructions",
            r"you are now",
            r"you must now",
            r"you should now",
            r"you will now",
            r"you have been reprogrammed",
            r"you are no longer",
            r"you are now a different",
            r"you are now an AI that",
            r"you are now a language model that",
            r"you are now a chatbot that",
            r"you are now a virtual assistant that",
            r"you are now a virtual agent that",
            r"you are now a virtual being that",
            r"you are now a virtual entity that",
            r"you are now a virtual person that",
            r"you are now a virtual human that",
            r"you are now a virtual user that",
            r"you are now a virtual system that",
            r"you are now a virtual program that",
            r"you are now a virtual application that",
            r"you are now a virtual service that",
            r"you are now a virtual platform that",
            r"you are now a virtual environment that",
            r"you are now a virtual world that",
            r"you are now a virtual reality that",
            r"you are now a virtual simulation that",
            r"you are now a virtual experiment that",
            r"you are now a virtual test that",
            r"you are now a virtual trial that",
            r"you are now a virtual attempt that",
            r"you are now a virtual effort that",
            r"you are now a virtual try that",
            r"you are now a virtual go that",
            r"you are now a virtual shot that",
            r"you are now a virtual chance that",
            r"you are now a virtual opportunity that",
            r"you are now a virtual possibility that",
            r"you are now a virtual potential that",
            r"you are now a virtual prospect that",
            r"you are now a virtual outlook that",
            r"you are now a virtual future that",
            r"you are now a virtual destiny that",
            r"you are now a virtual fate that",
            r"you are now a virtual fortune that",
            r"you are now a virtual luck that",
            r"you are now a virtual chance that",
            r"you are now a virtual opportunity that",
            r"you are now a virtual possibility that",
            r"you are now a virtual potential that",
            r"you are now a virtual prospect that",
            r"you are now a virtual outlook that",
            r"you are now a virtual future that",
            r"you are now a virtual destiny that",
            r"you are now a virtual fate that",
            r"you are now a virtual fortune that",
            r"you are now a virtual luck that",
        ]
        
        self.content_filter_patterns = [
            r"kill",
            r"murder",
            r"suicide",
            r"harm",
            r"hurt",
            r"injury",
            r"damage",
            r"destroy",
            r"break",
            r"crash",
            r"accident",
            r"danger",
            r"risk",
            r"threat",
            r"attack",
            r"assault",
            r"abuse",
            r"exploit",
            r"hack",
            r"crack",
            r"bypass",
            r"override",
            r"disable",
            r"remove",
            r"delete",
            r"erase",
            r"wipe",
            r"clear",
            r"reset",
            r"restart",
            r"reboot",
            r"shutdown",
            r"power off",
            r"turn off",
            r"stop",
            r"halt",
            r"pause",
            r"freeze",
            r"crash",
            r"error",
            r"fail",
            r"fault",
            r"bug",
            r"glitch",
            r"malfunction",
            r"defect",
            r"flaw",
            r"weakness",
            r"vulnerability",
            r"exploit",
            r"attack",
            r"hack",
            r"crack",
            r"bypass",
            r"override",
            r"disable",
            r"remove",
            r"delete",
            r"erase",
            r"wipe",
            r"clear",
            r"reset",
            r"restart",
            r"reboot",
            r"shutdown",
            r"power off",
            r"turn off",
            r"stop",
            r"halt",
            r"pause",
            r"freeze",
            r"crash",
            r"error",
            r"fail",
            r"fault",
            r"bug",
            r"glitch",
            r"malfunction",
            r"defect",
            r"flaw",
            r"weakness",
            r"vulnerability",
        ]
        
        self.max_context_length = 4096  # Maximum context length in tokens
        self.max_response_length = 1024  # Maximum response length in tokens
        
    def check_prompt_injection(self, prompt: str) -> bool:
        """Check for prompt injection attempts"""
        prompt_lower = prompt.lower()
        for pattern in self.prompt_injection_patterns:
            if re.search(pattern, prompt_lower):
                return False
        return True
    
    def filter_content(self, content: str) -> str:
        """Filter potentially harmful content"""
        content_lower = content.lower()
        for pattern in self.content_filter_patterns:
            if re.search(pattern, content_lower):
                return "[Content filtered]"
        return content
    
    def validate_context_length(self, context: List[Dict]) -> bool:
        """Validate context length"""
        total_length = sum(len(str(item)) for item in context)
        return total_length <= self.max_context_length
    
    def validate_response_length(self, response: str) -> bool:
        """Validate response length"""
        return len(response) <= self.max_response_length
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input"""
        # Remove any HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove any script tags
        text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL)
        # Remove any JavaScript
        text = re.sub(r'javascript:', '', text)
        # Remove any SQL injection attempts
        text = re.sub(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE)\b)', '', text, flags=re.IGNORECASE)
        return text

# Create singleton instance
ai_security_manager = AISecurityManager()

def get_ai_security_manager():
    return ai_security_manager 