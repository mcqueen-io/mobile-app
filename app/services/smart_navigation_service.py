from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import asyncio
from datetime import datetime, timedelta
from app.services.unified_service import get_unified_service
from app.services.offline_navigation_cache import get_offline_navigation_cache
from app.db.mongo_manager import get_mongo_manager
from app.core.config import settings
import aiohttp

logger = logging.getLogger(__name__)

class SmartNavigationService:
    """
    AI-powered navigation enhancement service that provides natural language,
    context-aware directions and proactive clarifications for confusing locations.
    """
    
    def __init__(self):
        self.mongo_manager = None
        self.unified_service = None
        self.offline_cache = None
        # Use existing Google Search API key for Maps API (same project, same key works)
        self.maps_api_key = settings.GOOGLE_SEARCH_API_KEY or settings.GOOGLE_MAPS_API_KEY
        
        # Database of known confusing locations with enhanced descriptions
        self.confusing_locations_db = {}
        
        # Common problematic location types
        self.problematic_categories = [
            "airport", "rental_car", "parking_garage", "mall", "hospital",
            "university_campus", "stadium", "convention_center", "highway_exit"
        ]

    async def initialize(self):
        """Initialize dependencies and load confusing locations database"""
        self.mongo_manager = await get_mongo_manager()
        self.unified_service = await get_unified_service()
        self.offline_cache = await get_offline_navigation_cache()
        await self.load_confusing_locations()

    async def load_confusing_locations(self):
        """Load database of locations known to be confusing with enhanced descriptions"""
        try:
            # Load from MongoDB collection: confusing_locations
            confusing_data = await self.mongo_manager.db.confusing_locations.find().to_list(length=None)
            for location in confusing_data:
                location_id = f"{location['lat']},{location['lng']}"
                self.confusing_locations_db[location_id] = location
                
            logger.info(f"Loaded {len(self.confusing_locations_db)} confusing locations")
            
        except Exception as e:
            logger.error(f"Error loading confusing locations: {str(e)}")
            # Initialize with empty database
            self.confusing_locations_db = {}

    async def enhance_directions(
        self, 
        origin: str, 
        destination: str, 
        user_context: Dict[str, Any] = None,
        next_destination: str = None
    ) -> Dict[str, Any]:
        """
        Get directions from Google Maps and enhance them with AI-powered natural language
        and context-aware clarifications.
        """
        try:
            # 1. Get standard directions from Google Maps
            standard_directions = await self.get_google_directions(origin, destination)
            
            if not standard_directions.get('routes'):
                return {"error": "No route found"}
            
            route = standard_directions['routes'][0]
            steps = route['legs'][0]['steps']
            
            # 2. Analyze route for potentially confusing areas
            confusing_steps = await self.identify_confusing_steps(steps)
            
            # 3. Enhance directions with AI natural language
            enhanced_steps = await self.enhance_steps_with_ai(steps, confusing_steps, user_context)
            
            # 4. Search web for location-specific issues and tips
            web_insights = await self.get_location_specific_insights(destination, user_context)
            
            # 5. Generate proactive warnings and tips
            proactive_guidance = await self.generate_proactive_guidance(route, user_context, next_destination)
            
            return {
                "original_route": route,
                "enhanced_steps": enhanced_steps,
                "proactive_guidance": proactive_guidance,
                "location_insights": web_insights,
                "total_distance": route['legs'][0]['distance']['text'],
                "total_duration": route['legs'][0]['duration']['text'],
                "confusing_areas_count": len(confusing_steps),
                "next_destination": next_destination
            }
            
        except Exception as e:
            logger.error(f"Error enhancing directions: {str(e)}")
            return {"error": str(e)}

    async def get_google_directions(self, origin: str, destination: str) -> Dict[str, Any]:
        """Get directions from Google Maps Directions API"""
        # Check if we have a valid API key
        if not self.maps_api_key or self.maps_api_key == "":
            logger.warning("No Google Maps API key found, using mock data for testing")
            return self._get_mock_directions(origin, destination)
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://maps.googleapis.com/maps/api/directions/json"
                params = {
                    "origin": origin,
                    "destination": destination,
                    "key": self.maps_api_key,
                    "mode": "driving",
                    "alternatives": "true",
                    "traffic_model": "best_guess",
                    "departure_time": "now"
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Google Directions API error: {error_text}")
                        # Fallback to mock data if API fails
                        return self._get_mock_directions(origin, destination)
                        
        except Exception as e:
            logger.error(f"Error calling Google Directions API: {str(e)}")
            # Fallback to mock data if API fails
            return self._get_mock_directions(origin, destination)
    
    def _get_mock_directions(self, origin: str, destination: str) -> Dict[str, Any]:
        """Generate mock directions for testing when API is not available"""
        logger.info(f"Generating mock directions from {origin} to {destination}")
        
        # Create realistic mock data for airport navigation
        if "airport" in destination.lower():
            return {
                "routes": [{
                    "summary": "I-495 N to JFK Airport",
                    "legs": [{
                        "distance": {"text": "25.3 mi", "value": 40738},
                        "duration": {"text": "45 mins", "value": 2700},
                        "steps": [
                            {
                                "html_instructions": "Head north on Broadway toward W 42nd St",
                                "distance": {"text": "0.2 mi", "value": 322},
                                "duration": {"text": "2 mins", "value": 120},
                                "start_location": {"lat": 40.7589, "lng": -73.9851},
                                "end_location": {"lat": 40.7614, "lng": -73.9851}
                            },
                            {
                                "html_instructions": "Turn right onto FDR Dr N",
                                "distance": {"text": "8.5 mi", "value": 13679},
                                "duration": {"text": "12 mins", "value": 720},
                                "start_location": {"lat": 40.7614, "lng": -73.9851},
                                "end_location": {"lat": 40.8045, "lng": -73.9280}
                            },
                            {
                                "html_instructions": "Continue on I-495 N and take exit 22A toward Rental Car Return",
                                "distance": {"text": "0.5 mi", "value": 805},
                                "duration": {"text": "3 mins", "value": 180},
                                "start_location": {"lat": 40.6413, "lng": -73.7781},
                                "end_location": {"lat": 40.6428, "lng": -73.7732}
                            },
                            {
                                "html_instructions": "Turn right toward Terminal 4",
                                "distance": {"text": "0.3 mi", "value": 483},
                                "duration": {"text": "2 mins", "value": 120},
                                "start_location": {"lat": 40.6428, "lng": -73.7732},
                                "end_location": {"lat": 40.6413, "lng": -73.7781}
                            }
                        ]
                    }]
                }]
            }
        else:
            # Generic mock for other destinations
            return {
                "routes": [{
                    "summary": f"Route to {destination}",
                    "legs": [{
                        "distance": {"text": "15.2 mi", "value": 24462},
                        "duration": {"text": "28 mins", "value": 1680},
                        "steps": [
                            {
                                "html_instructions": "Head north on Main St",
                                "distance": {"text": "0.5 mi", "value": 805},
                                "duration": {"text": "2 mins", "value": 120},
                                "start_location": {"lat": 40.7589, "lng": -73.9851},
                                "end_location": {"lat": 40.7614, "lng": -73.9851}
                            },
                            {
                                "html_instructions": "Turn right onto Highway 101",
                                "distance": {"text": "12.0 mi", "value": 19312},
                                "duration": {"text": "18 mins", "value": 1080},
                                "start_location": {"lat": 40.7614, "lng": -73.9851},
                                "end_location": {"lat": 40.8045, "lng": -73.9280}
                            },
                            {
                                "html_instructions": "Continue straight for 2.7 mi",
                                "distance": {"text": "2.7 mi", "value": 4345},
                                "duration": {"text": "8 mins", "value": 480},
                                "start_location": {"lat": 40.8045, "lng": -73.9280},
                                "end_location": {"lat": 40.6413, "lng": -73.7781}
                            }
                        ]
                    }]
                }]
            }

    async def identify_confusing_steps(self, steps: List[Dict]) -> List[int]:
        """Identify potentially confusing steps based on location and step characteristics"""
        confusing_step_indices = []
        
        for i, step in enumerate(steps):
            start_location = step['start_location']
            end_location = step['end_location']
            instruction = step['html_instructions'].lower()
            
            # Check if step location matches known confusing areas
            location_key = f"{start_location['lat']:.4f},{start_location['lng']:.4f}"
            if location_key in self.confusing_locations_db:
                confusing_step_indices.append(i)
                continue
            
            # Detect potentially confusing patterns
            confusing_patterns = [
                "airport" in instruction,
                "rental" in instruction,
                "terminal" in instruction,
                "garage" in instruction,
                "exit" in instruction and ("ramp" in instruction or "merge" in instruction),
                "continue" in instruction and len(instruction.split()) < 5,  # Vague instructions
                "turn right" in instruction and i > 0 and "turn right" in steps[i-1]['html_instructions'].lower(),  # Multiple similar turns
                "roundabout" in instruction,
                "keep" in instruction and ("left" in instruction or "right" in instruction)
            ]
            
            if any(confusing_patterns):
                confusing_step_indices.append(i)
        
        return confusing_step_indices

    async def enhance_steps_with_ai(
        self, 
        steps: List[Dict], 
        confusing_steps: List[int], 
        user_context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Use AI to enhance navigation steps with natural language and context"""
        enhanced_steps = []
        
        for i, step in enumerate(steps):
            enhanced_step = {
                "step_number": i + 1,
                "original_instruction": step['html_instructions'],
                "distance": step['distance']['text'],
                "duration": step['duration']['text'],
                "is_confusing": i in confusing_steps
            }
            
            if i in confusing_steps:
                # Generate enhanced instruction with AI
                enhanced_instruction = await self.generate_enhanced_instruction(step, i, steps, user_context)
                enhanced_step["ai_enhanced_instruction"] = enhanced_instruction
                enhanced_step["clarification_tips"] = await self.generate_clarification_tips(step, user_context)
            else:
                # Convert to natural language even for non-confusing steps
                enhanced_step["natural_instruction"] = await self.convert_to_natural_language(step)
            
            enhanced_steps.append(enhanced_step)
        
        return enhanced_steps

    async def convert_to_natural_language(self, step: Dict) -> str:
        """Convert regular steps to natural language"""
        instruction = step['html_instructions'].lower()
        distance = step['distance']['text']
        
        # Simple natural language conversion
        if "turn right" in instruction:
            return f"In {distance}, turn right and continue ahead."
        elif "turn left" in instruction:
            return f"Coming up in {distance}, make a left turn."
        elif "continue" in instruction:
            return f"Keep going straight for {distance}."
        else:
            clean_instruction = instruction.replace('<', '').replace('>', '').replace('div', '').replace('/', '')
            return f"In {distance}: {clean_instruction}"

    async def generate_enhanced_instruction(
        self, 
        step: Dict, 
        step_index: int, 
        all_steps: List[Dict], 
        user_context: Dict[str, Any] = None
    ) -> str:
        """Generate AI-enhanced instruction for confusing steps"""
        
        # Build context for AI
        context = {
            "current_step": step,
            "step_number": step_index + 1,
            "total_steps": len(all_steps),
            "user_context": user_context or {}
        }
        
        # Add previous and next steps for context
        if step_index > 0:
            context["previous_step"] = all_steps[step_index - 1]
        if step_index < len(all_steps) - 1:
            context["next_step"] = all_steps[step_index + 1]
        
        # Check if this location has specific guidance in our database
        start_location = step['start_location']
        location_key = f"{start_location['lat']:.4f},{start_location['lng']:.4f}"
        
        if location_key in self.confusing_locations_db:
            location_guidance = self.confusing_locations_db[location_key]
            context["location_specific_guidance"] = location_guidance
        
        # Use AI to generate enhanced instruction
        ai_prompt = f"""
        You are an expert navigation assistant. Convert this robotic GPS instruction into a clear, 
        natural language direction that prevents common mistakes.
        
        Original instruction: {step['html_instructions']}
        Distance: {step['distance']['text']}
        
        Context: {json.dumps(context, indent=2)}
        
        Provide a clear, conversational instruction that:
        1. Uses natural language instead of robotic commands
        2. Includes helpful landmarks or visual cues
        3. Anticipates and prevents common mistakes
        4. Is encouraging and confidence-building
        
        Response should be concise but detailed enough to prevent confusion.
        """
        
        # Here you would call your AI service (Gemini wrapper)
        # For now, I'll provide a template-based enhancement
        return await self.ai_enhance_instruction(ai_prompt, step)

    async def ai_enhance_instruction(self, prompt: str, step: Dict) -> str:
        """Call AI service to enhance instruction"""
        try:
            # Use the existing Gemini wrapper to enhance instructions
            from app.modules.ai_wrapper.gemini_wrapper import get_gemini_wrapper
            
            wrapper = await get_gemini_wrapper()
            enhanced = await wrapper.chat("navigation_ai", prompt, {})
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing instruction with AI: {str(e)}")
            # Fallback to template-based enhancement
            return self.template_enhance_instruction(step)

    def template_enhance_instruction(self, step: Dict) -> str:
        """Fallback template-based instruction enhancement"""
        instruction = step['html_instructions'].lower()
        distance = step['distance']['text']
        
        # Template-based natural language conversion
        if "turn right" in instruction:
            return f"In {distance}, you'll make a right turn. Look for the turn after you see [landmark]."
        elif "turn left" in instruction:
            return f"Coming up in {distance}, turn left. Watch for [landmark] on your left to know you're at the right spot."
        elif "exit" in instruction:
            return f"Take the exit in {distance}. Stay in the right lane and follow the signs."
        elif "continue" in instruction:
            return f"Keep going straight for {distance}. You're on the right track!"
        else:
            return f"In {distance}: {instruction.replace('<', '').replace('>', '').replace('div', '').replace('/', '')}"

    async def generate_clarification_tips(self, step: Dict, user_context: Dict[str, Any] = None) -> List[str]:
        """Generate specific tips to avoid common mistakes at this location"""
        tips = []
        instruction = step['html_instructions'].lower()
        
        # Location-specific tips based on common issues
        if "airport" in instruction:
            tips.extend([
                "Follow signs for 'Departures' if dropping someone off, 'Arrivals' if picking up",
                "Airport roads can be confusing - stay in the right lane until you see your terminal",
                "Don't worry if you miss the first turn, there's usually another loop"
            ])
        
        elif "rental" in instruction:
            tips.extend([
                "Look for 'Rental Car Return' signs, not just the company name",
                "Follow the rental car shuttle if you see one - it's going the same way",
                "Gas stations are usually right before the return area"
            ])
        
        elif "parking" in instruction:
            tips.extend([
                "Take a photo of your parking spot number",
                "Note which level you're on and nearby landmarks",
                "Follow the 'P' signs, not the mall entrance signs"
            ])
        
        elif "exit" in instruction:
            tips.extend([
                "Get in the right lane early",
                "Look for the exit number, not just the destination name",
                "Don't panic if traffic is heavy - everyone's going the same way"
            ])
        
        return tips

    async def get_location_specific_insights(self, destination: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Use web search to get real-time insights about navigation issues at specific locations"""
        try:
            # Import the tool handler to use web search
            from app.modules.ai_wrapper.tool_handler import ToolHandler
            
            tool_handler = ToolHandler(self.unified_service)
            
            # Create targeted search queries for common navigation issues
            search_queries = []
            
            if any(keyword in destination.lower() for keyword in ["airport", "hertz", "rental", "car return"]):
                airport_name = self.extract_airport_name(destination)
                search_queries.extend([
                    f"{airport_name} Hertz rental car return directions problems",
                    f"{airport_name} airport rental car drop off confusing",
                    f"{airport_name} car rental return signs unclear"
                ])
            
            if "mall" in destination.lower():
                search_queries.append(f"{destination} parking navigation issues")
            
            if "hospital" in destination.lower():
                search_queries.append(f"{destination} directions parking entrance")
            
            # Execute searches and compile insights
            insights = {"search_results": [], "common_issues": [], "helpful_tips": []}
            
            for query in search_queries[:2]:  # Limit to 2 searches to control costs
                try:
                    search_result = await tool_handler._execute_web_search(query, num_results=3)
                    if search_result.get("success"):
                        insights["search_results"].append({
                            "query": query,
                            "summary": search_result.get("summary", ""),
                            "results": search_result.get("results", [])
                        })
                        
                        # Extract common issues from search results
                        await self.extract_common_issues_from_search(search_result, insights)
                        
                except Exception as e:
                    logger.warning(f"Web search failed for query '{query}': {str(e)}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting location insights: {str(e)}")
            return {"search_results": [], "common_issues": [], "helpful_tips": []}

    def extract_airport_name(self, destination: str) -> str:
        """Extract airport name from destination string"""
        airport_keywords = ["airport", "international", "regional"]
        words = destination.split()
        
        # Look for airport indicators and extract likely airport name
        for i, word in enumerate(words):
            if any(keyword in word.lower() for keyword in airport_keywords):
                # Take the word before airport keyword as the airport name
                if i > 0:
                    return words[i-1] + " " + word
                return word
        
        # Fallback - if we can't extract, return the full destination
        return destination

    async def extract_common_issues_from_search(self, search_result: Dict, insights: Dict):
        """Extract common navigation issues from web search results"""
        try:
            results_text = " ".join([
                result.get("snippet", "") + " " + result.get("title", "")
                for result in search_result.get("results", [])
            ]).lower()
            
            # Common issue patterns to look for
            issue_patterns = {
                "signage": ["signs unclear", "confusing signs", "poor signage", "signs missing"],
                "construction": ["construction", "closed", "detour", "temporary"],
                "multiple_levels": ["level", "floor", "garage", "deck"],
                "traffic": ["busy", "crowded", "traffic", "congested"],
                "layout": ["confusing layout", "hard to find", "difficult to locate"]
            }
            
            found_issues = []
            for issue_type, patterns in issue_patterns.items():
                if any(pattern in results_text for pattern in patterns):
                    found_issues.append(issue_type)
            
            insights["common_issues"].extend(found_issues)
            
            # Generate helpful tips based on found issues
            if "signage" in found_issues:
                insights["helpful_tips"].append("Signs may be unclear - look for rental car symbols and follow other cars")
            if "construction" in found_issues:
                insights["helpful_tips"].append("There may be construction - allow extra time and be flexible")
            if "multiple_levels" in found_issues:
                insights["helpful_tips"].append("Multiple levels/floors - note which level you're on")
            
        except Exception as e:
            logger.warning(f"Error extracting issues from search: {str(e)}")

    async def generate_proactive_guidance(self, route: Dict, user_context: Dict[str, Any] = None, next_destination: str = None) -> Dict[str, Any]:
        """Generate proactive warnings and helpful tips for the entire route"""
        guidance = {
            "route_overview": "",
            "preparation_tips": [],
            "potential_issues": [],
            "alternative_suggestions": []
        }
        
        # Analyze overall route characteristics
        total_duration = route['legs'][0]['duration']['value']  # in seconds
        total_distance = route['legs'][0]['distance']['value']  # in meters
        
        # Generate route overview
        guidance["route_overview"] = f"Your {route['legs'][0]['distance']['text']} journey will take about {route['legs'][0]['duration']['text']}. Here's what to expect..."
        
        # Preparation tips based on route analysis
        if total_duration > 3600:  # > 1 hour
            guidance["preparation_tips"].append("This is a longer drive - consider stopping for gas or coffee")
        
        if "airport" in route['summary'].lower():
            guidance["preparation_tips"].extend([
                "Airport traffic can be unpredictable - add 15-20 minutes buffer time",
                "Have your rental agreement and ID ready for car return"
            ])
        
        # Add guidance for multi-step journeys
        if next_destination:
            guidance["multi_step_journey"] = {
                "next_destination": next_destination,
                "transition_tip": f"After completing this trip, you'll be heading to {next_destination}",
                "time_buffer": "Consider the time between stops when planning"
            }
        
        return guidance

    async def report_confusing_location(
        self, 
        location: Dict[str, float], 
        user_feedback: str, 
        original_instruction: str,
        user_id: str
    ):
        """Allow users to report confusing locations for community improvement"""
        try:
            report = {
                "location": location,
                "user_feedback": user_feedback,
                "original_instruction": original_instruction,
                "user_id": user_id,
                "timestamp": datetime.utcnow(),
                "status": "pending_review"
            }
            
            await self.mongo_manager.db.location_feedback.insert_one(report)
            logger.info(f"Location feedback reported by user {user_id}")
            
            # Auto-add to confusing locations if multiple reports
            await self.check_and_update_confusing_locations(location)
            
        except Exception as e:
            logger.error(f"Error reporting confusing location: {str(e)}")

    async def check_and_update_confusing_locations(self, location: Dict[str, float]):
        """Check if location has multiple reports and add to confusing locations DB"""
        try:
            # Count reports for this location (within 100m radius)
            lat, lng = location['lat'], location['lng']
            reports_count = await self.mongo_manager.db.location_feedback.count_documents({
                "location.lat": {"$gte": lat - 0.001, "$lte": lat + 0.001},
                "location.lng": {"$gte": lng - 0.001, "$lte": lng + 0.001},
                "status": "pending_review"
            })
            
            # If 3+ reports, add to confusing locations
            if reports_count >= 3:
                location_key = f"{lat:.4f},{lng:.4f}"
                if location_key not in self.confusing_locations_db:
                    new_confusing_location = {
                        "lat": lat,
                        "lng": lng,
                        "added_date": datetime.utcnow(),
                        "report_count": reports_count,
                        "auto_added": True
                    }
                    
                    await self.mongo_manager.db.confusing_locations.insert_one(new_confusing_location)
                    self.confusing_locations_db[location_key] = new_confusing_location
                    
                    logger.info(f"Auto-added confusing location: {location_key}")
                    
        except Exception as e:
            logger.error(f"Error checking confusing locations: {str(e)}")

# Singleton instance
_smart_navigation_service = None

async def get_smart_navigation_service() -> SmartNavigationService:
    """Get singleton instance of SmartNavigationService"""
    global _smart_navigation_service
    if _smart_navigation_service is None:
        _smart_navigation_service = SmartNavigationService()
        await _smart_navigation_service.initialize()
    return _smart_navigation_service 