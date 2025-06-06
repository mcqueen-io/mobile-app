"""
Offline Navigation Cache Service
Intelligent caching system for offline navigation assistance when connection is lost.
Pre-caches prominent places, hotels, gas stations, and emergency routes.
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from app.db.mongo_manager import get_mongo_manager
from app.core.config import settings
import aiohttp
from geopy.distance import geodesic
import math

logger = logging.getLogger(__name__)

@dataclass
class CachedLocation:
    """Represents a cached location with metadata"""
    name: str
    address: str
    lat: float
    lng: float
    category: str  # 'hotel', 'gas_station', 'downtown', 'hospital', 'police'
    cached_at: datetime
    confidence_score: float
    additional_info: Dict[str, Any]

@dataclass
class OfflineRoute:
    """Simplified route for offline use"""
    origin: Dict[str, float]
    destination: Dict[str, float]
    simplified_steps: List[str]
    distance_miles: float
    estimated_time_minutes: int
    cached_at: datetime
    route_type: str  # 'emergency', 'hotel', 'gas_station', 'downtown'

class OfflineNavigationCache:
    """Intelligent caching system for offline navigation"""
    
    def __init__(self):
        self.mongo_manager = None
        self.maps_api_key = settings.GOOGLE_SEARCH_API_KEY or settings.GOOGLE_MAPS_API_KEY
        
        # Cache configuration
        self.cache_radius_miles = 50  # Cache locations within 50 miles
        self.max_cached_locations = 200  # Limit cache size
        self.cache_expiry_hours = 24  # Refresh cache daily
        
        # Location categories to cache
        self.essential_categories = [
            'gas_station', 'hospital', 'police_station', 'hotel', 
            'downtown', 'pharmacy', 'grocery_store', 'auto_repair'
        ]

    async def initialize(self):
        """Initialize the caching system"""
        self.mongo_manager = await get_mongo_manager()
        logger.info("Offline Navigation Cache initialized")

    async def preload_trip_cache(self, trip_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preload cache for an upcoming trip with essential locations
        
        Args:
            trip_data: {
                'origin': {'lat': float, 'lng': float, 'address': str},
                'destination': {'lat': float, 'lng': float, 'address': str},
                'hotel_address': str (optional),
                'waypoints': List[Dict] (optional),
                'trip_duration_days': int (optional)
            }
        """
        try:
            logger.info(f"Preloading trip cache for route: {trip_data['origin']['address']} → {trip_data['destination']['address']}")
            
            # 1. Cache prominent places along the route
            route_locations = await self._cache_route_locations(trip_data['origin'], trip_data['destination'])
            
            # 2. Cache hotel and nearby amenities if provided
            hotel_locations = []
            if trip_data.get('hotel_address'):
                hotel_locations = await self._cache_hotel_vicinity(trip_data['hotel_address'])
            
            # 3. Cache waypoint amenities
            waypoint_locations = []
            if trip_data.get('waypoints'):
                for waypoint in trip_data['waypoints']:
                    waypoint_cache = await self._cache_location_vicinity(waypoint)
                    waypoint_locations.extend(waypoint_cache)
            
            # 4. Create emergency offline routes
            emergency_routes = await self._create_emergency_routes(trip_data)
            
            # 5. Store everything in cache
            cache_summary = await self._store_trip_cache({
                'trip_id': f"trip_{datetime.now().isoformat()}",
                'route_locations': route_locations,
                'hotel_locations': hotel_locations,
                'waypoint_locations': waypoint_locations,
                'emergency_routes': emergency_routes,
                'cached_at': datetime.now(),
                'trip_data': trip_data
            })
            
            logger.info(f"Trip cache preloaded: {len(route_locations + hotel_locations + waypoint_locations)} locations, {len(emergency_routes)} emergency routes")
            
            return cache_summary
            
        except Exception as e:
            logger.error(f"Error preloading trip cache: {str(e)}")
            return {"error": str(e)}

    async def get_offline_guidance(self, current_location: Dict[str, float], user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Provide offline navigation guidance when connection is lost
        
        Args:
            current_location: {'lat': float, 'lng': float}
            user_context: {'need_type': str, 'hotel_address': str, etc.}
        """
        try:
            need_type = user_context.get('need_type', 'general') if user_context else 'general'
            
            # Get nearby cached locations
            nearby_locations = await self._get_nearby_cached_locations(
                current_location, 
                max_distance_miles=20
            )
            
            # Prioritize locations based on need
            prioritized_locations = self._prioritize_by_need(nearby_locations, need_type)
            
            # Get cached routes to key locations
            cached_routes = await self._get_cached_routes_from_location(current_location)
            
            # Generate simple directions to top 3 locations
            simplified_directions = []
            for location in prioritized_locations[:3]:
                direction = self._generate_simple_direction(current_location, location)
                simplified_directions.append(direction)
            
            return {
                "offline_mode": True,
                "current_location": current_location,
                "nearby_essential_locations": [asdict(loc) for loc in prioritized_locations[:5]],
                "simplified_directions": simplified_directions,
                "cached_routes": cached_routes,
                "guidance_message": self._generate_offline_guidance_message(need_type, prioritized_locations),
                "last_cache_update": await self._get_last_cache_update()
            }
            
        except Exception as e:
            logger.error(f"Error getting offline guidance: {str(e)}")
            return {"error": str(e), "offline_mode": True}

    async def _cache_route_locations(self, origin: Dict, destination: Dict) -> List[CachedLocation]:
        """Cache essential locations along a route"""
        cached_locations = []
        
        try:
            # Get route from Google Maps
            route = await self._get_simple_route(origin, destination)
            if not route:
                return []
            
            # Extract waypoints along the route (every 25 miles or so)
            route_waypoints = self._extract_route_waypoints(route, interval_miles=25)
            
            # For each waypoint, find and cache essential locations
            for waypoint in route_waypoints:
                for category in self.essential_categories:
                    locations = await self._find_and_cache_category(waypoint, category)
                    cached_locations.extend(locations)
            
            return cached_locations
            
        except Exception as e:
            logger.error(f"Error caching route locations: {str(e)}")
            return []

    async def _cache_hotel_vicinity(self, hotel_address: str) -> List[CachedLocation]:
        """Cache locations near the hotel"""
        try:
            # Geocode hotel address
            hotel_coords = await self._geocode_address(hotel_address)
            if not hotel_coords:
                return []
            
            # Cache hotel itself
            hotel_location = CachedLocation(
                name="Your Hotel",
                address=hotel_address,
                lat=hotel_coords['lat'],
                lng=hotel_coords['lng'],
                category='hotel',
                cached_at=datetime.now(),
                confidence_score=1.0,
                additional_info={'is_user_hotel': True}
            )
            
            # Cache nearby amenities
            vicinity_locations = [hotel_location]
            for category in ['gas_station', 'pharmacy', 'grocery_store', 'hospital']:
                locations = await self._find_and_cache_category(hotel_coords, category, max_results=2)
                vicinity_locations.extend(locations)
            
            return vicinity_locations
            
        except Exception as e:
            logger.error(f"Error caching hotel vicinity: {str(e)}")
            return []

    async def _find_and_cache_category(self, location: Dict[str, float], category: str, max_results: int = 3) -> List[CachedLocation]:
        """Find and cache locations of a specific category near a point"""
        try:
            if not self.maps_api_key:
                return []
            
            # Map categories to Google Places types
            place_types = {
                'gas_station': 'gas_station',
                'hospital': 'hospital',
                'police_station': 'police',
                'hotel': 'lodging',
                'pharmacy': 'pharmacy',
                'grocery_store': 'grocery_or_supermarket',
                'auto_repair': 'car_repair',
                'downtown': 'downtown'  # Special handling needed
            }
            
            place_type = place_types.get(category, category)
            
            # Special handling for downtown
            if category == 'downtown':
                return await self._find_downtown_area(location)
            
            # Use Google Places API
            async with aiohttp.ClientSession() as session:
                url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
                params = {
                    "location": f"{location['lat']},{location['lng']}",
                    "radius": 8000,  # 5 miles
                    "type": place_type,
                    "key": self.maps_api_key
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        cached_locations = []
                        for place in data.get('results', [])[:max_results]:
                            cached_location = CachedLocation(
                                name=place['name'],
                                address=place.get('vicinity', ''),
                                lat=place['geometry']['location']['lat'],
                                lng=place['geometry']['location']['lng'],
                                category=category,
                                cached_at=datetime.now(),
                                confidence_score=place.get('rating', 3.0) / 5.0,
                                additional_info={
                                    'place_id': place['place_id'],
                                    'rating': place.get('rating'),
                                    'open_now': place.get('opening_hours', {}).get('open_now')
                                }
                            )
                            cached_locations.append(cached_location)
                        
                        return cached_locations
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding {category} locations: {str(e)}")
            return []

    def _generate_simple_direction(self, from_location: Dict[str, float], to_location: CachedLocation) -> Dict[str, Any]:
        """Generate simple compass-based direction"""
        # Calculate bearing and distance
        distance_miles = geodesic(
            (from_location['lat'], from_location['lng']),
            (to_location.lat, to_location.lng)
        ).miles
        
        bearing = self._calculate_bearing(from_location, {'lat': to_location.lat, 'lng': to_location.lng})
        direction = self._bearing_to_direction(bearing)
        
        return {
            "destination": to_location.name,
            "category": to_location.category,
            "distance_miles": round(distance_miles, 1),
            "simple_direction": f"Head {direction} for {round(distance_miles, 1)} miles",
            "compass_bearing": bearing,
            "detailed_instruction": f"From your location, drive {direction} toward {to_location.name}. It's approximately {round(distance_miles, 1)} miles away."
        }

    def _calculate_bearing(self, point1: Dict[str, float], point2: Dict[str, float]) -> float:
        """Calculate bearing between two points"""
        lat1, lng1 = math.radians(point1['lat']), math.radians(point1['lng'])
        lat2, lng2 = math.radians(point2['lat']), math.radians(point2['lng'])
        
        d_lng = lng2 - lng1
        
        y = math.sin(d_lng) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lng)
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing

    def _bearing_to_direction(self, bearing: float) -> str:
        """Convert bearing to simple direction"""
        directions = [
            "North", "Northeast", "East", "Southeast", 
            "South", "Southwest", "West", "Northwest"
        ]
        index = round(bearing / 45) % 8
        return directions[index]

    def _calculate_distance_miles(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate approximate distance in miles using Haversine formula"""
        # Simplified distance calculation for offline use
        R = 3959  # Earth's radius in miles
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlng/2) * math.sin(dlng/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

    def _prioritize_by_need(self, locations: List[CachedLocation], need_type: str) -> List[CachedLocation]:
        """Prioritize locations based on user's current need"""
        priority_map = {
            'emergency': ['hospital', 'police_station', 'gas_station'],
            'fuel': ['gas_station', 'auto_repair'],
            'accommodation': ['hotel', 'downtown'],
            'supplies': ['pharmacy', 'grocery_store', 'gas_station'],
            'general': ['gas_station', 'downtown', 'hospital', 'hotel']
        }
        
        priorities = priority_map.get(need_type, priority_map['general'])
        
        # Sort locations by priority and distance
        def sort_key(location: CachedLocation):
            priority_score = priorities.index(location.category) if location.category in priorities else len(priorities)
            return (priority_score, -location.confidence_score)
        
        return sorted(locations, key=sort_key)

    def _generate_offline_guidance_message(self, need_type: str, locations: List[CachedLocation]) -> str:
        """Generate helpful guidance message for offline mode"""
        if not locations:
            return "No cached locations available. Try to get to higher ground or a main road to regain signal."
        
        top_location = locations[0]
        
        messages = {
            'emergency': f"Connection lost. Nearest help: {top_location.name} ({top_location.category}). Head {self._get_simple_direction_to(top_location)} if needed.",
            'fuel': f"Connection lost. Nearest gas station: {top_location.name}. Follow offline directions to refuel.",
            'accommodation': f"Connection lost. Your hotel or nearest lodging: {top_location.name}. Use cached directions to get there safely.",
            'general': f"Connection lost. Nearest services: {top_location.name} ({top_location.category}). Use offline cache for basic navigation."
        }
        
        return messages.get(need_type, messages['general'])

    async def _store_trip_cache(self, cache_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store trip cache in database"""
        try:
            await self.mongo_manager.db.offline_navigation_cache.insert_one(cache_data)
            
            return {
                "trip_id": cache_data['trip_id'],
                "cached_locations": len(cache_data['route_locations'] + cache_data['hotel_locations'] + cache_data['waypoint_locations']),
                "emergency_routes": len(cache_data['emergency_routes']),
                "cached_at": cache_data['cached_at'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error storing trip cache: {str(e)}")
            return {"error": str(e)}

    def _get_simple_direction_to(self, location: CachedLocation) -> str:
        """Get simple direction description"""
        return f"toward {location.name}"

    # Simplified implementations for now - can be expanded later
    async def _get_simple_route(self, origin: Dict, destination: Dict) -> Optional[Dict]:
        return None
    
    def _extract_route_waypoints(self, route: Dict, interval_miles: int = 25) -> List[Dict[str, float]]:
        return []
    
    async def _geocode_address(self, address: str) -> Optional[Dict[str, float]]:
        return None
    
    async def _get_nearby_cached_locations(self, location: Dict[str, float], max_distance_miles: float = 20) -> List[CachedLocation]:
        return []
    
    async def _get_cached_routes_from_location(self, location: Dict[str, float]) -> List[OfflineRoute]:
        return []
    
    async def _create_emergency_routes(self, trip_data: Dict[str, Any]) -> List[OfflineRoute]:
        return []
    
    async def _get_last_cache_update(self) -> str:
        return datetime.now().isoformat()
    
    async def _cache_location_vicinity(self, location: Dict) -> List[CachedLocation]:
        return []
    
    async def _find_downtown_area(self, location: Dict[str, float]) -> List[CachedLocation]:
        return []

# Singleton instance
_offline_cache: Optional[OfflineNavigationCache] = None

async def get_offline_navigation_cache() -> OfflineNavigationCache:
    """Get singleton instance of OfflineNavigationCache"""
    global _offline_cache
    if _offline_cache is None:
        _offline_cache = OfflineNavigationCache()
        await _offline_cache.initialize()
    return _offline_cache 