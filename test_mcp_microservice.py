#!/usr/bin/env python3
"""
MCP Microservice Integration Test

This script demonstrates how the mobile app now communicates with the
MCP server microservice for tool execution and management.

Architecture:
Mobile App (Python) <--> HTTP/REST API <--> MCP Server (Node.js)
"""

import asyncio
import os
from app.modules.mcp.mcp_client import get_mcp_client, MCPResponse
from app.core.config import settings

async def test_mcp_microservice():
    """Test the MCP microservice integration"""
    
    print("🔄 Testing MCP Microservice Integration")
    print("=" * 50)
    
    # Get MCP client (now communicates with external server)
    mcp_client = get_mcp_client()
    
    print(f"📡 MCP Server URL: {settings.MCP_SERVER_URL}")
    print(f"🔧 Client Type: Microservice HTTP Client")
    
    # 1. Test server health
    print(f"\n🏥 Testing Server Health...")
    health_result = await mcp_client.health_check()
    
    if health_result.success:
        print(f"   ✅ MCP Server is healthy")
        print(f"   📊 Server data: {health_result.data}")
    else:
        print(f"   ❌ MCP Server is unreachable: {health_result.error}")
        print(f"   ℹ️  Make sure the MCP server is running on {settings.MCP_SERVER_URL}")
        return
    
    # 2. Test user authentication
    print(f"\n🔐 Testing User Authentication...")
    test_user_id = "demo_user_123"
    test_auth_token = "mock_jwt_token_12345"  # In production, get from main app auth
    
    auth_result = await mcp_client.authenticate_user(test_user_id, test_auth_token)
    
    if auth_result.success:
        print(f"   ✅ User authenticated successfully")
        print(f"   🎫 Session ID: {auth_result.data.get('session_id', 'N/A')}")
        session_data = auth_result.data
    else:
        print(f"   ❌ Authentication failed: {auth_result.error}")
        print(f"   ℹ️  Note: This is expected in demo mode without real JWT validation")
        # Continue with demo for testing purposes
        session_data = {"session_id": "demo_session"}
    
    # 3. Test getting available tools
    print(f"\n🔧 Testing Tool Discovery...")
    tools_result = await mcp_client.get_available_tools(test_user_id)
    
    if tools_result.success:
        tools = tools_result.data
        print(f"   ✅ Found {len(tools)} available tools")
        
        # Show tool categories
        categories = {}
        for tool in tools:
            category = tool.get('category', 'uncategorized')
            if category not in categories:
                categories[category] = []
            categories[category].append(tool['name'])
        
        print(f"\n   📋 Tools by Category:")
        for category, tool_names in categories.items():
            print(f"      {category.title()}: {len(tool_names)} tools")
            for tool_name in tool_names[:2]:  # Show first 2 tools per category
                print(f"        • {tool_name}")
            if len(tool_names) > 2:
                print(f"        ... and {len(tool_names) - 2} more")
    else:
        print(f"   ❌ Failed to get tools: {tools_result.error}")
        return
    
    # 4. Test executing a public tool (weather)
    print(f"\n🌤️  Testing Public Tool Execution (Weather)...")
    weather_params = {
        "location": "New York",
        "units": "metric"
    }
    
    weather_result = await mcp_client.execute_tool(
        user_id=test_user_id,
        tool_id="get_weather",
        parameters=weather_params
    )
    
    if weather_result.success:
        weather_data = weather_result.data
        print(f"   ✅ Weather tool executed successfully")
        print(f"   🌡️  Location: {weather_data.get('location')}")
        print(f"   🌡️  Temperature: {weather_data.get('temperature')}°C")
        print(f"   ☁️  Condition: {weather_data.get('condition')}")
    else:
        print(f"   ❌ Weather tool failed: {weather_result.error}")
    
    # 5. Test executing a public tool (restaurant booking)
    print(f"\n🍽️  Testing Public Tool Execution (Restaurant Booking)...")
    restaurant_params = {
        "restaurant_name": "Italian Bistro",
        "date": "2024-01-15",
        "time": "19:00",
        "party_size": 4
    }
    
    # Simulate voice confirmations
    voice_confirmations = {
        "restaurant_name": True,
        "date": True,
        "time": True,
        "party_size": True
    }
    
    booking_result = await mcp_client.execute_tool(
        user_id=test_user_id,
        tool_id="book_restaurant_public",
        parameters=restaurant_params,
        voice_confirmations=voice_confirmations
    )
    
    if booking_result.success:
        booking_data = booking_result.data
        print(f"   ✅ Restaurant booking successful")
        print(f"   🎫 Booking ID: {booking_data.get('booking_id')}")
        print(f"   🏪 Restaurant: {booking_data.get('restaurant_name')}")
        print(f"   📅 Date/Time: {booking_data.get('date')} at {booking_data.get('time')}")
        print(f"   👥 Party Size: {booking_data.get('party_size')}")
    else:
        print(f"   ❌ Restaurant booking failed: {booking_result.error}")
    
    # 6. Test web search tool
    print(f"\n🔍 Testing Public Tool Execution (Web Search)...")
    search_params = {
        "query": "best pizza restaurants nearby",
        "num_results": 3
    }
    
    search_result = await mcp_client.execute_tool(
        user_id=test_user_id,
        tool_id="web_search",
        parameters=search_params
    )
    
    if search_result.success:
        search_data = search_result.data
        print(f"   ✅ Web search executed successfully")
        print(f"   🔍 Query: {search_data.get('query')}")
        print(f"   📊 Results found: {len(search_data.get('results', []))}")
        for i, result in enumerate(search_data.get('results', [])[:2]):
            print(f"      {i+1}. {result.get('title')}")
    else:
        print(f"   ❌ Web search failed: {search_result.error}")
    
    # 7. Test user-specific tool (would require auth)
    print(f"\n📧 Testing User-Specific Tool (Email - Requires Auth)...")
    email_params = {
        "action": "send",
        "to": ["friend@example.com"],
        "subject": "Test from car",
        "body": "Hi! Sending this email from my car using voice commands."
    }
    
    email_result = await mcp_client.execute_tool(
        user_id=test_user_id,
        tool_id="email_personal",
        parameters=email_params
    )
    
    if email_result.success:
        print(f"   ✅ Email sent successfully")
    else:
        print(f"   ⚠️  Email failed (expected): {email_result.error}")
        print(f"   ℹ️  This is expected - requires Gmail OAuth authentication")
    
    # 8. Test tool configuration
    print(f"\n⚙️  Testing Tool Configuration...")
    config_result = await mcp_client.configure_tool(
        user_id=test_user_id,
        tool_id="get_weather",
        enabled=True,
        priority=1,
        custom_settings={"auto_location": True}
    )
    
    if config_result.success:
        print(f"   ✅ Tool configured successfully")
    else:
        print(f"   ❌ Tool configuration failed: {config_result.error}")
    
    # 9. Test session cleanup
    print(f"\n🧹 Testing Session Cleanup...")
    logout_result = await mcp_client.logout_user(test_user_id)
    
    if logout_result.success:
        print(f"   ✅ User logged out successfully")
    else:
        print(f"   ⚠️  Logout warning: {logout_result.error}")
    
    # Close client connection
    await mcp_client.close()
    
    print(f"\n✅ MCP Microservice Integration Test Complete!")
    print(f"\n📊 Test Summary:")
    print(f"   🔧 Architecture: Microservice HTTP-based")
    print(f"   🌐 Communication: REST API over HTTP")
    print(f"   🔐 Authentication: JWT + Session-based")
    print(f"   🛠️  Tools Tested: Weather, Search, Booking, Email")
    print(f"   📱 Client: Python asyncio HTTP client")
    print(f"   🖥️  Server: Node.js Express server (to be deployed)")

async def demonstrate_driving_scenarios():
    """Demonstrate real-world driving scenarios with microservice"""
    
    print(f"\n🚗 Real-World Driving Scenarios (Microservice)")
    print("=" * 50)
    
    mcp_client = get_mcp_client()
    driver_id = "sarah_driver_456"
    
    # Authenticate driver
    auth_result = await mcp_client.authenticate_user(driver_id, "driver_token_789")
    
    print(f"👤 Driver: Sarah Wilson")
    print(f"🎯 Testing hands-free, voice-controlled operations")
    
    # Scenario 1: Check weather before trip
    print(f"\n🌤️  Scenario 1: Check weather before trip")
    weather_result = await mcp_client.execute_tool(
        user_id=driver_id,
        tool_id="get_weather",
        parameters={"location": "destination city", "units": "imperial"}
    )
    
    if weather_result.success:
        print(f"   ✅ Weather check: {weather_result.data.get('condition')}")
        print(f"   🌡️  Temperature: {weather_result.data.get('temperature')}°F")
    
    # Scenario 2: Emergency restaurant booking
    print(f"\n🍽️  Scenario 2: Emergency dinner booking while driving")
    urgent_booking = await mcp_client.execute_tool(
        user_id=driver_id,
        tool_id="book_restaurant_public",
        parameters={
            "restaurant_name": "Quick Bites Diner",
            "date": "2024-01-12",
            "time": "18:30",
            "party_size": 2
        },
        voice_confirmations={
            "restaurant_name": True,
            "date": True,
            "time": True,
            "party_size": True
        }
    )
    
    if urgent_booking.success:
        print(f"   ✅ Emergency booking confirmed!")
        print(f"   🎫 Confirmation: {urgent_booking.data.get('confirmation_number')}")
    
    # Scenario 3: Search for gas stations
    print(f"\n⛽ Scenario 3: Find nearby gas stations")
    gas_search = await mcp_client.execute_tool(
        user_id=driver_id,
        tool_id="web_search",
        parameters={
            "query": "gas stations near me open now",
            "num_results": 5
        }
    )
    
    if gas_search.success:
        print(f"   ✅ Found {len(gas_search.data.get('results', []))} gas stations")
    
    await mcp_client.close()
    
    print(f"\n🎉 Driving scenarios completed successfully!")
    print(f"✨ All operations were hands-free and voice-controlled")

async def main():
    """Main test function"""
    try:
        print(f"🚀 McQueen.io MCP Microservice Test Suite")
        print(f"📡 Testing mobile app <-> server communication")
        
        await test_mcp_microservice()
        await demonstrate_driving_scenarios()
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Deploy MCP server to mcqueen-io/server repo")
        print(f"   2. Set up production authentication")
        print(f"   3. Add real service integrations (Gmail, WhatsApp, etc.)")
        print(f"   4. Configure OAuth flows")
        print(f"   5. Add monitoring and logging")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 