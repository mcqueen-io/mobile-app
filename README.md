# In-Car AI Assistant

A sophisticated AI-powered mobile application designed for in-car use, featuring advanced voice recognition, intelligent memory management, and seamless integration with external services through a microservice architecture.

## 🚗 Features

### Core Capabilities
- **Advanced Voice Recognition** with noise suppression and speaker identification
- **Intelligent Memory Management** with ChromaDB and MongoDB integration
- **Context-Aware Conversations** with intelligent context management
- **MCP (Model Control Protocol) Integration** for external service access
- **Real-time Navigation** with maps integration
- **User Profile Management** with preference learning
- **AI-Powered Reflection** and conversation analysis

### Microservice Architecture
- **MCP Server Integration** for external service access (Gmail, WhatsApp, Calendar, etc.)
- **Three-tier Tool Access Model** (Public, User-specific, Hybrid)
- **Secure Authentication** with JWT tokens and session management
- **Scalable Design** supporting horizontal scaling

## 🏗️ Project Architecture

```
mobile-app/
├── app/
│   ├── api/                          # API endpoints
│   │   ├── navigation.py             # Navigation API
│   │   └── reflection.py             # AI reflection API
│   ├── config/                       # Configuration management
│   │   └── config.template.py        # Configuration template
│   ├── core/                         # Core services
│   │   ├── ai_security.py            # AI security management
│   │   ├── cache_manager.py          # Cache management
│   │   ├── config.py                 # Application configuration
│   │   └── security.py               # Security middleware
│   ├── db/                           # Database management
│   │   ├── chroma_manager.py         # ChromaDB vector database
│   │   ├── mongo_manager.py          # MongoDB integration
│   │   └── schema_validator.py       # Data validation
│   ├── models/                       # Data models
│   │   ├── data_transformer.py       # Data transformation utilities
│   │   └── user.py                   # User model with MCP profile support
│   ├── modules/                      # Feature modules
│   │   ├── ai_wrapper/               # AI service wrappers
│   │   │   ├── gemini_wrapper.py     # Google Gemini integration
│   │   │   ├── reflection_manager.py # AI reflection management
│   │   │   ├── tool_handler.py       # Tool execution handler
│   │   │   └── user_specific_gemini_wrapper.py
│   │   ├── context/                  # Context management
│   │   │   ├── api.py                # Context API endpoints
│   │   │   ├── context_manager.py    # Context management logic
│   │   │   ├── conversation_buffer_manager.py
│   │   │   └── intelligent_context_manager.py
│   │   ├── maps/                     # Maps integration
│   │   │   ├── api.py                # Maps API endpoints
│   │   │   └── maps_optimizer.py     # Maps optimization
│   │   ├── mcp/                      # MCP client integration
│   │   │   ├── api.py                # MCP API endpoints
│   │   │   ├── mcp_client.py         # MCP server client
│   │   │   └── services/             # MCP service adapters
│   │   ├── memory/                   # Memory management
│   │   │   ├── api.py                # Memory API endpoints
│   │   │   └── memory_store.py       # Memory storage
│   │   ├── user_info/                # User management
│   │   │   ├── api.py                # User API endpoints
│   │   │   └── user_graph.py         # User relationship graph
│   │   └── voice_layer/              # Voice processing
│   │       ├── api.py                # Voice API endpoints
│   │       ├── conversation_manager.py
│   │       ├── deepgram.py           # Deepgram integration
│   │       ├── google_transcriber.py # Google Speech-to-Text
│   │       ├── transcriber_factory.py
│   │       └── voice_processor.py    # Voice processing core
│   ├── services/                     # Business logic services
│   │   ├── ai_wrapper.py             # AI service wrapper
│   │   ├── memory_intelligence_service.py
│   │   ├── offline_navigation_cache.py
│   │   ├── smart_navigation_service.py
│   │   └── unified_service.py        # Unified service interface
│   └── main.py                       # FastAPI application entry point
├── data/                             # Data storage
├── docs/                             # Documentation
├── tests/                            # Test suites
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
└── README.md                         # This file
```

## 🚀 Setup

### Prerequisites
- Python 3.8+
- MongoDB (for user data and sessions)
- ChromaDB (for vector memory storage)
- MCP Server (external microservice for tool access)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd mobile-app
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configuration:**
   - Copy `app/config/config.template.py` to `app/config/config.py`
   - Update the configuration values in `config.py` with your settings
   - Configure MCP server URL and authentication
   - Set up database connections (MongoDB, ChromaDB)
   - Configure AI service API keys (Google Gemini, Deepgram)
   - **Never commit `config.py` to version control**

5. **Start the application:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 🔧 MCP Server Integration

The application integrates with an external MCP (Model Control Protocol) server for accessing external services:

### Available Tools
- **Email Management** (Gmail integration)
- **Calendar Operations** (Google Calendar)
- **Messaging** (WhatsApp, Slack)
- **Navigation** (Google Maps)
- **Weather Information**
- **Restaurant Booking**
- **Financial Services**

### Authentication Flow
1. User authenticates with the mobile app
2. App creates session with MCP server
3. User can access tools based on their authentication level
4. Session management handles token refresh and cleanup

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

### Key Test Files
- `test_mcp_integration.py` - MCP server integration tests
- `test_voice_memory_integration.py` - Voice and memory integration
- `test_intelligent_memory.py` - Memory management tests
- `test_smart_navigation.py` - Navigation functionality tests

## 🔒 Security

- **API Key Management**: Secure storage of sensitive credentials
- **JWT Authentication**: Token-based authentication with MCP server
- **Rate Limiting**: Protection against API abuse
- **CORS Protection**: Configurable allowed origins
- **Request Validation**: Comprehensive input validation
- **Session Management**: Secure session handling with automatic cleanup

## 📊 Performance

- **Connection Pooling**: Optimized database and external API connections
- **Caching**: Intelligent caching for frequently accessed data
- **Async Processing**: Non-blocking operations for better responsiveness
- **Memory Optimization**: Efficient memory usage for mobile environments

## 🚀 Deployment

### Production Considerations
- Use environment variables for all sensitive configuration
- Set up proper logging and monitoring
- Configure health checks for all services
- Implement proper backup strategies for databases
- Use load balancing for high availability

### Docker Support
```bash
docker build -t in-car-ai-assistant .
docker run -p 8000:8000 in-car-ai-assistant
```

## 📚 Documentation

- `MCP_SERVER_IMPLEMENTATION.md` - Complete MCP server implementation guide
- `MICROSERVICE_MIGRATION_COMPLETE.md` - Migration details and benefits
- `MCP_SERVER_SETUP.md` - Server setup instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

[Your License Here]

## 🆘 Support

For issues and questions:
- Check the documentation in the `docs/` directory
- Review the test files for usage examples
- Open an issue on the repository 