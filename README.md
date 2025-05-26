# In-Car AI Assistant

An intelligent in-car assistant that provides personalized experiences through voice interaction, safety monitoring, and smart features.

## Features

- Voice-based interactive AI wrapper
- User data management with family tree support
- Smart conversation and activity recommendations
- Event tracking and follow-ups
- Safety monitoring and sleep detection
- Integration with various services (email, maps, etc.)
- Natural language navigation assistance

## Project Structure

```
backend/
├── app/
│   ├── api/            # API endpoints
│   ├── core/           # Core business logic
│   ├── db/             # Database models and connections
│   ├── services/       # External service integrations
│   └── utils/          # Utility functions
├── tests/              # Test files
└── config/             # Configuration files
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the development server:
```bash
uvicorn app.main:app --reload
```

## Development

- Backend: Python with FastAPI
- Database: Neo4j (Graph DB), ChromaDB (Vector DB)
- Voice Processing: Deepgram
- AI: OpenAI GPT

## License

Proprietary - All rights reserved 