# MCP Server Microservice Setup

## Overview
This guide shows how to set up the MCP server as a separate microservice in the mcqueen-io/server repository.

## Architecture Changes Made

### Mobile App (Current Repo)
✅ **Updated MCP Client** - Now communicates with external server via HTTP
✅ **Updated API Endpoints** - Proxy requests to MCP server
✅ **Session Management** - Handles user authentication with server

### Server Repo (To Be Created)
🔲 **Complete MCP Implementation** - All tool logic moved to server
🔲 **Three-Tier Access Model** - Public, User-Specific, Hybrid tools
🔲 **Enhanced Security** - User isolation, encrypted credentials
🔲 **Service Adapters** - Gmail, WhatsApp, LinkedIn integrations

## Quick Setup for Server Repo

```bash
# 1. Clone server repository
git clone https://github.com/mcqueen-io/server.git
cd server

# 2. Initialize Node.js project
npm init -y

# 3. Install dependencies
npm install express cors helmet express-rate-limit
npm install jsonwebtoken mongoose redis winston

# 4. Create directory structure
mkdir -p src/{routes,controllers,services,middleware,utils}

# 5. Copy environment template
echo "NODE_ENV=development
PORT=3001
JWT_SECRET=your-secret-key
ALLOWED_ORIGINS=http://localhost:3000" > .env
```

## Key Files to Create in Server

### 1. src/app.js - Main server
### 2. src/routes/mcp.js - MCP endpoints  
### 3. src/services/toolRegistry.js - Tool definitions
### 4. src/controllers/tools.js - Tool execution
### 5. src/middleware/auth.js - Authentication

## Updated Mobile App Configuration

The mobile app now needs to point to the MCP server:

```python
# app/core/config.py
MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://localhost:3001")
```

## Testing the Integration

```bash
# 1. Start MCP server
cd server && npm start

# 2. Test from mobile app
cd mobile-app
python test_mcp_integration.py
```

## Benefits of This Architecture

✅ **Scalability** - Independent scaling of MCP services
✅ **Security** - Isolated credential management  
✅ **Flexibility** - Different tech stacks possible
✅ **Maintainability** - Separate deployment cycles

The mobile app is now ready to work with the MCP server microservice! 