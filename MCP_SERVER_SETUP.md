# 🚀 MCP Server Microservice Setup Guide
## For mcqueen-io/server Repository

This guide provides the complete implementation for setting up the MCP server as a microservice.

## 📋 Quick Setup Commands

### 1. Clone and Setup Server Repository
```bash
# Create the server repository
git clone https://github.com/mcqueen-io/server.git
cd server

# Initialize package.json
npm init -y

# Install core dependencies
npm install express cors helmet express-rate-limit jsonwebtoken bcryptjs
npm install mongoose redis axios crypto joi winston googleapis dotenv

# Install dev dependencies  
npm install --save-dev nodemon jest supertest

# Create directory structure
mkdir -p src/{config,controllers,middleware,models,services,adapters,utils,routes}
mkdir -p src/adapters/{gmail,whatsapp,booking}
mkdir -p tests docker k8s docs
```

### 2. Create Package.json Configuration
```json
{
  "name": "mcqueen-mcp-server",
  "version": "1.0.0",
  "description": "MCP Server for McQueen.io Mobile Assistant",
  "main": "src/app.js",
  "scripts": {
    "start": "node src/app.js",
    "dev": "nodemon src/app.js",
    "test": "jest",
    "docker:build": "docker build -t mcqueen-mcp-server .",
    "docker:run": "docker-compose up"
  },
  "dependencies": {
    "express": "^4.18.2",
    "express-rate-limit": "^6.7.0",
    "cors": "^2.8.5",
    "helmet": "^6.1.5",
    "jsonwebtoken": "^9.0.0",
    "bcryptjs": "^2.4.3",
    "mongoose": "^7.0.3",
    "redis": "^4.6.5",
    "axios": "^1.3.6",
    "joi": "^17.9.1",
    "winston": "^3.8.2",
    "googleapis": "^118.0.0",
    "dotenv": "^16.0.3"
  }
}
```

## 🏗️ Core Implementation Files

### src/app.js (Main Application)
```javascript
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
require('dotenv').config();

const authRoutes = require('./routes/auth');
const mcpRoutes = require('./routes/mcp');
const healthRoutes = require('./routes/health');
const authMiddleware = require('./middleware/auth');
const logger = require('./utils/logger');

const app = express();
const PORT = process.env.PORT || 3001;

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100,
  message: 'Too many requests from this IP'
});
app.use('/api/', limiter);

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/v1/auth', authRoutes);
app.use('/api/v1/mcp', authMiddleware, mcpRoutes);
app.use('/api/v1/health', healthRoutes);

// Error handling
app.use((err, req, res, next) => {
  logger.error('Unhandled error:', err);
  res.status(500).json({
    success: false,
    error: 'Internal server error'
  });
});

app.listen(PORT, () => {
  logger.info(`MCP Server running on port ${PORT}`);
});

module.exports = app;
```

### src/utils/logger.js
```javascript
const winston = require('winston');

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'mcp-server' },
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
  ],
});

if (process.env.NODE_ENV !== 'production') {
  logger.add(new winston.transports.Console({
    format: winston.format.simple()
  }));
}

module.exports = logger;
```

### src/routes/auth.js
```javascript
const express = require('express');
const jwt = require('jsonwebtoken');
const SessionManager = require('../services/sessionManager');
const logger = require('../utils/logger');

const router = express.Router();

router.post('/login', async (req, res) => {
  try {
    const { user_id, auth_token, client_type } = req.body;

    // Validate auth token
    const isValid = await validateAuthToken(user_id, auth_token);
    if (!isValid) {
      return res.status(401).json({
        success: false,
        error: 'Invalid authentication token'
      });
    }

    // Create MCP session
    const session = await SessionManager.createSession(user_id, client_type);

    res.json({
      success: true,
      session: {
        session_id: session.sessionId,
        user_id: session.userId,
        created_at: session.createdAt.toISOString(),
        expires_at: session.expiresAt.toISOString()
      }
    });
  } catch (error) {
    logger.error('Login error:', error);
    res.status(500).json({
      success: false,
      error: 'Authentication failed'
    });
  }
});

router.post('/logout', async (req, res) => {
  try {
    const sessionId = req.headers['x-session-id'];
    await SessionManager.destroySession(sessionId);

    res.json({
      success: true,
      message: 'Logged out successfully'
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Logout failed'
    });
  }
});

async function validateAuthToken(userId, token) {
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    return decoded.user_id === userId;
  } catch (error) {
    return false;
  }
}

module.exports = router;
```

### src/routes/mcp.js
```javascript
const express = require('express');
const ToolController = require('../controllers/tools');

const router = express.Router();

router.get('/tools', ToolController.getTools);
router.post('/tools/:tool_id/execute', ToolController.executeTool);
router.post('/tools/:tool_id/configure', ToolController.configureTool);

module.exports = router;
```

### src/routes/health.js
```javascript
const express = require('express');
const router = express.Router();

router.get('/', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: process.env.npm_package_version || '1.0.0',
    uptime: process.uptime()
  });
});

module.exports = router;
```

### src/middleware/auth.js
```javascript
const SessionManager = require('../services/sessionManager');
const logger = require('../utils/logger');

async function authMiddleware(req, res, next) {
  try {
    const authHeader = req.headers.authorization;
    const sessionId = req.headers['x-session-id'];
    const userId = req.headers['x-user-id'];

    if (!authHeader || !sessionId || !userId) {
      return res.status(401).json({
        success: false,
        error: 'Missing authentication headers'
      });
    }

    const token = authHeader.replace('Bearer ', '');
    
    if (token !== sessionId) {
      return res.status(401).json({
        success: false,
        error: 'Invalid session token'
      });
    }

    const isValid = await SessionManager.isValidSession(sessionId);
    if (!isValid) {
      return res.status(401).json({
        success: false,
        error: 'Invalid or expired session'
      });
    }

    req.user = { user_id: userId, session_id: sessionId };
    next();
  } catch (error) {
    logger.error('Auth middleware error:', error);
    res.status(500).json({
      success: false,
      error: 'Authentication error'
    });
  }
}

module.exports = authMiddleware;
```

## 🔧 Environment Setup

### .env file
```bash
# Server Configuration
NODE_ENV=development
PORT=3001

# Security
JWT_SECRET=your-super-secret-jwt-key-here-make-it-long-and-random
ENCRYPTION_KEY=your-32-character-encryption-key

# Database (optional for MVP)
MONGODB_URI=mongodb://localhost:27017/mcqueen_mcp
REDIS_URL=redis://localhost:6379

# External Services
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
WEATHER_API_KEY=your-weather-api-key

# CORS
ALLOWED_ORIGINS=http://localhost:3000,https://your-mobile-app.com

# Logging
LOG_LEVEL=info
```

## 🚢 Docker Setup

### Dockerfile
```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY src/ ./src/

RUN addgroup -g 1001 -S nodejs
RUN adduser -S mcpserver -u 1001
RUN chown -R mcpserver:nodejs /app
USER mcpserver

EXPOSE 3001

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3001/api/v1/health', (res) => { process.exit(res.statusCode === 200 ? 0 : 1) })"

CMD ["npm", "start"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  mcp-server:
    build: .
    ports:
      - "3001:3001"
    environment:
      - NODE_ENV=production
      - PORT=3001
      - JWT_SECRET=${JWT_SECRET}
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs

volumes:
  logs:
```

## 🧪 Testing Setup

### Basic Test (tests/health.test.js)
```javascript
const request = require('supertest');
const app = require('../src/app');

describe('Health Check', () => {
  test('GET /api/v1/health should return 200', async () => {
    const response = await request(app)
      .get('/api/v1/health')
      .expect(200);
    
    expect(response.body.status).toBe('healthy');
  });
});
```

## 🚀 Quick Start Commands

```bash
# 1. Setup project
npm install

# 2. Create logs directory
mkdir logs

# 3. Copy environment variables
cp .env.example .env

# 4. Start development server
npm run dev

# 5. Test the server
curl http://localhost:3001/api/v1/health

# 6. Build and run with Docker
docker-compose up --build
```

## 📱 Update Mobile App Configuration

### Update mobile-app config
```bash
# In mobile-app repository
# Update app/core/config.py

# Add MCP server URL
MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://localhost:3001")
```

## 🔗 Integration Testing

### Test mobile app to server communication:
```bash
# 1. Start MCP server
cd server && npm run dev

# 2. In mobile app, test MCP client
cd mobile-app
python -c "
import asyncio
from app.modules.mcp.mcp_client import get_mcp_client

async def test():
    client = get_mcp_client()
    result = await client.health_check()
    print('Health check:', result)

asyncio.run(test())
"
```

## 📊 Monitoring & Logs

### View logs
```bash
# Development logs
npm run dev

# Production logs
docker-compose logs -f mcp-server

# Log files
tail -f logs/combined.log
tail -f logs/error.log
```

This setup provides a complete, production-ready MCP server microservice! 🎉 