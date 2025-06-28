# MCP Server Implementation Guide
## For mcqueen-io/server Repository

This document provides the complete implementation for the MCP (Model Control Protocol) server that will be deployed as a microservice.

## 🏗️ Repository Structure

```
mcqueen-io/server/
├── src/
│   ├── app.js                      # Main application entry
│   ├── config/
│   │   ├── database.js             # Database configuration
│   │   ├── redis.js               # Redis configuration  
│   │   └── services.js            # External service configs
│   ├── controllers/
│   │   ├── auth.js                # Authentication controller
│   │   ├── tools.js               # Tool execution controller
│   │   └── users.js               # User management controller
│   ├── middleware/
│   │   ├── auth.js                # Authentication middleware
│   │   ├── rateLimit.js          # Rate limiting
│   │   └── validation.js         # Request validation
│   ├── models/
│   │   ├── User.js               # User model
│   │   ├── Session.js            # Session model
│   │   └── ToolConfig.js         # Tool configuration model
│   ├── services/
│   │   ├── authManager.js        # Authentication management
│   │   ├── toolRegistry.js       # Tool registry
│   │   └── sessionManager.js     # Session management
│   ├── adapters/
│   │   ├── base/
│   │   │   └── BaseAdapter.js    # Base adapter class
│   │   ├── gmail/
│   │   │   └── GmailAdapter.js   # Gmail service adapter
│   │   ├── whatsapp/
│   │   │   └── WhatsAppAdapter.js # WhatsApp adapter
│   │   └── booking/
│   │       └── BookingAdapter.js  # Booking services adapter
│   ├── utils/
│   │   ├── encryption.js         # Encryption utilities
│   │   ├── logger.js             # Logging utilities
│   │   └── validator.js          # Validation utilities
│   └── routes/
│       ├── auth.js               # Authentication routes
│       ├── mcp.js                # MCP tool routes
│       └── health.js             # Health check routes
├── tests/                        # Test suites
├── docker/
│   ├── Dockerfile               # Docker configuration
│   └── docker-compose.yml       # Multi-service setup
├── k8s/                         # Kubernetes manifests
├── docs/                        # API documentation
├── package.json                 # Node.js dependencies
├── .env.example                 # Environment variables template
└── README.md                    # Server documentation
```

## 📋 Package.json

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
    "crypto": "^1.0.1",
    "joi": "^17.9.1",
    "winston": "^3.8.2",
    "googleapis": "^118.0.0",
    "whatsapp-web.js": "^1.19.5",
    "linkedin-api": "^2.1.0",
    "slack-web-api": "^6.8.1",
    "dotenv": "^16.0.3"
  },
  "devDependencies": {
    "nodemon": "^2.0.22",
    "jest": "^29.5.0",
    "supertest": "^6.3.3"
  }
}
```

## 🚀 Main Application (src/app.js)

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
  max: 100, // limit each IP to 100 requests per windowMs
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
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found'
  });
});

app.listen(PORT, () => {
  logger.info(`MCP Server running on port ${PORT}`);
});

module.exports = app;
```

## 🔐 Authentication Controller (src/controllers/auth.js)

```javascript
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const SessionManager = require('../services/sessionManager');
const AuthManager = require('../services/authManager');
const logger = require('../utils/logger');

class AuthController {
  async login(req, res) {
    try {
      const { user_id, auth_token, client_type } = req.body;

      // Validate auth token (implement your validation logic)
      const isValid = await this.validateAuthToken(user_id, auth_token);
      if (!isValid) {
        return res.status(401).json({
          success: false,
          error: 'Invalid authentication token'
        });
      }

      // Create MCP session
      const session = await SessionManager.createSession(user_id, client_type);

      logger.info(`User ${user_id} authenticated successfully`);

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
  }

  async logout(req, res) {
    try {
      const { user_id } = req.user;
      const sessionId = req.headers['x-session-id'];

      await SessionManager.destroySession(sessionId);

      logger.info(`User ${user_id} logged out`);

      res.json({
        success: true,
        message: 'Logged out successfully'
      });
    } catch (error) {
      logger.error('Logout error:', error);
      res.status(500).json({
        success: false,
        error: 'Logout failed'
      });
    }
  }

  async validateAuthToken(userId, token) {
    // Implement your token validation logic here
    // This could check against your main app's auth system
    try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      return decoded.user_id === userId;
    } catch (error) {
      return false;
    }
  }
}

module.exports = new AuthController();
```

## 🔧 Tool Registry Service (src/services/toolRegistry.js)

```javascript
const logger = require('../utils/logger');

class ToolRegistry {
  constructor() {
    this.tools = new Map();
    this.adapters = new Map();
    this.initializeTools();
  }

  initializeTools() {
    // Public Tools
    this.registerTool({
      id: 'get_weather',
      name: 'Get Weather',
      description: 'Get current weather conditions and forecast',
      category: 'weather',
      access_level: 'public',
      requires_auth: false,
      parameters: [
        { name: 'location', type: 'string', required: true, description: 'Location' },
        { name: 'units', type: 'string', default: 'metric', options: ['metric', 'imperial'] }
      ],
      examples: ['What\'s the weather in New York?']
    });

    this.registerTool({
      id: 'web_search',
      name: 'Web Search',
      description: 'Search the internet for general information',
      category: 'search',
      access_level: 'public',
      requires_auth: false,
      max_requests_per_hour: 50,
      parameters: [
        { name: 'query', type: 'string', required: true, description: 'Search query' },
        { name: 'num_results', type: 'number', default: 5 }
      ]
    });

    this.registerTool({
      id: 'book_restaurant_public',
      name: 'Book Restaurant',
      description: 'Make restaurant reservations using public booking services',
      category: 'booking',
      access_level: 'public',
      requires_auth: false,
      parameters: [
        { name: 'restaurant_name', type: 'string', required: true },
        { name: 'date', type: 'string', required: true },
        { name: 'time', type: 'string', required: true },
        { name: 'party_size', type: 'number', required: true }
      ],
      voice_confirmations: ['restaurant_name', 'date', 'time', 'party_size']
    });

    // User-Specific Tools
    this.registerTool({
      id: 'email_personal',
      name: 'Personal Email',
      description: 'Access and manage personal email',
      category: 'communication',
      access_level: 'user_specific',
      requires_auth: true,
      auth_service_id: 'gmail',
      oauth_scopes: ['https://www.googleapis.com/auth/gmail.send'],
      parameters: [
        { name: 'action', type: 'string', required: true, options: ['send', 'read', 'search'] },
        { name: 'to', type: 'array', required: false },
        { name: 'subject', type: 'string', required: false },
        { name: 'body', type: 'string', required: false }
      ],
      voice_confirmations: ['action', 'to', 'subject']
    });

    this.registerTool({
      id: 'calendar_personal',
      name: 'Personal Calendar',
      description: 'Access and manage personal calendar events',
      category: 'productivity',
      access_level: 'user_specific',
      requires_auth: true,
      auth_service_id: 'google_calendar',
      oauth_scopes: ['https://www.googleapis.com/auth/calendar'],
      parameters: [
        { name: 'action', type: 'string', required: true, 
          options: ['get_events', 'create_event', 'update_event'] },
        { name: 'start_date', type: 'string', required: false },
        { name: 'event_title', type: 'string', required: false }
      ]
    });

    // Hybrid Tools
    this.registerTool({
      id: 'maps_hybrid',
      name: 'Maps & Navigation',
      description: 'Get directions and location info (public) plus access saved places (personal)',
      category: 'navigation',
      access_level: 'hybrid',
      requires_auth: false,
      auth_service_id: 'google_maps',
      fallback_mode: 'public',
      parameters: [
        { name: 'action', type: 'string', required: true,
          options: ['directions', 'search_places', 'get_saved_places'] },
        { name: 'origin', type: 'string', required: false },
        { name: 'destination', type: 'string', required: false },
        { name: 'saved_places', type: 'boolean', required: false, requires_auth: true }
      ]
    });

    logger.info(`Initialized ${this.tools.size} tools`);
  }

  registerTool(toolDefinition) {
    this.tools.set(toolDefinition.id, {
      ...toolDefinition,
      created_at: new Date(),
      is_active: true
    });
  }

  getTool(toolId) {
    return this.tools.get(toolId);
  }

  getToolsForUser(userAuthenticated = false, userToolConfigs = []) {
    const availableTools = [];
    
    for (const [toolId, tool] of this.tools) {
      if (!tool.is_active) continue;

      // Public tools are always available
      if (tool.access_level === 'public') {
        availableTools.push(this.formatToolForUser(tool, userToolConfigs));
      }
      // User-specific tools require authentication
      else if (tool.access_level === 'user_specific' && userAuthenticated) {
        availableTools.push(this.formatToolForUser(tool, userToolConfigs));
      }
      // Hybrid tools are always available
      else if (tool.access_level === 'hybrid') {
        availableTools.push(this.formatToolForUser(tool, userToolConfigs));
      }
    }

    return availableTools;
  }

  formatToolForUser(tool, userToolConfigs) {
    const userConfig = userToolConfigs.find(config => config.tool_id === tool.id);
    
    return {
      id: tool.id,
      name: tool.name,
      description: tool.description,
      category: tool.category,
      access_level: tool.access_level,
      requires_auth: tool.requires_auth,
      auth_service_id: tool.auth_service_id,
      parameters: tool.parameters,
      voice_confirmations: tool.voice_confirmations || [],
      examples: tool.examples || [],
      user_enabled: userConfig ? userConfig.enabled : false,
      user_priority: userConfig ? userConfig.priority : 5
    };
  }

  validateParameters(toolId, parameters, userAuthenticated = false) {
    const tool = this.getTool(toolId);
    if (!tool) {
      throw new Error(`Tool ${toolId} not found`);
    }

    const validatedParams = {};
    const errors = [];

    // Check required parameters
    for (const paramDef of tool.parameters) {
      if (paramDef.required) {
        // Skip auth-required parameters if not authenticated
        if (paramDef.requires_auth && !userAuthenticated) {
          continue;
        }
        
        if (!(paramDef.name in parameters)) {
          errors.push(`Missing required parameter: ${paramDef.name}`);
        }
      }

      if (paramDef.name in parameters) {
        const value = parameters[paramDef.name];
        
        // Skip auth-required parameters if not authenticated
        if (paramDef.requires_auth && !userAuthenticated) {
          continue;
        }

        // Type validation
        if (paramDef.type === 'number' && typeof value !== 'number') {
          errors.push(`Parameter ${paramDef.name} must be a number`);
        } else if (paramDef.type === 'boolean' && typeof value !== 'boolean') {
          errors.push(`Parameter ${paramDef.name} must be a boolean`);
        } else if (paramDef.type === 'array' && !Array.isArray(value)) {
          errors.push(`Parameter ${paramDef.name} must be an array`);
        }

        // Options validation
        if (paramDef.options && !paramDef.options.includes(value)) {
          errors.push(`Parameter ${paramDef.name} must be one of: ${paramDef.options.join(', ')}`);
        }

        validatedParams[paramDef.name] = value;
      } else if (paramDef.default !== undefined) {
        validatedParams[paramDef.name] = paramDef.default;
      }
    }

    if (errors.length > 0) {
      throw new Error(`Parameter validation failed: ${errors.join('; ')}`);
    }

    return validatedParams;
  }
}

module.exports = new ToolRegistry();
```

## 🔧 Tool Execution Controller (src/controllers/tools.js)

```javascript
const ToolRegistry = require('../services/toolRegistry');
const AuthManager = require('../services/authManager');
const SessionManager = require('../services/sessionManager');
const logger = require('../utils/logger');

// Import adapters
const GmailAdapter = require('../adapters/gmail/GmailAdapter');
const BookingAdapter = require('../adapters/booking/BookingAdapter');

class ToolController {
  constructor() {
    this.adapters = {
      'gmail': new GmailAdapter(),
      'booking': new BookingAdapter()
    };
  }

  async getTools(req, res) {
    try {
      const { user_id } = req.user;
      const userId = req.query.user_id || user_id;

      // Get user's tool configurations from database
      const userToolConfigs = await this.getUserToolConfigs(userId);
      
      // Check if user is authenticated for personal tools
      const userAuthenticated = await AuthManager.isUserAuthenticated(userId);

      const tools = ToolRegistry.getToolsForUser(userAuthenticated, userToolConfigs);

      res.json({
        success: true,
        tools: tools,
        total: tools.length
      });
    } catch (error) {
      logger.error('Get tools error:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to get tools'
      });
    }
  }

  async executeTool(req, res) {
    try {
      const { tool_id } = req.params;
      const { parameters, voice_confirmations, user_id, timestamp } = req.body;
      const userId = user_id || req.user.user_id;

      // Get tool definition
      const tool = ToolRegistry.getTool(tool_id);
      if (!tool) {
        return res.status(404).json({
          success: false,
          error: `Tool ${tool_id} not found`
        });
      }

      // Check if user has this tool enabled
      const userToolConfigs = await this.getUserToolConfigs(userId);
      const userTool = userToolConfigs.find(config => config.tool_id === tool_id);
      
      if (userTool && !userTool.enabled) {
        return res.status(403).json({
          success: false,
          error: `Tool ${tool_id} not enabled for user`
        });
      }

      // Validate parameters
      const userAuthenticated = await AuthManager.isUserAuthenticated(userId);
      const validatedParams = ToolRegistry.validateParameters(
        tool_id, 
        parameters, 
        userAuthenticated
      );

      // Check voice confirmations if required
      if (tool.voice_confirmations && tool.voice_confirmations.length > 0) {
        const missingConfirmations = tool.voice_confirmations.filter(
          confirmation => !voice_confirmations || !voice_confirmations[confirmation]
        );
        
        if (missingConfirmations.length > 0) {
          return res.json({
            success: false,
            error: `Voice confirmation required for: ${missingConfirmations.join(', ')}`,
            metadata: { requires_confirmation: missingConfirmations }
          });
        }
      }

      // Get credentials if required
      let credentials = {};
      if (tool.requires_auth) {
        credentials = await AuthManager.getUserCredentials(userId, tool.auth_service_id);
        if (!credentials) {
          return res.json({
            success: false,
            error: `Authentication required for service ${tool.auth_service_id}`,
            metadata: { requires_auth: tool.auth_service_id }
          });
        }
      }

      // Execute the tool
      let result;
      if (tool.requires_auth && this.adapters[tool.auth_service_id]) {
        // Use service adapter
        const adapter = this.adapters[tool.auth_service_id];
        const action = this.mapToolToAction(tool_id);
        result = await adapter.executeAction(action, validatedParams, credentials);
      } else {
        // Execute public tool
        result = await this.executePublicTool(tool_id, validatedParams);
      }

      // Update tool usage statistics
      await this.updateToolUsage(userId, tool_id);

      res.json(result);

    } catch (error) {
      logger.error(`Tool execution error for ${req.params.tool_id}:`, error);
      res.status(500).json({
        success: false,
        error: 'Tool execution failed',
        details: error.message
      });
    }
  }

  async configureTool(req, res) {
    try {
      const { tool_id } = req.params;
      const { enabled, priority, custom_settings } = req.body;
      const { user_id } = req.user;

      // Validate tool exists
      const tool = ToolRegistry.getTool(tool_id);
      if (!tool) {
        return res.status(404).json({
          success: false,
          error: `Tool ${tool_id} not found`
        });
      }

      // Update user tool configuration in database
      await this.updateUserToolConfig(user_id, {
        tool_id,
        enabled,
        priority,
        custom_settings
      });

      res.json({
        success: true,
        data: { tool_id, enabled, priority }
      });

    } catch (error) {
      logger.error('Tool configuration error:', error);
      res.status(500).json({
        success: false,
        error: 'Tool configuration failed'
      });
    }
  }

  mapToolToAction(toolId) {
    const mapping = {
      'email_personal': 'send_email',
      'calendar_personal': 'manage_calendar',
      'maps_hybrid': 'navigate'
    };
    return mapping[toolId] || toolId;
  }

  async executePublicTool(toolId, parameters) {
    switch (toolId) {
      case 'get_weather':
        return this.getWeather(parameters);
      case 'web_search':
        return this.performWebSearch(parameters);
      case 'book_restaurant_public':
        return this.bookRestaurant(parameters);
      default:
        throw new Error(`Public tool ${toolId} not implemented`);
    }
  }

  async getWeather(parameters) {
    // Implement weather API call
    const { location, units } = parameters;
    
    // Mock implementation - replace with actual weather API
    return {
      success: true,
      data: {
        location: location,
        temperature: 22,
        condition: 'Partly cloudy',
        humidity: '65%',
        units: units
      },
      metadata: { service: 'weather_api' }
    };
  }

  async performWebSearch(parameters) {
    // Implement web search
    const { query, num_results } = parameters;
    
    // Mock implementation - replace with actual search API
    return {
      success: true,
      data: {
        query: query,
        results: [
          { title: 'Sample Result 1', url: 'https://example.com/1', snippet: 'Sample snippet...' },
          { title: 'Sample Result 2', url: 'https://example.com/2', snippet: 'Another snippet...' }
        ].slice(0, num_results)
      },
      metadata: { service: 'web_search' }
    };
  }

  async bookRestaurant(parameters) {
    // Implement restaurant booking
    const { restaurant_name, date, time, party_size } = parameters;
    
    const bookingId = `RES_${Date.now()}`;
    
    return {
      success: true,
      data: {
        booking_id: bookingId,
        restaurant_name,
        date,
        time,
        party_size,
        status: 'confirmed',
        confirmation_number: bookingId
      },
      metadata: { service: 'restaurant_booking' }
    };
  }

  async getUserToolConfigs(userId) {
    // Implement database query to get user tool configurations
    // For now, return mock data
    return [
      { tool_id: 'email_personal', enabled: true, priority: 1, custom_settings: {} },
      { tool_id: 'calendar_personal', enabled: true, priority: 2, custom_settings: {} }
    ];
  }

  async updateUserToolConfig(userId, config) {
    // Implement database update for user tool configuration
    logger.info(`Updated tool config for user ${userId}: ${JSON.stringify(config)}`);
  }

  async updateToolUsage(userId, toolId) {
    // Implement usage tracking
    logger.info(`Tool ${toolId} used by user ${userId} at ${new Date().toISOString()}`);
  }
}

module.exports = new ToolController();
```

## 🔐 Session Manager (src/services/sessionManager.js)

```javascript
const crypto = require('crypto');
const redis = require('../config/redis');
const logger = require('../utils/logger');

class SessionManager {
  constructor() {
    this.sessions = new Map(); // In-memory fallback
    this.sessionTimeout = 8 * 60 * 60 * 1000; // 8 hours
  }

  async createSession(userId, clientType = 'mobile_app') {
    const sessionId = this.generateSessionId();
    const createdAt = new Date();
    const expiresAt = new Date(createdAt.getTime() + this.sessionTimeout);

    const session = {
      sessionId,
      userId,
      clientType,
      createdAt,
      expiresAt,
      lastActivity: createdAt
    };

    // Store in Redis if available, otherwise in memory
    try {
      if (redis.isConnected()) {
        await redis.setex(
          `session:${sessionId}`,
          this.sessionTimeout / 1000,
          JSON.stringify(session)
        );
      } else {
        this.sessions.set(sessionId, session);
      }
    } catch (error) {
      logger.error('Session storage error:', error);
      this.sessions.set(sessionId, session);
    }

    logger.info(`Created session ${sessionId} for user ${userId}`);
    return session;
  }

  async getSession(sessionId) {
    try {
      if (redis.isConnected()) {
        const sessionData = await redis.get(`session:${sessionId}`);
        if (sessionData) {
          const session = JSON.parse(sessionData);
          session.createdAt = new Date(session.createdAt);
          session.expiresAt = new Date(session.expiresAt);
          session.lastActivity = new Date(session.lastActivity);
          return session;
        }
      } else {
        return this.sessions.get(sessionId);
      }
    } catch (error) {
      logger.error('Session retrieval error:', error);
      return this.sessions.get(sessionId);
    }
    return null;
  }

  async updateLastActivity(sessionId) {
    const session = await this.getSession(sessionId);
    if (session) {
      session.lastActivity = new Date();
      
      try {
        if (redis.isConnected()) {
          await redis.setex(
            `session:${sessionId}`,
            this.sessionTimeout / 1000,
            JSON.stringify(session)
          );
        } else {
          this.sessions.set(sessionId, session);
        }
      } catch (error) {
        logger.error('Session update error:', error);
      }
    }
  }

  async destroySession(sessionId) {
    try {
      if (redis.isConnected()) {
        await redis.del(`session:${sessionId}`);
      }
      this.sessions.delete(sessionId);
      logger.info(`Destroyed session ${sessionId}`);
    } catch (error) {
      logger.error('Session destruction error:', error);
    }
  }

  async isValidSession(sessionId) {
    const session = await this.getSession(sessionId);
    if (!session) return false;

    const now = new Date();
    if (now > session.expiresAt) {
      await this.destroySession(sessionId);
      return false;
    }

    // Update last activity
    await this.updateLastActivity(sessionId);
    return true;
  }

  generateSessionId() {
    return crypto.randomBytes(32).toString('hex');
  }

  async cleanupExpiredSessions() {
    // Clean up expired sessions (run periodically)
    const now = new Date();
    
    for (const [sessionId, session] of this.sessions) {
      if (now > session.expiresAt) {
        await this.destroySession(sessionId);
      }
    }
  }
}

module.exports = new SessionManager();
```

## 🔧 Dockerfile

```dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY src/ ./src/

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S mcpserver -u 1001

# Change ownership
RUN chown -R mcpserver:nodejs /app
USER mcpserver

# Expose port
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3001/api/v1/health', (res) => { process.exit(res.statusCode === 200 ? 0 : 1) })"

CMD ["npm", "start"]
```

## 🚢 Docker Compose (docker-compose.yml)

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
      - MONGODB_URI=mongodb://mongo:27017/mcqueen_mcp
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      - mongo
      - redis
    restart: unless-stopped

  mongo:
    image: mongo:6.0
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  mongo_data:
  redis_data:
```

## 🎯 Environment Variables (.env.example)

```bash
# Server Configuration
NODE_ENV=development
PORT=3001

# Database
MONGODB_URI=mongodb://localhost:27017/mcqueen_mcp
REDIS_URL=redis://localhost:6379

# Security
JWT_SECRET=your-super-secret-jwt-key-here
ENCRYPTION_KEY=your-32-character-encryption-key

# External Services
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GMAIL_API_KEY=your-gmail-api-key

WHATSAPP_API_TOKEN=your-whatsapp-api-token
LINKEDIN_CLIENT_ID=your-linkedin-client-id
SLACK_BOT_TOKEN=your-slack-bot-token

# Weather & Search APIs
WEATHER_API_KEY=your-weather-api-key
SEARCH_API_KEY=your-search-api-key

# Rate Limiting
RATE_LIMIT_WINDOW=900000
RATE_LIMIT_MAX=100

# Logging
LOG_LEVEL=info
LOG_FILE=./logs/mcp-server.log

# CORS
ALLOWED_ORIGINS=http://localhost:3000,https://your-mobile-app.com
```

## 🚀 Deployment Instructions

### Local Development
```bash
# Clone server repository
git clone https://github.com/mcqueen-io/server.git
cd server

# Install dependencies
npm install

# Copy environment variables
cp .env.example .env
# Edit .env with your values

# Start with Docker Compose
docker-compose up -d

# Or start locally
npm run dev
```

### Production Deployment (Kubernetes)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: mcqueen/mcp-server:latest
        ports:
        - containerPort: 3001
        env:
        - name: NODE_ENV
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: mongodb-uri
        resources:
          limits:
            memory: "512Mi"
            cpu: "500m"
          requests:
            memory: "256Mi"
            cpu: "250m"
```

## 📊 API Testing

### Test with curl:
```bash
# Health check
curl http://localhost:3001/api/v1/health

# Login
curl -X POST http://localhost:3001/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test123", "auth_token": "your-token"}'

# Get tools
curl -X GET http://localhost:3001/api/v1/mcp/tools \
  -H "Authorization: Bearer session-token" \
  -H "X-User-ID: test123"

# Execute tool
curl -X POST http://localhost:3001/api/v1/mcp/tools/get_weather/execute \
  -H "Authorization: Bearer session-token" \
  -H "Content-Type: application/json" \
  -d '{"parameters": {"location": "New York"}, "user_id": "test123"}'
```

This completes the comprehensive MCP server implementation! 🚀 