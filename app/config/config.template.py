"""
Configuration template for the application.
Copy this file to config.py and fill in your actual values.
DO NOT commit config.py to version control.
"""

# API Keys and Credentials
API_KEYS = {
    # Add your API keys here
    # 'service_name': 'your_api_key_here',
}

# Database Configuration
DATABASE = {
    'host': 'localhost',
    'port': 5432,
    'database': 'your_database_name',
    'user': 'your_username',
    'password': 'your_password'
}

# Voice Processing Configuration
VOICE_PROCESSING = {
    'sample_rate': 16000,
    'chunk_size': 1024,
    'channels': 1,
    'format': 'int16'
}

# Application Settings
APP_SETTINGS = {
    'debug': False,
    'log_level': 'INFO',
    'max_upload_size': 10 * 1024 * 1024  # 10MB
} 