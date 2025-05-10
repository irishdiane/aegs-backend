# flask-server/config.py

import os
from datetime import timedelta

class Config:
    """Base configuration."""
    # Secret key for session management
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-for-testing')
    
    # Debug mode
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # File upload settings
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
    
    # Evaluator settings
    EVALUATORS = {
        'grammar': True,
        'ideas': True,
        'organization': True,
        'evidence': True, 
        'language_tone': True
    }
    
    # Weights for different evaluation criteria
    EVALUATION_WEIGHTS = {
        'grammar': 0.2,
        'ideas': 0.25,
        'organization': 0.2,
        'evidence': 0.2,
        'language_tone': 0.15
    }
    
    # Result caching settings
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    
    # Database settings (if using a database)
    # SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///essay_evaluator.db')
    # SQLALCHEMY_TRACK_MODIFICATIONS = False

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    # Use in-memory database for testing
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    # In production, these should be set as environment variables
    SECRET_KEY = os.getenv('SECRET_KEY')

# Configure based on environment
config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

# Set active configuration
active_config = config_by_name[os.getenv('FLASK_ENV', 'default')]