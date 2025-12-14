"""
WSGI entry point for production deployment.
Used by Gunicorn, uWSGI, and other WSGI servers.
"""
import os
import sys

# Add backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_dir)
sys.path.insert(0, os.path.dirname(__file__))

# Set environment if not set
if not os.environ.get('FLASK_ENV'):
    os.environ['FLASK_ENV'] = 'production'

# Import the Flask app
from backend.app import app

# For WSGI servers
application = app

if __name__ == "__main__":
    # For development/testing
    from backend.config import get_config
    config = get_config()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)

