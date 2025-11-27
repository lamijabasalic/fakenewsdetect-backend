#!/usr/bin/env python
"""Simple script to run the backend server"""
import uvicorn
import sys
import os

# Add app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

if __name__ == "__main__":
    print("=" * 50)
    print("Starting Bosnian Fake News Detector Backend")
    print("=" * 50)
    print("Server will be available at: http://127.0.0.1:8000")
    print("API docs will be available at: http://127.0.0.1:8000/docs")
    print("=" * 50)
    print()
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

