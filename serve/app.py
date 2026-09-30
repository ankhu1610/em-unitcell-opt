"""
Entry point for serving the EM Unit Cell AI Design Studio.
Redirects to root app.py for unified interactive application.
"""
import os
import sys

# Ensure root is on path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import create_app

if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
