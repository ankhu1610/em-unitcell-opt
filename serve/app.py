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
    import socket
    def find_free_port(start_port=7860):
        for port in range(start_port, start_port + 50):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        return 7860

    port = find_free_port(7860)
    print(f"\n🚀 Opening EM Unit Cell AI Studio at http://127.0.0.1:{port} ...")
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=True)
