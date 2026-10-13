"""
Tanim Guard - start the app.
Run with: python run.py   (or double-click start.bat)
"""
import os
import socket
import threading
import webbrowser

from app import create_app

app = create_app()


def lan_address() -> str | None:
    """This PC's address on the Wi-Fi, for opening the app on a phone."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; picks the LAN interface
            return s.getsockname()[0]
    except OSError:
        return None


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    print(f"\n  Tanim Guard:  http://localhost:{port}")
    if (ip := lan_address()):
        print(f"  Sa cellphone (parehong Wi-Fi):  http://{ip}:{port}\n")
    if os.getenv('TANIM_OPEN_BROWSER') == '1':
        threading.Timer(1.5, webbrowser.open, args=(f"http://localhost:{port}",)).start()
    # 0.0.0.0 so a phone on the same Wi-Fi can open it. Debug stays OFF: with
    # debug on, anyone on that Wi-Fi gets the Werkzeug console and can run code
    # on this PC. Set TANIM_DEBUG=1 only when working alone.
    app.run(host='0.0.0.0', port=port, debug=os.getenv('TANIM_DEBUG') == '1', threaded=True)
