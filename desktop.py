"""TanimGuard.exe entry point (PyInstaller, see build_exe.ps1).

Same rule as CarScoot Hub: nothing the app writes may live inside the bundle -
PyInstaller unpacks into a temp folder that is deleted on exit. So the database,
the photos and the secret key go in data/ next to the exe, and the app opens in
the normal browser (a phone on the same Wi-Fi can use it too).
"""

import os
import secrets
import sys
import threading
import webbrowser


def _disable_quick_edit():
    """A click in a Windows console window starts text selection ("QuickEdit"),
    and while it is active every write to the console blocks - so the server,
    which logs each request, freezes until someone presses a key. Turn it off."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            ENABLE_QUICK_EDIT_MODE, ENABLE_EXTENDED_FLAGS = 0x40, 0x80
            kernel32.SetConsoleMode(handle, (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS)
    except Exception:
        pass


def main():
    _disable_quick_edit()
    home =os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
        else os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(home, "data")
    os.makedirs(os.path.join(data, "uploads"), exist_ok=True)
    os.chdir(home)

    key_file = os.path.join(data, "secret_key")
    if not os.path.exists(key_file):
        with open(key_file, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_hex(32))
    with open(key_file, encoding="utf-8") as handle:
        os.environ["SECRET_KEY"] = handle.read().strip()

    from app import create_app
    from run import lan_address

    app = create_app({
        "DATABASE_URL": "sqlite:///" + os.path.join(data, "tanim_guard.db"),
        "UPLOAD_FOLDER": os.path.join(data, "uploads"),
    })
    port = int(os.getenv("PORT", "5000"))
    print(f"\n  Tanim Guard is running:  http://localhost:{port}")
    if (ip := lan_address()):
        print(f"  Sa cellphone (parehong Wi-Fi):  http://{ip}:{port}")
    print("  Close this window to stop.\n")
    threading.Timer(1.5, webbrowser.open, args=(f"http://localhost:{port}",)).start()
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
