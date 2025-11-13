"""Unified entrypoint: launch GUI or run smoke tests.

Usage:
    python final.py gui        # launch the GUI
    python final.py smoke      # run baseline smoke test headless
"""
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python final.py [gui|smoke]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "gui":
        # Lazy import GUI so we don't import heavy modules during smoke runs
        from gui import CellularSimApp
        app = CellularSimApp()
        app.mainloop()
    elif cmd == "smoke":
        from tests.test_baseline_smoke import test_baseline_smoke

        test_baseline_smoke()
    else:
        print("Unknown command. Use 'gui' or 'smoke'.")
        sys.exit(1)
