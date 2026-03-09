import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Default to offline mode on CI (GitHub Actions) unless explicitly overridden.
# Locally, no default is set so you can run online tests with your own keys.
if not os.getenv("OFFLINE_MODE") and os.getenv("GITHUB_ACTIONS") == "true":
    os.environ["OFFLINE_MODE"] = "1"
