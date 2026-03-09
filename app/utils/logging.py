import logging
import os

LOG_FILE = "app.log"

# Suppress noisy pkg_resources UserWarning from third-party packages during tests
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

# File handler (keeps INFO logs in file)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)

# Stream handler (stderr) should be less noisy for test runs; only show WARNING+
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[file_handler, stream_handler],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
