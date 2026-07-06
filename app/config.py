"""Pydantic settings loaded from environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration from env vars."""

    LIVENESS_THRESHOLD: float = 0.5
    HOST: str = "0.0.0.0"
    PORT: int = 8090
    MODEL_DIR: Path = Path(__file__).resolve().parent.parent / "models"
    DEVICE: str = "auto"
    MAX_BATCH: int = 16

    # Phase 1 PAD-gate integration (BACKEND_REQUIREMENTS_2026-07-06)
    SERVICE_TOKEN: str = ""  # X-Service-Token shared secret with Laravel; empty = auth disabled
    RATE_LIMIT_BURST: int = 20  # max concurrent requests (per-second burst)
    RATE_LIMIT_SUSTAINED: float = 5.0  # sustained requests per second
    SAVE_FRAME_VERDICTS: str = "spoof"  # comma-separated verdicts that trigger save_frame=true

    model_config = {"env_prefix": ""}


def resolve_device(requested: str) -> str:
    """Map device string to actual torch device name."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested
