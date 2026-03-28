"""GPU VRAM management utilities for sequential model loading."""
import gc
import logging

log = logging.getLogger(__name__)


def clear_gpu() -> None:
    """Aggressively free all GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    gc.collect()


def gpu_memory_info() -> dict:
    """Return current GPU memory statistics in MB."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "allocated_mb": round(torch.cuda.memory_allocated(0) / 1024**2, 1),
            "reserved_mb": round(torch.cuda.memory_reserved(0) / 1024**2, 1),
            "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1),
        }
    except ImportError:
        return {"available": False}


def log_memory_state(label: str = "") -> None:
    info = gpu_memory_info()
    if not info.get("available"):
        log.info("[VRAM] CUDA not available")
        return
    prefix = f"[VRAM {label}] " if label else "[VRAM] "
    log.info(
        "%s%s — allocated: %.0f MB / reserved: %.0f MB / total: %.0f MB",
        prefix, info["device"],
        info["allocated_mb"], info["reserved_mb"], info["total_mb"],
    )
