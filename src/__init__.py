import logging
import os

from hydra._internal.utils import is_under_debugger as _is_under_debugger
from omegaconf import OmegaConf

logger = logging.getLogger("src")
_hit_debug, debugging = True, False


def is_under_debugger():
    if os.environ.get("DEBUG_MODE", "").lower() in ("true", "t", "1", "yes", "y"):
        result = True
    else:
        result = _is_under_debugger()
    global _hit_debug, debugging
    if result and _hit_debug:
        logger.warning("Debug mode.")
        _hit_debug = False
        debugging = True
    return result


OmegaConf.register_new_resolver(
    "in_debugger", lambda x, default=None: x if is_under_debugger() else default
)


def in_cluster():
    return os.environ.get("SLURM_JOB_ID") is not None


OmegaConf.register_new_resolver("in_cluster", in_cluster)


def huggingface_path_helper(name, local_path):
    if os.path.exists(local_path):
        return local_path
    return name


OmegaConf.register_new_resolver("hf", huggingface_path_helper)
