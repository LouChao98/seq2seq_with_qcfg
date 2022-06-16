import contextlib
import pytorch_lightning as pl
import pickle
from pathlib import Path
from .components.file_utils import get_hash, iter_dir
import logging

logger = logging.getLogger(__file__)


class _DataModule(pl.LightningDataModule):
    """
    1. support caching
    2. support persistent variables
    """

    def __init__(self, cache_dir="data/cache", enable_cache=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir

        self._persistent_variables = []
        self._tracing_flag = False

        if enable_cache:
            self.setup_orig = self.setup
            self.setup = self.setup_proxy

    @contextlib.contextmanager
    def trace_persistent_variables(self):
        self._tracing_flag = True
        yield
        self._tracing_flag = False

    def __setattr__(self, key, value):
        if not key.startswith("_") and getattr(self, "_tracing_flag", False):
            self._persistent_variables.append(key)
        self.__dict__[key] = value

    def setup_proxy(self, *args, **kwargs):
        hash_val = get_hash(iter_dir(Path(__file__).parent), **self.hparams)
        cache_file = Path(self.cache_dir, hash_val)
        if cache_file.exists():
            if len(args) + len(kwargs) > 0:
                logger.warning("Args are ignored because I loads data from caches.")
            with open(cache_file, "rb") as f:
                self.__dict__.update(pickle.load(f))
        else:
            self.setup_orig(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(
                    {k: self.__dict__[k] for k in self._persistent_variables}, f
                )
