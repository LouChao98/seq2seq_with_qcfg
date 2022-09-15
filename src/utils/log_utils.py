import logging
import sys

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.INFO, strm=sys.stdout):
        super().__init__(level)
        self.strm = strm

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.strm)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
