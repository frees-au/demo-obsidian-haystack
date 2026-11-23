"""Narrow logging configuration for local embedding scripts."""

import logging
import sys


class _MacOSRedirectNotice(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Torch emits this at import time even though these scripts do not use
        # its distributed multiprocessing output redirection.
        return record.getMessage() != "NOTE: Redirects are currently not supported in MacOs."


def configure_logging() -> None:
    if sys.platform == "darwin":
        logger = logging.getLogger("torch.distributed.elastic.multiprocessing.redirects")
        if not any(isinstance(item, _MacOSRedirectNotice) for item in logger.filters):
            logger.addFilter(_MacOSRedirectNotice())
