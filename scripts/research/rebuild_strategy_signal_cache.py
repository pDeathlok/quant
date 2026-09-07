#!/usr/bin/env python
"""Compatible entrypoint for the packaged signal-cache rebuild."""

import sys

from quant.research import strategy_signal_cache as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
