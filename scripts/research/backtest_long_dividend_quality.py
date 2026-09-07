"""Compatible entrypoint for the packaged dividend-quality implementation."""

import sys

from quant.research import long_dividend_quality as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
