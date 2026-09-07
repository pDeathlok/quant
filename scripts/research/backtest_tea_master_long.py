"""Compatible entrypoint for the packaged Tea Master implementation."""

import sys

from quant.research import tea_master_long as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
