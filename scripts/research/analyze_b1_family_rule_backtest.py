"""Compatible entrypoint for the packaged B1-family rules."""

import sys

from quant.research import b1_family_rules as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
