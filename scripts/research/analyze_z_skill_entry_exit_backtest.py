"""Compatible entrypoint for the packaged Z-skill rules."""

import sys

from quant.research import z_skill_rules as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
