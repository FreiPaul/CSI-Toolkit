"""Allow package to be run as a module: python -m csi_toolkit"""

import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())
