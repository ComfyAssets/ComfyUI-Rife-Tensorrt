import sys
import os
from unittest.mock import MagicMock

# Root of the project (parent of tests/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mock comfy and other ComfyUI-only modules before anything imports them
for mod_name in [
    "comfy", "comfy.model_management", "comfy.utils",
    "folder_paths", "colored",
]:
    sys.modules.setdefault(mod_name, MagicMock())

# Add root to path so we can import vfi_utilities, utilities, etc.
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
