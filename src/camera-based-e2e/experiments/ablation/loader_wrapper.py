"""
Wrapper loader that fixes the protobuf import issue.
This imports the local protos instead of waymo_open_dataset.protos
"""
import sys
from pathlib import Path
import types

# Add base directory (camera-based-e2e) to path so we can import protos and loader
base_dir = Path(__file__).parent.parent.parent  # camera-based-e2e
base_dir_str = str(base_dir)
if base_dir_str not in sys.path:
    sys.path.insert(0, base_dir_str)

# Patch the import before loading the original loader
# Create a mock waymo_open_dataset.protos module that points to local protos
sys.modules['waymo_open_dataset'] = types.ModuleType('waymo_open_dataset')

# Import the actual local protos - this will work because base_dir is in sys.path
from protos import e2e_pb2

# Create mock module that has the expected attribute name
mock_protos = types.ModuleType('waymo_open_dataset.protos')
mock_protos.end_to_end_driving_data_pb2 = e2e_pb2
sys.modules['waymo_open_dataset.protos'] = mock_protos

# Now import the original loader - it will use our mocked module
from loader import WaymoE2E

# Export the class
__all__ = ['WaymoE2E']
