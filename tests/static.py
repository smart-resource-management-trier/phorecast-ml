"""
This file contains static variables like paths or constants which are referenced from multiple
places in the code.
"""
import os
from pathlib import Path

# Persistent/dynamic data
# -Directories
project_root_path = Path(os.path.dirname(__file__)).parent

## static directories
test_path = os.path.join(project_root_path, "tests")

## static test data
test_dataset_file = os.path.join(test_path, "test_dataset.csv")
