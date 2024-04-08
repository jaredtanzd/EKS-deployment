import os
import sys

import yaml
from pathlib import Path
from importlib import import_module

path = Path(__file__).parent

with open(path / "backend_model.yaml", 'r') as f:
    valuesYaml = yaml.load(f, Loader=yaml.FullLoader)

# # ICH project API
# sys.path.insert(1, str( path/ 'models' / valuesYaml['ICH']['project_path']))
# ich_model = import_module(path.stem + '.models.'+ valuesYaml['ICH']['project_path'])

# # Tissueseg project API
# sys.path.insert(1, str( path/ 'models' / valuesYaml['tissueseg']['project_path']))
# p2p = import_module(path.stem + '.models.'+ valuesYaml['tissueseg']['project_path'])

# Baseline project API
sys.path.insert(1, str( path/ 'models' / valuesYaml['baseline']['project_path']))
unet = import_module(path.stem + '.models.'+ valuesYaml['baseline']['project_path']) 
