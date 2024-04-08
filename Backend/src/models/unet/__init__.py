
import yaml

from pathlib import Path

from .run_baseline import unet
# from . import utils

with open(Path(__file__).parent / "backend_unet.yaml", 'r') as f:
    valuesYaml = yaml.load(f, Loader=yaml.FullLoader)

tissue_classes = [ 'wm', 'gm', 'csf', 'vent', 'bet']


#model = unet(valuesYaml)

models = dict(zip(tissue_classes,map(lambda c: unet(valuesYaml, c) , tissue_classes)))
