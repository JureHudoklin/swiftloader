from typing import Dict, List, Tuple, Callable, Any, Literal
from pathlib import Path
import json
import io
import tempfile
from collections import defaultdict
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as PILImage

import torch
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms.v2 import functional as TF
from pycocotools.coco import COCO
from target_utils import Target
from target_utils.util import target_filter
from target_utils.formating import target_set_dtype

from .folder_dataset import FolderDataset
from .parquet_dataset import ParquetDataset 
from .util.type_structs import DatasetInfo, CocoCat
from .util.misc import HiddenPrints, loader

import numpy as np




class TempleteDetectionBase:
    def __init__(self,
                 root_dir: str | Path,
                 objects_root_dir: str | Path,
                 datasets_info: List[DatasetInfo],
                 ):
        
        self.root_dir = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.objects_root_dir = objects_root_dir if isinstance(objects_root_dir, Path) else Path(objects_root_dir)
        self.datasets_info = datasets_info