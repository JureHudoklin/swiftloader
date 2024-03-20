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
                 datasets_info: List[DatasetInfo],
                 ):
        
        self.root_dir = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.objects_root_dir = objects_root_dir if isinstance(objects_root_dir, Path) else Path(objects_root_dir)
        self.datasets_info = datasets_info
        
        self.cat_map, self.cats = self._build_category_map(self.root_dir, self.datasets_info)
        self.objects = self._load_objects(self.objects_root_dir)
        
    def _build_category_map(
        self, data_root: Path, datasets_info: List[DatasetInfo]
    ) -> Tuple[Dict[str, Dict[int, int]], Dict[int, CocoCat]]:
        cat_map = defaultdict(dict)
        cats = {}
        
        dataset_cats = {}
        for dataset_info in datasets_info:
            cats_path = data_root / dataset_info["name"] / "categories.json"
            if not cats_path.exists():
                raise ValueError(f"Categories file not found for dataset {dataset_info['name']}")
            with open(data_root / dataset_info["name"] / "categories.json") as f:
                categories = json.load(f)
                dataset_cats[dataset_info["name"]] = categories
        new_cat_id = 0

        for dataset, categories in dataset_cats.items():
            for cat in categories:
                new_cat_id += 1
                cat_map[dataset][cat["id"]] = new_cat_id
                cat["id"] = new_cat_id
                cats[new_cat_id] = cat

        return cat_map, cats
        
    def _load_objects(self, objects_root_dir: Path) -> Dict[str, List[str]]:
        objects = {}
        for object_dir in objects_root_dir.iterdir():
            object_name = object_dir.name
            
            
            files = [str(file) for file in object_dir.iterdir() if file.is_file()]
            files = np.array(sorted(files)).astype(np.string_)
            objects[object_name] = files

        return objects
        
        
class TemplateDetectionDatasetParqeut(ParquetDataset, TempleteDetectionBase):
    def __init__(self,
                    root_dir: str | Path,   
                    datasets_info: List[DatasetInfo],
                    batch_size: int,
                    format_data: Callable[[List[dict]], Any] | None = None,
                    drop_last: bool = False,
                    shuffle: bool = True,
                 ) -> None:
        
        ParquetDataset.__init__(self,
                                root_dir=root_dir,
                                datasets_info=datasets_info,
                                batch_size=batch_size,
                                format_data=format_data,
                                drop_last=drop_last,
                                shuffle=shuffle)
        TempleteDetectionBase.__init__(self,
                                        root_dir=root_dir,
                                        objects_root_dir=objects_root_dir,
                                        datasets_info=datasets_info)
        