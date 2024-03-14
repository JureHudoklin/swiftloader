import os
import time
import logging
import json
import tempfile
import numpy as np
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as PILImage
from pathlib import Path
from typing import Callable, List, Dict, Optional, Tuple, Literal, Any, Sequence, TypeAlias
from collections import defaultdict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .util.type_structs import DatasetInfo

ImageFile.LOAD_TRUNCATED_IMAGES = True

def loader(ext: str, path: Path) -> Any:
    if ext == ".json":
        with open(path, "r") as f:
            return json.load(f)    
    elif ext in [".jpg", ".jpeg", ".png"]:
        with Image.open(path) as img:
                image = img.convert("RGB")
                image = ImageOps.exif_transpose(image)
        return image
    elif ext in [".npy"]:
        return np.load(path)
    elif ext in [".pt", ".pth"]:
        return torch.load(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

class FolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        datasets_info: List[DatasetInfo],
        format_data: Callable[[dict], Any] | None = None,
        data_folders: List[Dict[Literal["name", "ext"], str]] = [{"name": "images", "ext": "jpg"}],
        annotations_folders: List[str] = ["annotations"],
        data_loader: Callable[[str, Path], Any] = loader,
        *args,
        **kwargs
    ) -> None:
        """_summary_

        Parameters
        ----------
        root_dir : str | Path
            A path to the root directory of the datasets. Each dataset should be in a separate folder.
            The dataset folder should contain the following subfolders listed in `data_folders` and `annotations_folders`.
            To get the number of entries in the dataset, the first folder in `data_folders` will be used.
        datasets_info : List[DatasetInfo]
            A list of dictionaries containing information about the datasets. Each dictionary should contain the following
                - name: str
                    The name of the dataset
                - scenes: List[str]
                    A list of scenes to load from the dataset. If None, all scenes will be loaded.
        format_data : Callable[[List[dict]], Any] | None, optional
            A function that takes as an input a dictionary of data and returns a formatted version of the data. If None, the default
        """
        self.root_dir = Path(root_dir)
        self.datasets_info = sorted(datasets_info, key=lambda x: x["name"])
        self.format_data = format_data
        self.data_folders = data_folders
        self.annotations_folders = annotations_folders
        self.data_loader = data_loader

        self._check_datasets_exist(self.root_dir, self.datasets_info)
        self.data = self._load_dataset(self.root_dir, self.datasets_info)

        logging.info(f"Loaded {len(self.data)} images from {datasets_info} datasets")
        
        self.fail_save = self.__getitem__(0)

    def _check_datasets_exist(self, root_dir: Path, datasets_info: List[DatasetInfo]):
        for dataset_info in datasets_info:
            if not (root_dir / dataset_info["name"]).exists():
                raise FileNotFoundError(
                    f"Dataset {dataset_info['name']} could not be found in {root_dir}"
                )

    def _load_dataset(self, root_dir: Path, datasets_info: List[DatasetInfo]) -> np.ndarray:
        data_ = []
        for dataset_info in datasets_info:
            # Get all subfolders in the dataset folder (exclude files)
            scenes = sorted((root_dir / dataset_info["name"]).glob("*"))
            scenes = [scene for scene in scenes if scene.is_dir()]
            if "scenes" in dataset_info:
                scenes = [scene for scene in scenes if scene.name in dataset_info["scenes"]]
            
            for scene in scenes:
                data_path = scene / "data"
                
                files = os.scandir(data_path / self.data_folders[0]["name"])
                data_.extend(
                    [
                        f"{dataset_info['name']}/{scene.name}/{file.name}"
                        for file in files
                        if file.is_file()
                    ]
                )

        if len(data_) < 1000000:
            data_.sort()
        else:
            logging.warning(
                "Sorting data will take a long time. Skipping sorting. data will be loaded in random order."
            )

        data = np.array(data_).astype(np.string_)
        return data

    def _get_data_paths(self, idx: int) -> Tuple[List[Path], dict]:
        # Get data paths
        data = self.data[idx].decode("utf-8")
        dataset, scene, data_name = data.split("/")
        data_name = data_name.split(".")[0]
        data_info = {"dataset": dataset, "scene": scene, "image_name": data_name}

        paths = []
        for folder in self.data_folders:
            paths.append(self.root_dir / dataset / scene / folder["name"] / f"{data_name}.{folder['ext']}")
        for folder in self.annotations_folders:
            paths.append(self.root_dir / dataset / scene / folder / f"{data_name}.json")

        return paths, data_info

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        paths, data_info = self._get_data_paths(idx)
        
        data_dict = {}
        for path in paths:
            if not path.exists():
                continue
            data = self.data_loader(path.suffix, path)
            data_dict[path.parent.name] = data

        return self.format_data(data_dict) if self.format_data is not None else data_dict
