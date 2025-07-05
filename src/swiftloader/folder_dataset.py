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

class FolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        datasets_info: List[DatasetInfo],
        dataset_schema: List[Dict[Literal["field", "dtype", "loader"], Any]],
        format_data: Callable[[dict], Any] | None = None,
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
        self.dataset_schema = dataset_schema

        self._check_datasets_exist(self.root_dir, self.datasets_info)
        self.data = self._load_dataset(self.root_dir, self.datasets_info)

        logging.info(f"Loaded {len(self.data)} images from {datasets_info} datasets")
        
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
                for i in range(len(self.dataset_schema)):
                    try:
                        data_path = scene / self.dataset_schema[i]["field"]

                        files = os.scandir(data_path)
                        break

                    except FileNotFoundError:
                        logging.warning(
                        f"Dataset {dataset_info['name']} scene {scene.name} does not contain the expected data folder: {data_path}. Skipping."
                    )
                    continue
                else:
                    raise FileNotFoundError(
                        f"Dataset {dataset_info['name']} scene {scene.name} does not contain any of the expected data folders: {[entry['field'] for entry in self.dataset_schema]}."
                    )
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

    def _get_data(self, idx: int) -> Tuple[List[Dict], dict]:
        # Get data paths
        data = self.data[idx].decode("utf-8")
        dataset, scene, data_name = data.split("/")
        data_name = data_name.split(".")[0]
        data_info = {"dataset": dataset, "scene": scene, "image_name": data_name}

        paths = []
        for entry_description in self.dataset_schema:
            paths.append({
                "data": self.root_dir / dataset / scene / entry_description["field"] / f"{data_name}",
                "field": entry_description["field"],
                "dtype": entry_description["dtype"],
                "loader": entry_description["loader"],
            })
        return paths, data_info

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        data, data_info = self._get_data(idx)
        
        data_dict = {}
        for entry in data:
            if not entry["data"].parent.exists():
                continue
            entry_out = entry["loader"](entry["data"])
            data_dict[entry["field"]] = entry_out
        return self.format_data(data_dict) if self.format_data is not None else data_dict


class DataToFolder():
    def __init__(self,
                 root_dir: str | Path,
                 dataset_name: str,
                 scene_name: str,
                 save_resolver: Callable | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.scene_name = scene_name
        
        self._count = 0
        
        if save_resolver is None:
            save_resolver = self._save_resolver
        
    def _save_resolver(self,
                       data: Any,
                       path: Path,
                       entry_name: str,
                       ) -> None:
        if isinstance(data, PILImage):
            data.save(path / f"{entry_name}.jpg")
        elif isinstance(data, np.ndarray):
            np.savez_compressed(path / f"{entry_name}.npz", data)
        elif isinstance(data, torch.Tensor):
            torch.save(data, path / f"{entry_name}.pt")
        elif isinstance(data, dict | list):
            with open(path / f"{entry_name}.json", "w") as f:
                json.dump(data, f)
        else:
            print(data)
            raise ValueError(f"Unsupported data type: {type(data)}")
        
    def get_count(self):
        return self._count
            
    def add_entry(self, data_dict):
        scene_dir = self.root_dir / self.dataset_name / self.scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        entry_name = str(int(time.time() * 1e6))
        
        for key, value in data_dict.items():
            # Make a directory for the key
            (scene_dir / key).mkdir(parents=True, exist_ok=True)
                        
            self._save_resolver(value, scene_dir / key, entry_name)
            
        self._count += 1