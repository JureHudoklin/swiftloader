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
from .util.misc import save_resolver

logger = logging.getLogger(__name__)

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
        data_info = {"dataset": dataset, "scene": scene, "entry_name": data_name}

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

    def __getitem__(self, idx: int) -> dict:
        data, data_info = self._get_data(idx)
        
        data_dict = {
            "dataset_idx": idx,
        }
        for entry in data:
            if not entry["data"].parent.exists():
                continue
            entry_out = entry["loader"](entry["data"])
            data_dict[entry["field"]] = entry_out
        return self.format_data(data_dict) if self.format_data is not None else data_dict
    
    def modify_entry(self,
                     idx: int,
                     data_dict: dict,
                     save_resolver_fn: Callable | None = None,
                     ) -> None:
        """Modify an entry in the dataset.

        Parameters
        ----------
        idx : int
            The index of the entry to modify.
        data_dict : dict
            A dictionary containing the data to modify the entry with.
        """
        paths, data_info = self._get_data(idx)
        existing_fields = {entry["field"] for entry in paths}
        
        if save_resolver_fn is None:
            save_resolver_fn = save_resolver
        
        for field_key, field_value in data_dict.items():
            # check if the folders exist
            path=self.root_dir / data_info["dataset"] / data_info["scene"] / field_key
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            
            if field_key in existing_fields:
                # Modify existing entry
                for entry in paths:
                    if entry["field"] == field_key:
                        logger.debug(f"Modifying entry {data_info['entry_name']} with value {field_value}.")
                        save_resolver_fn(
                            data=field_value,
                            path=entry["data"].parent,
                            entry_name=data_info["entry_name"],
                        )
                        break
                    
            else:
                save_resolver_fn(
                    data=field_value,
                    path=self.root_dir / data_info["dataset"] / data_info["scene"] / field_key,
                    entry_name=data_info["entry_name"],
                )
                
    def split(
        self,
        split_ratio: float = 0.8,
        shuffle: bool = True,
    ) -> Tuple["FolderDataset", "FolderDataset"]:
        """Split the dataset into two parts.

        Parameters
        ----------
        split_ratio : float, optional
            The ratio of the first part of the split, by default 0.8.
        shuffle : bool, optional
            Whether to shuffle the dataset before splitting, by default True.

        Returns
        -------
        Tuple[FolderDataset, FolderDataset]
            The two parts of the split dataset.
        """
        if shuffle:
            np.random.shuffle(self.data)
        
        split_idx = int(len(self.data) * split_ratio)
        data1 = self.data[:split_idx]
        data2 = self.data[split_idx:]
        
        dataset1 = FolderDataset(
            root_dir=self.root_dir,
            datasets_info=self.datasets_info,
            dataset_schema=self.dataset_schema,
            format_data=self.format_data,
        )
        dataset1.data = data1
        
        dataset2 = FolderDataset(
            root_dir=self.root_dir,
            datasets_info=self.datasets_info,
            dataset_schema=self.dataset_schema,
            format_data=self.format_data,
        )
        dataset2.data = data2
        
        return dataset1, dataset2


class DataToFolder():
    def __init__(self,
                 root_dir: str | Path,
                 dataset_name: str,
                 scene_name: str,
                 save_resolver_fn: Callable | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.scene_name = scene_name
        
        self._count = 0
        
        if save_resolver_fn is None:
            save_resolver_fn = save_resolver 
            
        self.save_resolver_fn = save_resolver_fn
        
    def get_count(self):
        return self._count
            
    def add_entry(self, data_dict, entry_name: Optional[str] = None):
        scene_dir = self.root_dir / self.dataset_name / self.scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        if entry_name is None:
            entry_name = str(int(time.time() * 1e6))
        
        for key, value in data_dict.items():
            # Make a directory for the key
            (scene_dir / key).mkdir(parents=True, exist_ok=True)
                        
            self.save_resolver_fn(value, scene_dir / key, entry_name)
            
        self._count += 1