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
from torchvision import tv_tensors
from torchvision.ops import box_convert
from pycocotools.coco import COCO

from target_utils import Target
from target_utils.util import target_get_boxarea, target_filter
from target_utils.formating import target_set_dtype

from .util.type_structs import (
    CocoCat,
    DatasetInfo
)
from .util.misc import HiddenPrints

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SwiftObjectDetection(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        datasets_info: List[DatasetInfo],
        base_transforms: Optional[Callable] = None,
        input_transforms: Optional[Callable] = None,
        attributes: Optional[List[str]] = None,
        filter_by_property: Optional[Dict[str, Any]] = None,
        classless: bool = True,
    ) -> None:
        """Dataloader for COCO dataset

        Parameters
        ----------
        img_folder : str
            Path to the folder where the images are stored
        ann_file : str
            Path to the annotation file
        """
        self.classless = classless
        self.data_root = Path(data_root)
        self.datasets_info = sorted(datasets_info, key=lambda x: x["name"])

        self._check_datasets_exist(self.data_root, self.datasets_info)
        self.images_data = self._load_dataset(self.data_root, self.datasets_info)
        self.cat_map, self.cats = self._build_category_map(
            self.data_root, self.datasets_info
        )
        logging.info(f"Loaded {len(self.images_data)} images from {datasets_info} datasets")

        self.base_transforms = base_transforms
        self.input_transforms = input_transforms
        self.attributes = attributes
        self.filter_by_property = filter_by_property

        self.fail_save = self.__getitem__(0)

    def _check_datasets_exist(self, data_root: Path, datasets_info: List[DatasetInfo]):
        for dataset_info in datasets_info:
            assert (data_root / dataset_info["name"]).exists(), f"Dataset {dataset_info['name']} does not exist"

    def _load_dataset(self, data_root: Path, datasets_info: List[DatasetInfo]) -> np.ndarray:
        images_ = []
        for dataset_info in datasets_info:
            # Get all subfolders in the dataset folder (exclude files)
            scenes = sorted((data_root / dataset_info["name"]).glob("*"))
            scenes = [scene for scene in scenes if scene.is_dir()]
            if "scenes" in dataset_info:
                scenes = [scene for scene in scenes if scene.name in dataset_info["scenes"]]
            
            for scene in scenes:
                image_annotations_path = scene / "image_annotations"
                files = os.scandir(image_annotations_path)
                images_.extend(
                    [
                        f"{dataset_info['name']}/{scene.name}/{file.name}"
                        for file in files
                        if file.is_file()
                    ]
                )

        if len(images_) < 1000000:
            images_.sort()
        else:
            logging.warning(
                "Sorting images will take a long time. Skipping sorting. Images will be loaded in random order."
            )

        images = np.array(images_).astype(np.string_)
        return images

    def _build_category_map(
        self, data_root: Path, datasets_info: List[DatasetInfo]
    ) -> Tuple[Dict[str, Dict[int, int]], Dict[int, CocoCat]]:
        cat_map = defaultdict(dict)
        cats = {}
        attr_map = {} # {"ATTRIBUTE": {"value": "mapped_value"}"}}
        attrs = {} # {"ATTRIBUTE": {"type": "type", "values": ["value1", "value2"]}}
        
        dataset_cats = {}
        for dataset_info in datasets_info:
            with open(data_root / dataset_info["name"] / "categories.json") as f:
                categories = json.load(f)
                dataset_cats[dataset_info["name"]] = categories
        new_cat_id = 0

        for dataset, categories in dataset_cats.items():
            for cat in categories:
                if self.classless:
                    cat_map[dataset][cat["id"]] = 1
                    cat["id"] = 1
                    cat["name"] = "object"
                    cat["supercategory"] = "object"
                    cats[1] = cat
                else:
                    new_cat_id += 1
                    cat_map[dataset][cat["id"]] = new_cat_id
                    cat["id"] = new_cat_id
                    cats[new_cat_id] = cat

        return cat_map, cats

    def _get_data_paths(self, idx: int) -> Tuple[Path, Path, Dict]:
        # Get data paths
        image_data = self.images_data[idx].decode("utf-8")
        dataset, scene, image_name = image_data.split("/")
        image_name = image_name.split(".")[0]
        image_data = {"dataset": dataset, "scene": scene, "image_name": image_name}

        img_ann_path = (
            self.data_root
            / dataset
            / scene
            / "image_annotations"
            / f"{image_name}.json"
        )
        ann_path = (
            self.data_root / dataset / scene / "annotations" / f"{image_name}.json"
        )

        return ann_path, img_ann_path, image_data

    def _get_target(self, image: PILImage, ann, img_ann) -> Target:
        w, h = image.size
        size = torch.tensor([int(h), int(w)])
        image_id = torch.tensor(img_ann["id"])

        # attributes
        if self.attributes is not None:
            attributes = {prop: torch.tensor([obj["attributes"][prop] for obj in ann]) \
                        for prop in self.attributes}
        else:
            attributes = None

        boxes = [obj["bbox"] for obj in ann]  # xywh
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        boxes = box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")
        boxes = tv_tensors.BoundingBoxes(boxes, canvas_size=(h, w), format=tv_tensors.BoundingBoxFormat.XYXY)  # type: ignore[call-overload]

        labels = torch.tensor([obj["category_id"] for obj in ann])

        target = Target(
            image_id=image_id,
            boxes=boxes,
            labels=labels,
            orig_size=size,
            size=size,
            box_format=torch.tensor(0),
            attributes=attributes,
        )
        
        if self.filter_by_property is not None:
            for prop, value in self.filter_by_property.items():
                target_prop = target["attributes"][prop] # type: ignore[index]
                keep = target_prop == value
                target = target_filter(target, keep)

        return target

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx: int) -> Tuple[Any, Target]:
        ann_path, img_ann_path, image_data = self._get_data_paths(idx)

        # Load image and annotations
        with open(img_ann_path) as f:
            img_ann = json.load(f)
        with open(ann_path) as f:
            ann = json.load(f)
        image_path = self.data_root / image_data["dataset"] / image_data["scene"] / "images" / img_ann["file_name"]
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                image = ImageOps.exif_transpose(image)
        except Exception as e:
            logging.warning(
                f"Failed to load image: img_ann_path={img_ann_path},\n idx={idx}) \n Error: {e}"
            )
            return self.fail_save
            
        target = self._get_target(image, ann, img_ann)
        target["image_id"] = torch.tensor([idx])

        # Remap category ids
        new_labels = torch.zeros_like(target["labels"])
        for i, label in enumerate(target["labels"]):
            new_labels[i] = self.cat_map[image_data["dataset"]][label.item()]  # type: ignore[index]
        target["labels"] = new_labels

        if self.base_transforms is not None:
            image, target = self.base_transforms(image, target)

        if self.input_transforms is not None:
            image, target = self.input_transforms(image, target)

        target_set_dtype(target)
        return image, target

    def get_categories(self) -> List:
        return list(self.cats.values())

    def get_dataset_api(self, valid_categories: List[Dict] | None = None) -> Tuple[COCO, Dict]:
        images, annotations, categories = [], [], []

        ann_id = 0
        for idx in range(len(self)):
            ann_path, img_ann_path, image_data = self._get_data_paths(idx)

            with open(img_ann_path) as f:
                img_ann = json.load(f)
                img_ann["id"] = idx
                images.append(img_ann)

            with open(ann_path) as f:
                ann = json.load(f)
                for obj in ann:
                    obj["image_id"] = idx
                    obj["id"] = ann_id
                    obj["category_id"] = self.cat_map[image_data["dataset"]][obj["category_id"]]  # type: ignore[index]
                    ann_id += 1
                    annotations.append(obj)

        categories = self.get_categories()
        if valid_categories is not None:
            categories = valid_categories

        info = {
            "description": f"Datasets: {self.datasets_info}",
            "data_root": str(self.data_root),
        }

        dataset = {
            "info": info,
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            with HiddenPrints():
                json.dump(dataset, f)
                f.flush()
                coco = COCO(f.name)

        return coco, dataset


# if __name__ == "__main__":

#     base_transforms = T.Resize((512, 512))

#     dataloader = DatasetLoader(
#         "/home/jure/datasets/OBJECTS_DATASET",
#         ["AHIL", "ICBIN"],
#         "val",
#         input_transforms=None,
#         base_transforms=base_transforms,
#         keep_crowded=True,
#     )
#     dataloader.get_dataset_api(None)

# img, target = dataloader[0]

# plt.imshow(img)
# plt.savefig("test.png")
# print(target)
