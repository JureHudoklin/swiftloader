from collections import defaultdict
from functools import partial
import os
import time
import logging
import json
import ijson
import tempfile
import numpy as np
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as PILImage
from pathlib import Path
from typing import Callable, List, Dict, Optional, Tuple, Literal, Any, Sequence, TypeAlias

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import box_convert, clip_boxes_to_image
import torchvision.transforms.v2.functional as TF
from pycocotools.coco import COCO

from target_utils import Target
from target_utils.formating import target_check_dtype, target_set_dtype, target_reset_tvtensor
from target_utils.util import target_get_boxarea, target_filter

from .util.type_structs import (
    CocoCat,
    DatasetInfo
)
from .util.misc import HiddenPrints

ImageFile.LOAD_TRUNCATED_IMAGES = True

def mapping_function(x, values):
    return values.index(x)

class SwiftClassification(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        datasets_info: List[DatasetInfo],
        base_transforms: Optional[Callable] = None,
        input_transforms: Optional[Callable] = None,
        attributes: Optional[List[str]] = None,
    ) -> None:
        """Dataloader for loading classification data

        Parameters
        ----------
        img_folder : str
            Path to the folder where the images are stored
        ann_file : str
            Path to the annotation file
        """
        self.data_root = Path(data_root)
        self.datasets_info = sorted(datasets_info, key=lambda x: x["name"])
        self.attributes = attributes

        self._check_datasets_exist(self.data_root, self.datasets_info)
        self.images_data, self.cumsum_annotations = self._load_dataset(self.data_root, self.datasets_info)
        self.cat_map, self.cats= self._build_category_map(
            self.data_root, self.datasets_info
        )
        if self.attributes is not None:
            self.attr_map, self.attrs = self._build_attributes_map(self.data_root, self.datasets_info)
        logging.info(f"Loaded {len(self.images_data)} images from {datasets_info} datasets")

        self.base_transforms = base_transforms
        self.input_transforms = input_transforms

        self.fail_save = self.__getitem__(0)

    def _check_datasets_exist(self, data_root: Path, datasets_info: List[DatasetInfo]):
        for dataset_info in datasets_info:
            assert (data_root / dataset_info["name"]).exists(), f"Dataset {dataset_info['name']} does not exist"

    def _load_dataset(self, data_root: Path, datasets_info: List[DatasetInfo]):
        images_ = []
        for dataset_info in datasets_info:
            # Get all subfolders in the dataset folder (exclude files)
            scenes = sorted((data_root / dataset_info["name"]).glob("*"))
            scenes = [scene for scene in scenes if scene.is_dir()]
            if "scenes" in dataset_info:
                scenes = [scene for scene in scenes if scene.name in dataset_info["scenes"]]

            for scene in scenes:
                image_annotations_path = scene / "annotations"
                files = os.scandir(image_annotations_path)
                    
                images_.extend(
                    [
                        f"{dataset_info['name']}/{scene.name}/{file.name}"
                        for file in files
                        if file.is_file()
                    ]
                )
                
        image_numann = []
        for file in images_:
            f_path = Path(f"{file.split('/')[0]}/{file.split('/')[1]}/annotations/{file.split('/')[-1]}")
            ann_path = data_root / f_path
            with open(ann_path) as f:
                annotations = json.load(f)
                image_numann.append(len(annotations))
                
        images = np.array(images_).astype(np.string_)
        # Sort images by name
        sort_idx = np.argsort(images)
        images = images[sort_idx]
        image_numann = np.array(image_numann)[sort_idx]
        image_cumsum = np.cumsum(image_numann)

        return images, image_cumsum

    def _build_category_map(self, data_root: Path, datasets_info: List[DatasetInfo]
                            ) -> Tuple[Dict[str, Dict[int, int]], Dict[int, CocoCat]]:
        cat_map = defaultdict(dict)
        cats = {}

        dataset_cats = {}
        for dataset_info in datasets_info:
            with open(data_root / dataset_info["name"] / "categories.json") as f:
                categories = json.load(f)
                dataset_cats[dataset_info["name"]] = categories

        new_cat_id = 1
        for dataset, categories in dataset_cats.items():
            for cat in categories:
                cat_map[dataset][cat["id"]] = new_cat_id
                cat["id"] = new_cat_id
                cats[new_cat_id] = cat
                new_cat_id += 1

        return cat_map, cats

    def _build_attributes_map(self, data_root: Path, datasets_info: List[DatasetInfo]):
        attr_map = {} # {"ATTRIBUTE": {"value": mapping_function}"}}
        attrs = defaultdict(dict) # {"ATTRIBUTE": {"type": "type", "values": ["value1", "value2"]}}

        dataset_cats = {}
        for dataset_info in datasets_info:
            with open(data_root / dataset_info["name"] / "categories.json") as f:
                categories = json.load(f)
                dataset_cats[dataset_info["name"]] = categories

        # Collect all attributes
        cat_all = [cat for ds_cats in dataset_cats.values() for cat in ds_cats]
        attrs_list = [cat.get("attributes", []) for cat in cat_all]

        for ds_attrs in attrs_list:
            for attr_name, attr in ds_attrs.items():
                attr_type = attr["type"]
                attr_values = attr["values"]
                if attr_name not in attrs:
                    attrs[attr_name]["type"] = attr_type
                    attrs[attr_name]["values"] = attr_values
                else:
                    if attrs[attr_name]["type"] != attr_type:
                        raise ValueError(f"Attribute {attr_name} has different types in different datasets")
                    if attr_type == "str":
                        attrs[attr_name]["values"] = sorted(list(set(attrs[attr_name]["values"] + attr_values)))

        # Create attribute map
        for attr_name, attr in attrs.items():
            if attr["type"] == "str":
                attr_map[attr_name] = partial(mapping_function, values=attr["values"])
            else:
                attr_map[attr_name] = lambda x: x

        return attr_map, attrs

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

    def _get_target(self, image: PILImage, ann, img_ann, ann_idx):
        w, h = image.size
        size = torch.tensor([int(h), int(w)])
        image_id = torch.tensor(img_ann["id"])

        boxes = [obj["bbox"] for obj in ann]  # xywh
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        boxes = box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")
        boxes = tv_tensors.BoundingBoxes(boxes, canvas_size=(h, w), format=tv_tensors.BoundingBoxFormat.XYXY)  # type: ignore[call-overload]

        labels = torch.tensor([obj["category_id"] for obj in ann])
        
        # attributes
        if self.attributes is not None:
            attributes = {}
            for prop in self.attributes:
                for obj in ann:
                    func = self.attr_map[prop]
                    attributes[prop] = func(obj["attributes"][prop])
            
            attributes = {prop: torch.tensor([self.attr_map[prop](obj["attributes"][prop]) for obj in ann]) \
                        for prop in self.attributes}
        else:
            attributes = None

        target = Target(
            image_id=image_id,
            size=size,
            orig_size=size,
            boxes=boxes,
            labels=labels,
            box_format=torch.tensor(0),
            attributes=attributes,
        )
        keep = torch.zeros_like(target["labels"], dtype=torch.bool)
        keep[ann_idx] = True
        target = target_filter(target, keep)
            
        # Crop image to bounding box
        box_noise = torch.randint(-10, 30, (4,)).reshape(1, 4).float()
        #box_noise[0, :2] = box_noise[0, :2] * (-1)
        box = target["boxes"]
        box = box + box_noise
        # box = box + box_noise
        box = clip_boxes_to_image(box, size)
        box_xywh = box_convert(box, in_fmt="xyxy", out_fmt="xywh")[0].tolist()
        bw, bh = box_xywh[2], box_xywh[3]
        if (target["size"] != torch.tensor([bh, bw])).all():
            image = image.crop((box_xywh[0], box_xywh[1], box_xywh[0] + bw, box_xywh[1] + bh))
        target["orig_size"] = torch.tensor([bh, bw])
        target["size"] = torch.tensor([bh, bw])
        target["boxes"] = target["boxes"] - torch.tensor([0, 0, bh, bw]) # type: ignore[index]

        return target_reset_tvtensor(target), image

    def __len__(self):
        return self.cumsum_annotations[-1]

    def __getitem__(self, idx: int) -> Tuple[Any, Target]:
        img_idx = np.searchsorted(self.cumsum_annotations, idx, side="right")
        ann_idx = idx - self.cumsum_annotations[img_idx - 1] if img_idx > 0 else idx
        
        ann_path, img_ann_path, image_data = self._get_data_paths(img_idx) # type: ignore[index]

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
            
        target, image = self._get_target(image, ann, img_ann, ann_idx)

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

    def get_attributes(self) -> List:
        return list(self.attrs.values()) if self.attributes is not None else []
    
    def get_dataset_mean_std(self):
        mean, std = [], []
        for idx in range(len(self)):
            img_idx = np.searchsorted(self.cumsum_annotations, idx, side="right")
            ann_path, img_ann_path, image_data = self._get_data_paths(img_idx)
            with open(img_ann_path) as f:
                img_ann = json.load(f)
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
            image = TF.to_tensor(image)
            
            image_mean = image.mean(dim=(1,2))
            image_std = image.std(dim=(1,2))
            
            mean.append(image_mean)
            std.append(image_std)
            
        mean = torch.stack(mean).mean(dim=0)
        std = torch.stack(std).mean(dim=0)
        
        return mean, std

    def get_dataset_api(self, valid_categories: List[Dict] | None = None) -> COCO:
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

        return coco


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
