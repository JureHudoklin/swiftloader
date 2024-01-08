import os
import random
import logging
import json
import tempfile
import numpy as np
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as PILImage
from pathlib import Path
from typing import Callable, List, Dict, Optional, Tuple, Literal, Any, Iterable
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
from .util.misc import HiddenPrints, get_bbox_from_mask

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SwiftTemplateObjectDetection(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        datasets_info: List[DatasetInfo],
        num_templates: int = 3,
        ignore_templates: bool = False,
        base_transforms: Optional[Callable] = None,
        template_transforms: Optional[Callable] = None,
        input_transforms: Optional[Callable] = None,
        input_template_transforms: Optional[Callable] = None,
        attributes: Optional[List[str]] = None,
        filter_by_attribute: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Dataloader for COCO dataset

        Parameters
        ----------
        img_folder : str
            Path to the folder where the images are stored
        ann_file : str
            Path to the annotation file
        """
        self.data_root = Path(data_root)
        self.datasets_info = sorted(datasets_info, key=lambda x: x["name"])
        self.num_templates = num_templates
        self.ignore_templates = ignore_templates

        self._check_datasets_exist(self.data_root, self.datasets_info)
        self.images_data = self._load_dataset(self.data_root, self.datasets_info)
        self.cat_map, self.cats = self._build_category_map(
            self.data_root, self.datasets_info
        )
        logging.info(f"Loaded {len(self.images_data)} images from {datasets_info} datasets")
        
        if not self.ignore_templates:
            self._check_templates_exist(self.data_root, self.cats.values())

        self.base_transforms = base_transforms
        self.input_transforms = input_transforms
        self.template_transforms = template_transforms
        self.input_template_transforms = input_template_transforms
        self.attributes = attributes
        self.filter_by_attribute = filter_by_attribute

        self.fail_save = self.__getitem__(0)

    def _check_datasets_exist(self, data_root: Path, datasets_info: List[DatasetInfo]):
        for dataset_info in datasets_info:
            assert (data_root / dataset_info["name"]).exists(), f"Dataset {dataset_info['name']} does not exist"

    def _check_templates_exist(self, data_root: Path, categories: Iterable[CocoCat]):
        for cat in categories:
            template_path = data_root / "templates" / f"{cat['name']}" / "rgb"
            template_images = list(template_path.glob("*.png")) + list(template_path.glob("*.jpg"))
            if len(template_images) == 0:
                raise FileNotFoundError(f"Template {template_path} does not exist")

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
        
        if self.filter_by_attribute is not None:
            for prop, value in self.filter_by_attribute.items():
                target_prop = target["attributes"][prop] # type: ignore[index]
                keep = target_prop == value
                target = target_filter(target, keep)

        return target

    def _get_template(self, cat_id: int, target: Target) -> Tuple[List[PILImage], Target]:
        cat = self.cats[cat_id]
        
        # Add template attributes to target
        labels = target["labels"]
        sim_labels = torch.zeros_like(labels, dtype=torch.long)
        sim_labels[labels == cat_id] = 1
        
        if target.get("attributes") is None:
            target["attributes"] = {}
            
        target["attributes"]["sim_labels"] = sim_labels # type: ignore[index]
        if self.ignore_templates:
            target["attributes"]["sim_labels"][:] = 1
            return None, target
        
        # Get all template image files for the category
        template_root = self.data_root / "templates" / f"{cat['name']}"
        template_path = self.data_root / "templates" / f"{cat['name']}" / "rgb"
        mask_path = self.data_root / "templates" / f"{cat['name']}" / "mask"
        
        template_images_paths = list(template_path.glob("*.png")) + list(template_path.glob("*.jpg"))
        template_images_paths = random.sample(template_images_paths, self.num_templates)
        
        template_masks_paths = list(mask_path.glob("*.png"))
        
        # Load templates
        templates = []
        for template_image_path in template_images_paths:
            with Image.open(template_image_path) as template:
                template = template.convert("RGB")
            # Crop to template mask and apply it if available
            image_name = template_image_path.name.split(".")[0]
            template_mask_path = mask_path / f"{image_name}.png"
            if template_mask_path.exists():
                with Image.open(template_mask_path) as mask:
                    mask = mask.convert("L")
                template = Image.composite(template, Image.new("RGB", template.size, color=(120, 120, 120)), mask)
                # Convert mask to numpy array
                mask = torch.as_tensor(np.array(mask))
                bbox = get_bbox_from_mask(mask)
                # Crop template to mask
                template = template.crop(bbox.tolist())
                
            if self.template_transforms is not None:
                template = self.template_transforms(template)
            if self.input_template_transforms is not None:
                template = self.input_template_transforms(template)
            templates.append(template)
            
        return templates, target

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Target]:
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
        
        # Load templates
        if len(target["labels"]) == 0:
            cat_id = random.choice(list(self.cats.keys()))
        else:
            cat_id = random.choice(target["labels"]).item()
        templates, target = self._get_template(cat_id, target) # type: ignore[assignment]
        if templates is not None:
            templates = torch.stack(templates)

        if self.base_transforms is not None:
            image, target = self.base_transforms(image, target)

        if self.input_transforms is not None:
            image, target = self.input_transforms(image, target)

        target_set_dtype(target)
        return image, templates, target

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


