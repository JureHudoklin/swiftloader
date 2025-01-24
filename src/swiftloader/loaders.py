from typing import Any
from pathlib import Path

from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as PILImage
import json
import numpy as np
import torch
import io

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageLoader:
    def __init__(self, parquet: bool = False):
        self.parquet = parquet
        
        self._extensions = [".jpg", ".png"]
        
        self.file_ext_history = ".jpg"
        
    def _parquet_loader(self, data: bytes) -> PILImage:
        with Image.open(io.BytesIO(data)) as img:
            image = img.convert("RGB")
            image = ImageOps.exif_transpose(image)
        return image
    
    def _file_loader(self, data: str | Path) -> PILImage:
        try:
            with Image.open(str(data)+self.file_ext_history) as img:
                image = img.convert("RGB")
                image = ImageOps.exif_transpose(image)
            return image
        
        except FileNotFoundError:
            for ext in self._extensions:
                try:
                    with Image.open(str(data)+ext) as img:
                        image = img.convert("RGB")
                        image = ImageOps.exif_transpose(image)
                    self.file_ext_history = ext
                    return image
                except FileNotFoundError:
                    pass
        
    def __call__(self, data) -> Any:
        if self.parquet:
            return self._parquet_loader(data)
        else:
            return self._file_loader(data)


class JsonLoader:
    def __init__(self, parquet: bool = False):
        self.parquet = parquet
        self._extension = ".json"
    
    def __call__(self, data) -> Any:
        if self.parquet:
            return json.loads(data)
        else:
            with open(str(data)+self._extension, "r") as f:
                return json.load(f)


def image_loader(data, field: str, dtype: str) -> PILImage:
    if dtype == "binary":
        with Image.open(io.BytesIO(data)) as img:
            image = img.convert("RGB")
            image = ImageOps.exif_transpose(image)
        return image
    elif dtype in [".jpg", ".jpeg", ".png"]:
        with Image.open(data) as img:
            image = img.convert("RGB")
            image = ImageOps.exif_transpose(image)
        return image
    else:
        raise ValueError(f"Image loader support the following types: ['binary', '.jpg', '.jpeg', '.png']. Got {dtype} instead.")

def json_loader(data, field: str, dtype: str) -> Any:
    if dtype == "string":
        return json.loads(data)
    elif dtype == ".json":
        with open(data, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"JSON loader support the following types: ['string', '.json']. Got {dtype} instead.")
    
def npy_loader(data: str | Path, field: str, dtype: str) -> Any:
    if dtype == ".npy":
        return np.load(data)
    else:
        raise ValueError(f"Numpy loader support the following types: ['.npy']. Got {dtype} instead.")

def torch_loader(data: str | Path, field: str, dtype: str) -> Any:
    if dtype in [".pt", ".pth"]:
        return torch.load(data)
    else:
        raise ValueError(f"Torch loader support the following types: ['.pt', '.pth']. Got {dtype} instead.")

def identity_loader(data: Any, field: str, dtype: str) -> Any:
    return data