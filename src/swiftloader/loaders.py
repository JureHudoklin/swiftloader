from typing import Any, Literal
from pathlib import Path

from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as PILImage
import cv2
import json
import numpy as np
import torch
import torchvision
import io

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageLoader:
    def __init__(self, 
                 out_type: Literal["pil", "numpy", "torch"] = "pil",
                 parquet: bool = False):
        self.parquet = parquet
        self.out_type = out_type.lower()
        
        self._extensions = [".jpg", ".png"]
        
        self.file_ext_history = ".jpg"
        
    def _parquet_loader(self, data: bytes) -> PILImage:
        with Image.open(io.BytesIO(data)) as img:
            image = img.convert("RGB")
            ImageOps.exif_transpose(image, in_place=True)
        return image
    
    def _file_loader(self, data: str | Path) -> PILImage:
        # Check if the path exists with the current file extension
        # If not, try with the other extensions
        if isinstance(data, str):
            data = Path(data)
            
        ext = self.file_ext_history
        path = data.with_suffix(ext)
        
        if not data.exists():
            # If the file does not exist, try with the other extensions
            for ext in self._extensions:
                path = data.with_suffix(ext)
                if path.exists():
                    break
            else:
                raise FileNotFoundError(f"Image file not found: {data}")
            
        # Try to open the image with the current file extension
        return self._from_type(path)
        
    def _from_type(self, path: str | Path) -> Any:
        if self.out_type == "numpy":
            # Use cv2 to open the image
            return cv2.imread(str(path), cv2.IMREAD_COLOR_RGB)
        elif self.out_type == "torch":
            return torchvision.io.decode_image(str(path), mode = torchvision.io.ImageReadMode.RGB)
        elif self.out_type == "pil":
            with Image.open(path) as img:
                image = img.convert("RGB")
                ImageOps.exif_transpose(image, in_place=True)
            return image
        else:
            raise ValueError(f"Unsupported output type: {self.out_type}. Supported types are: ['numpy', 'torch', 'pil']")
        
    def __call__(self, data) -> Any:
        if self.parquet:
            raise NotImplementedError("Parquet loader is not implemented yet.")
            image = self._parquet_loader(data)
        else:
            image = self._file_loader(data)


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


class NumpyLoader():
    def __init__(self,
                 parquet: bool = False,
                 extension: Literal[".npy", ".npz"] = ".npz",
                 ):
        self.parquet = parquet
        self._extension = extension
        
    def __call__(self, data) -> Any:
        if self.parquet:
            return np.load(data)
        else:
            if self._extension == ".npz":
                data = np.load(str(data)+self._extension)
                arr = data["arr_0"]
                data.close()
                return arr
            else:
                return np.load(str(data)+self._extension)


