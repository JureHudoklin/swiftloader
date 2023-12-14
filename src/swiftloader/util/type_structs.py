from torch import Tensor
from torchvision import tv_tensors
from typing import List, Dict, TypedDict, Literal, TypeAlias

from torchvision.tv_tensors import BoundingBoxFormat

DatasetInfo: TypeAlias = Dict[Literal["name", "scenes"], str]

class CocoAnn(TypedDict):
    image_id: int
    category_id: int
    bbox: List[float] # x, y, w, h
    area: float
    segmentation: List[List[float]] | None
    iscrowd: int
    id: int
    
class CocoCat(TypedDict):
    id: int
    name: str
    supercategory: str
    isthing: int | None
    color: List[int] | None
    
class CocoImage(TypedDict):
    id: int
    width: int
    height: int
    file_name: str
    license: int
    flickr_url: str
    coco_url: str
    date_captured: str
    
class DdpConfig(TypedDict):
    world_size: int
    rank: int
    ddp_enabled: bool