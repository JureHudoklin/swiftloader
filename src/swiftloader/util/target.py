import torch

from .type_structs import Target
from torch import Tensor
from torchvision import tv_tensors
from torchvision.tv_tensors import BoundingBoxFormat


def target_reset_tvtensor(target: Target) -> Target:
    if isinstance(target['boxes'], tv_tensors.BoundingBoxes):
        target["size"] = torch.tensor(target["boxes"].canvas_size)
        target["box_format"] = target["boxes"].format.value
    elif isinstance(target["boxes"], Tensor):
        match target["box_format"]:
            case "XYXY":
                fmt = BoundingBoxFormat.XYXY
            case "XYWH":
                fmt = BoundingBoxFormat.XYWH
            case "CXCYWH":
                fmt = BoundingBoxFormat.CXCYWH
                
        canvas_size = (target["size"][0].item(), target["size"][1].item())
        
        target["boxes"] = tv_tensors.wrap(target["boxes"], format=fmt, canvas_size=canvas_size) # type: ignore[call-overload]
        
    return target
        
def target_recalculate_area(target: Target) -> Target:
    target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
    return target

def target_set_dtype(target: Target) -> Target:
    target["boxes"] = target["boxes"].to(torch.float32)
    target["labels"] = target["labels"].to(torch.int64)
    target["iscrowd"] = target["iscrowd"].to(torch.int64)
    target["area"] = target["area"].to(torch.float32)
    
    return target_reset_tvtensor(target)

def target_filter(target: Target, keep: Tensor) -> Target:
    return target_reset_tvtensor({
        "image_id": target["image_id"],
        "boxes": target["boxes"][keep],
        "labels": target["labels"][keep],
        "iscrowd": target["iscrowd"][keep],
        "area": target["area"][keep],
        "orig_size": target["orig_size"],
        "size": target["size"],
        "box_format": target["box_format"]
    })
    
def target_normalize(target: Target) -> Target:
    target = target_reset_tvtensor(target)
    if target["boxes"].shape[0] == 0:
        return target
    
    h, w = target["size"]
    target["boxes"] = target["boxes"] / torch.tensor([w, h, w, h], dtype=torch.float32)

    return target_reset_tvtensor(target)

def target_check_dtype(target: Target) -> bool:
    if target["boxes"].dtype != torch.float32:
        return False
    if target["labels"].dtype != torch.int64:
        return False
    if target["iscrowd"].dtype != torch.int64:
        return False
    if target["area"].dtype != torch.float32:
        return False
    return True