import signal
import logging
import torch
import torchvision

from torch import Tensor
from typing import List, Optional, Tuple, Any

def init_worker():
    """
    Catch Ctrl+C signal to termiante workers
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    
class NestedTensorBatch:
    def __init__(self,
                 batch,
                 size_constant: tuple[int, int] | None = None,
                 size_devisable_by: int | None = None):
        zipped_batch = list(zip(*batch))
        self.samples = nested_tensor_from_tensor_list(zipped_batch[0],
                                                      size_constant=size_constant,
                                                      size_devisable_by=size_devisable_by)
        self.targets = zipped_batch[1]
        
    def pin_memory(self):
        self.samples = self.samples.pin_memory()
        self.targets = self.targets
        return self 
    
def create_nested_tensor_batch(x, **args) -> NestedTensorBatch:
    return NestedTensorBatch(x, **args)

class NestedTensor(object):
    def __init__(self, tensors: Tensor, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)
    
    def cuda(self, non_blocking=False):
        cast_tensor = self.tensors.cuda(non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.cuda(non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)
    
    def pin_memory(self):
        self.tensors = self.tensors.pin_memory()
        if self.mask is not None:
            self.mask = self.mask.pin_memory()
        return self

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    
    def __len__(self):
        return len(self.tensors)
    
    @property
    def device(self):
        return self.tensors.device
    
    @property
    def shape(self):
        return self.tensors.shape
    
# def _max_by_axis(the_list: List[List[int]]) -> List[int]:
#     maxes = the_list[0]
#     for sublist in the_list[1:]:
#         for index, item in enumerate(sublist):
#             maxes[index] = max(maxes[index], item)
#     return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor],
                                   size_constant: tuple[int, int] | None = None, # H, W
                                   size_devisable_by: int | None = None
                                   ) -> NestedTensor:
    
    # Check that either constant_size or size_devisable_by is set or both are None
    assert size_constant is None or size_devisable_by is None, "Only one of constant_size and size_devisable_by can be set"
    
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        max_size = torch.stack([torch.tensor(img.shape) for img in tensor_list]).max(0)[0]
        
        # Make all images to the constant size
        if size_constant is not None:
            if not all(size_constant[i] >= max_size[-i] for i in range(2)):
                raise ValueError("size_constant must be larger than the largest image in the list when creating a NestedTensor.")
            max_size[1:] = torch.tensor(size_constant)
        
        # Pad all images to be devisable by size_devisable_by
        if size_devisable_by is not None:
            h, w = max_size[-2:]
            h = int((h + (size_devisable_by-1)) // size_devisable_by) * size_devisable_by
            w = int((w + (size_devisable_by-1))// size_devisable_by) * size_devisable_by
            max_size[1:] = torch.tensor([h, w])
        
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list), *max_size]
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)

# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused # type: ignore[no-redef]
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor],
                                         size_constant: tuple[int, int] | None = None, # H, W
                                         size_devisable_by: int | None = None
                                         ) -> NestedTensor:
    # max_size = []
    # for i in range(tensor_list[0].dim()):
    #     max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64) # type: ignore
    #     max_size.append(max_size_i)
    # max_size = tuple(max_size)
    max_size = torch.stack([torch.tensor(img.shape) for img in tensor_list]).max(0)[0]
    
    # Make all images to the constant size
    if size_constant is not None:
        if not all(size_constant[i] >= max_size[-i] for i in range(2)):
            raise ValueError("size_constant must be larger than the largest image in the list when creating a NestedTensor.")
        max_size[1:] = torch.tensor(size_constant)
    
    # Pad all images to be devisable by size_devisable_by
    if size_devisable_by is not None:
        h, w = max_size[-2:]
        h = int((h + (size_devisable_by-1)) // size_devisable_by) * size_devisable_by
        w = int((w + (size_devisable_by-1))// size_devisable_by) * size_devisable_by
        max_size[1:] = torch.tensor([h, w])

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2).item() for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0])) # type: ignore
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1) # type: ignore
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)