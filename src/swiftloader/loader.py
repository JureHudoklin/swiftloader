import torch
from torch import nn, Tensor
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    DistributedSampler,
    Dataset,
)

from typing import List, Tuple, Literal, Optional, Callable, Any

from .util.misc import set_worker_sharing_strategy
from .util.type_structs import DdpConfig, Target

def get_swift_loader(
    dataset: Dataset,
    split: Literal["train", "val", "test"],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    collate_fn: Callable[[List[Tuple[Any, Target]]], Any],
    ddp_config: DdpConfig | None = None,
) -> DataLoader:  
    if ddp_config:
        if split == "train":
            sampler = DistributedSampler(
                dataset,
                num_replicas=ddp_config["world_size"],
                rank=ddp_config["rank"],
                shuffle=True,
                drop_last=False,
            )
        elif split == "val" or split == "test":
            sampler = DistributedSampler(
                dataset,
                num_replicas=ddp_config["world_size"],
                rank=ddp_config["rank"],
                shuffle=False,
                drop_last=False,
            )
    else:
        if split == "train":
            sampler = RandomSampler(dataset) # type: ignore[assignment]
        elif split == "val" or split == "test":
            sampler = SequentialSampler(dataset) # type: ignore[assignment]

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        worker_init_fn=set_worker_sharing_strategy,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=False,
    )

    return data_loader

