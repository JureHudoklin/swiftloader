
import io
import logging
import random
from dataclasses import dataclass
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from typing import List, Callable, Any, Tuple
from pathlib import Path

from torch.utils.data import IterableDataset, get_worker_info

import fastparquet as fp

logger = logging.getLogger(__name__)

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)

@dataclass
class WorkerInfo:
    id: int
    num_workers: int


class ParquetDataloader(IterableDataset):
    def __init__(self,
                 root_dir: str | Path,
                 scene: str,
                 batch_size: int,
                 format_data: Callable[[List[dict]], Any] | None = None,
                 drop_last: bool = False,
                 shuffle: bool = True,
                 *args,
                 **kwargs
                 ) -> None:
        super(ParquetDataloader, self).__init__(*args, **kwargs)
        
        self.root_dir = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.batch_size = batch_size
        self.scene = scene
        self.drop_last = drop_last
        self.shuffle = shuffle
        if format_data is None:
            format_data = self._format_data
        self.format_data = format_data

        self.dataset = self._load_dataset(self.root_dir, self.scene)

    def _load_dataset(self, root_dir, scene: str):  
        path = str(root_dir / scene)
        if not Path(path).exists():
            raise FileNotFoundError(f"Directory {path} does not exist.")
        dataset = fp.ParquetFile(path)
        return dataset
    
    def __len__(self):
        return self.dataset.count() // self.batch_size
    
    def __iter__(self):
        num_row_groups = len(self.dataset.row_groups)
        worker_info = get_worker_info()

        # Only divide up batches when using multiple worker processes
        if worker_info != None:
            worker_load = num_row_groups // worker_info.num_workers
            
            # If more workers than batches exist, some won't be used
            if worker_load == 0:
                if worker_info.id < num_row_groups:
                    start = worker_info.id
                    end = worker_info.id + 1
                else: 
                    return
            else:
                start = worker_load * worker_info.id
                end = min(start + worker_load, num_row_groups)
                      
        else: 
            start = 0
            end = num_row_groups

        cache = []
        i = 0
        worker_load = np.arange(start, end)
        worker_load = np.random.permutation(worker_load)
        while True:
            if len(cache) >= self.batch_size:
                data = cache[:self.batch_size]
                cache = cache[self.batch_size:]
                yield self.format_data(data)
                continue

            if i >= (end - start):
                if len(cache) > 0:
                    yield self.format_data(cache)
                break

            batch_i = worker_load[i]
            batch = self.dataset[batch_i]
            batch = batch.to_pandas()
            # Convert to list of dictionaries
            batch = batch.to_dict(orient='records')
            cache.extend(batch)
            if self.shuffle:
                random.shuffle(cache)
            i += 1
            
    def _format_data(self, data: List[dict]) -> List[dict]:
        for entry in data:
            entry['image'] = Image.open(io.BytesIO(entry['image']))
            
        return data