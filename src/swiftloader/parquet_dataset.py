
import io
import logging
import random
import copy
from dataclasses import dataclass
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from typing import List, Callable, Any, Tuple
from pathlib import Path

from torch.utils.data import IterableDataset, get_worker_info

import fastparquet as fp
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

from .util.type_structs import DatasetInfo

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


class ParquetDataset(IterableDataset):
    def __init__(self,
                 root_dir: str | Path,
                 datasets_info: List[DatasetInfo],
                 batch_size: int,
                 format_data: Callable[[List[dict]], Any] | None = None,
                 drop_last: bool = False,
                 shuffle: bool = True,
                 *args,
                 **kwargs
                 ) -> None:
        
        self.root_dir = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.datasets_info = datasets_info
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        if format_data is None:
            format_data = self._format_data
        self.format_data = format_data

        self.datasets = []
        for dataset_info in self.datasets_info:
            name = dataset_info["name"]
            for scene in dataset_info["scenes"]:
                self.datasets.append(self._load_dataset(self.root_dir, name, scene))

    def _load_dataset(self, root_dir, name, scene):
        path = str(root_dir / name  / scene)
        if not Path(path).exists():
            raise FileNotFoundError(f"Directory {path} does not exist.")
        dataset = fp.ParquetFile(path)
        return dataset
    
    def __len__(self):
        total_len = sum([dataset.count() for dataset in self.datasets])
        return total_len // self.batch_size if self.drop_last else -(-total_len // self.batch_size)
    
    def __iter__(self):
        ds_num_row_groups = [len(dataset.row_groups) for dataset in self.datasets]
        worker_info = get_worker_info()

        # Only divide up batches when using multiple worker processe
        worker_load_info = []
        for i, num_row_groups in enumerate(ds_num_row_groups):
            if worker_info != None:
                worker_load = num_row_groups // worker_info.num_workers

                # If more workers than batches exist, some won't be used
                if worker_load == 0:
                    if worker_info.id < num_row_groups:
                        start = worker_info.id
                        end = worker_info.id + 1
                    else: 
                        start = 0
                        end = 0
                else:
                    start = worker_load * worker_info.id
                    end = min(start + worker_load, num_row_groups)

            else: 
                start = 0
                end = num_row_groups
            worker_load_info.append({"dataset": i, "start": start, "end": end, "idx": 0, "load": np.arange(start, end)})

        cache = []
        if self.shuffle:
            for i in range(len(worker_load_info)):
                worker_load_info[i]["load"] = np.random.permutation(worker_load_info[i]["load"])
       
        while True:
            if len(cache) >= self.batch_size:
                data = cache[:self.batch_size]
                cache = cache[self.batch_size:]
                yield self.format_data(data)
                continue

            for wli in worker_load_info:
                if wli["idx"] >= (wli["end"] - wli["start"]):
                    worker_load_info.remove(wli)

            if len(worker_load_info) == 0:
                if len(cache) > 0:
                    yield self.format_data(cache)
                break
            
            if self.shuffle:
                wli = random.choice(worker_load_info)
            else:
                wli = worker_load_info[0]
                
            batch_i = wli["load"][wli["idx"]]
            batch = self.datasets[wli["dataset"]][batch_i]

            batch = batch.to_pandas()
            # Convert to list of dictionaries
            batch = batch.to_dict(orient='records')
            cache.extend(batch)
            if self.shuffle:
                random.shuffle(cache)
            wli["idx"] += 1
            
    def _format_data(self, data: List[dict]) -> List[dict]:
        return data


class DataToParquet():
    def __init__(self,
                 save_dir: str | Path,
                 dataset_name: str,
                 schema: pa.Schema,
                 entry_per_file: int = 10000,
                 ) -> None:
        self.save_dir = Path(save_dir)
        self.dataset_name = dataset_name
        self.schema = schema
        self.entry_per_file = entry_per_file
        
        self.data = []
        
    def add_entry(self, data_dict):
        self.data.append(data_dict)
        
        if len(self.data) >= self.entry_per_file :
            self.save_data()
            
    def save_data(self):
        # Convert the data to a pandas dataframe
        df = pd.DataFrame(self.data[:self.entry_per_file ])
        _ = pa.Table.from_pandas(df)        
        
        # Save the dataframe to parquet
        pq.write_to_dataset(table=pa.Table.from_pandas(df),
                            root_path=str(self.save_dir / self.dataset_name),
                            schema=self.schema,
        )
        
        self.data = self.data[self.entry_per_file :]