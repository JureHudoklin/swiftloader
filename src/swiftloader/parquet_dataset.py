
import io
import logging
import random
import copy
from dataclasses import dataclass
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from typing import List, Callable, Any, Tuple, Dict, Literal
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
                 dataset_schema: List[Dict[Literal["field", "dtype", "loader"], Any]],
                 batch_size: int,
                 format_data: Callable[[List[dict]], Any] | None = None,
                 drop_last: bool = False,
                 shuffle: bool = True,
                 *args,
                 **kwargs
                 ) -> None:
        
        self.root_dir = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.datasets_info = datasets_info
        self.dataset_schema = dataset_schema
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
    
    def batch_format_data(self, data: List[dict]) -> List[dict]:
        data_out = map(lambda entry: self.format_data(entry), data)
        return list(data_out)
        
    def _format_data(self, data: dict) -> dict:
        return data
    
    def _load_data(self, data):
        data = copy.deepcopy(data)
        def format_entry(entry):
            for schema in self.dataset_schema:
                if entry.get(schema["field"]) is not None:
                    entry[schema["field"]] = schema["loader"](entry[schema["field"]], schema["field"], schema["dtype"])
                else:
                    continue
            return entry
        
        data = map(lambda entry: format_entry(entry), data)
        return data
    
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
                yield self.batch_format_data(self._load_data(data))
                continue

            for wli in worker_load_info:
                if wli["idx"] >= (wli["end"] - wli["start"]):
                    worker_load_info.remove(wli)

            if len(worker_load_info) == 0:
                if len(cache) > 0:
                    yield self.batch_format_data(self._load_data(data))
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



class DataToParquet():
    def __init__(self,
                 root_dir: str | Path,
                 dataset_info: DatasetInfo,
                 schema: pa.Schema,
                 entry_per_file: int = 10000,
                 ) -> None:
        self.root_dir = Path(root_dir)
        self.dataset_info = dataset_info
        self.schema = schema
        self.entry_per_file = entry_per_file
        
        self.data = []
        
    def add_entry(self, data_dict):
        self.data.append(data_dict)
        
        if len(self.data) >= self.entry_per_file :
            self.save_data()
            
    def save_data(self):
        if len(self.data) == 0:
            return
        # Convert the data to a pandas dataframe
        df = pd.DataFrame(self.data[:self.entry_per_file ])
        _ = pa.Table.from_pandas(df)        
        
        # Save the dataframe to parquet
        path = self.root_dir / self.dataset_info["name"] / self.dataset_info["scenes"][0]
        pq.write_to_dataset(table=pa.Table.from_pandas(df),
                            root_path=path,
                            schema=self.schema,
        )
        
        self.data = self.data[self.entry_per_file :]