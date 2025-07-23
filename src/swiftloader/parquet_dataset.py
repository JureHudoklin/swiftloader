import io
import logging
import random
import copy
import json
from dataclasses import dataclass
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from typing import List, Callable, Any, Tuple, Dict, Literal, Sequence, Optional
from pathlib import Path

from torch.utils.data import Dataset

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


class ParquetDataset(Dataset):
    def __init__(self,
                 root_dir: str | Path,
                 datasets_info: List[DatasetInfo],
                 format_data: Callable[[dict], Any] | None = None,
                 *args,
                 **kwargs
    ) -> None:
        
        self.root_dir = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.datasets_info = datasets_info
        
        if format_data is None:
            format_data = self._format_data
        self.format_data = format_data

        self.pq_datasets: List[pq.ParquetDataset] = []
        for dataset_info in self.datasets_info:
            name = dataset_info["name"]
            for scene in dataset_info["scenes"]:
                self.pq_datasets.append(self._load_dataset(self.root_dir, name, scene))
        
        if not self.pq_datasets:
            raise ValueError("No Parquet datasets found at the specified locations.")

        self.schema = self.pq_datasets[0].schema
        
        # --- Caching and Indexing ---
        # Initialize cache with a structure that satisfies type checkers
        self._cache: Dict[str, Any] = {"key": (-1, -1), "table": pa.Table.from_pydict({})}
        self._build_index_map()

    def _build_index_map(self):
        """
        Builds a memory-efficient NumPy structured array for quickly finding the location of any item.
        This is crucial for performance and multi-worker safety.
        """
        map_entries = []
        self._length = 0
        for ds_idx, ds in enumerate(self.pq_datasets):
            for frag_idx, fragment in enumerate(ds.fragments):
                for rg_idx in range(fragment.num_row_groups):
                    num_rows_in_group = fragment.metadata.row_group(rg_idx).num_rows
                    map_entries.append((ds_idx, frag_idx, rg_idx, self._length, self._length + num_rows_in_group))
                    self._length += num_rows_in_group
        
        # Define the structured data type
        dtype = [('ds_idx', 'i4'), ('frag_idx', 'i4'), ('rg_idx', 'i4'), ('start_pos', 'i8'), ('end_pos', 'i8')]
        self.index_map = np.array(map_entries, dtype=dtype)
        
    def _load_dataset(self, root_dir: Path, name: str, scene: str) -> pq.ParquetDataset:
        """Load a Parquet dataset.

        Parameters
        ----------
        root_dir : Path
            The root directory of the dataset.
        name : str
            The name of the dataset.
        scene : str
            The name of the scene within the dataset.

        Returns
        -------
        fp.ParquetFile
            The loaded Parquet dataset.

        Raises
        ------
        FileNotFoundError
            If the specified directory does not exist.
        """
        path = root_dir / name / scene
        if not path.exists():
            raise FileNotFoundError(f"Directory {path} does not exist.")
        dataset = pq.ParquetDataset(path)
        return dataset
        
    def _format_data(self, data: dict) -> dict:
        """ A dummy function for formatting a single entry of data.

        Parameters
        ----------
        data : dict
        
        Returns
        -------
        dict
            The formatted data.
        """
        return data
    
    def _deserialize_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Deserializes a single entry from a Parquet file using schema metadata."""
        deserialized_entry = {}
        for field_name, value in entry.items():
            if value is None:
                deserialized_entry[field_name] = None
                continue

            field = self.schema.field(field_name)
            logical_type = field.metadata.get(b'logical_type') if field.metadata else None

            if logical_type == b'image' or field_name == "image":
                deserialized_entry[field_name] = Image.open(io.BytesIO(value))
            elif logical_type == b'numpy':
                deserialized_entry[field_name] = np.load(io.BytesIO(value))
            elif logical_type == b'json' or field_name == "annotations" or field_name == "image_annotation":
                deserialized_entry[field_name] = json.loads(value)
            else:
                deserialized_entry[field_name] = value
                
        return deserialized_entry

    def __len__(self):
        return self._length
    
    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx = self._length + idx
        if not 0 <= idx < self._length:
            raise IndexError(f"Index {idx} is out of range for dataset with length {self._length}")

        # Find the correct row group using a fast, vectorized NumPy query
        map_entry = self.index_map[(self.index_map['start_pos'] <= idx) & (idx < self.index_map['end_pos'])][0]
        ds_idx, frag_idx, rg_idx = map_entry["ds_idx"], map_entry["frag_idx"], map_entry["rg_idx"]
        
        # Use the cache to avoid re-reading the same row group
        cache_key = (ds_idx, frag_idx, rg_idx)
        if self._cache["key"] != cache_key:
            # Cache miss: read the new row group and update the cache
            print(f"Updating cache for dataset {ds_idx}, fragment {frag_idx}, row group {rg_idx}")
            fragment = self.pq_datasets[ds_idx].fragments[frag_idx]
            table = fragment.to_table(columns=self.schema.names)
            self._cache["key"] = cache_key
            self._cache["table"] = table
        
        # Get the specific row from the cached table
        local_idx = idx - map_entry["start_pos"]
        row = self._cache["table"].slice(local_idx, 1).to_pydict()
        
        # Convert list-of-values to value-per-key and deserialize
        entry = {k: v[0] for k, v in row.items()}
        deserialized_entry = self._deserialize_entry(entry)
        deserialized_entry["dataset_idx"] = idx
        return self.format_data(deserialized_entry)


class DataToParquet():
    def __init__(self,
                 root_dir: str | Path,
                 dataset_info: DatasetInfo,
                 entry_per_file: int = 10000,
                 row_group_size: int = 256
                 ) -> None:
        self.root_dir = Path(root_dir)
        self.dataset_info = dataset_info
        self.entry_per_file = entry_per_file
        self.row_group_size = row_group_size
        
        self.schema: Optional[pa.Schema] = None
        self.data = []

    def _create_schema(self, data_dict: dict):
        """Creates a pyarrow schema from a data dictionary, embedding logical types in metadata."""
        fields = []
        for key, value in data_dict.items():
            if isinstance(value, PILImage):
                meta = {b'logical_type': b'image'}
                fields.append(pa.field(key, pa.binary(), metadata=meta))
            elif isinstance(value, np.ndarray):
                meta = {b'logical_type': b'numpy'}
                fields.append(pa.field(key, pa.binary(), metadata=meta))
            elif isinstance(value, str):
                fields.append(pa.field(key, pa.string()))
            elif isinstance(value, (int, float, bool)):
                # Use pyarrow's numpy integration for primitive types
                pa_type = pa.from_numpy_dtype(np.dtype(type(value)))
                fields.append(pa.field(key, pa_type))
            elif isinstance(value, (list, tuple, dict)):
                # Serialize complex types to JSON strings
                meta = {b'logical_type': b'json'}
                fields.append(pa.field(key, pa.string(), metadata=meta))
            else:
                raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")
        self.schema = pa.schema(fields)

    def add_entry(self, data_dict: dict):
        """Add an entry to the dataset.

        This method appends a data dictionary to the dataset. If the number of entries in the dataset
        reaches the specified limit, the data is saved to a file.
        
        It automatically serializes complex types like PIL.Image and np.ndarray.

        Parameters
        ----------
        data_dict : dict
            A dictionary containing the data to be added to the dataset.

        Returns
        -------
        None
        """
        if self.schema is None:
            self._create_schema(data_dict)
        
        processed_entry = {}
        for key, value in data_dict.items():
            if isinstance(value, PILImage):
                # Serialize PIL Image to bytes (e.g., PNG format)
                buf = io.BytesIO()
                value.save(buf, format='PNG')
                processed_entry[key] = buf.getvalue()
            elif isinstance(value, np.ndarray):
                # Serialize NumPy array to bytes using numpy's save function
                buf = io.BytesIO()
                np.save(buf, value, allow_pickle=False) # allow_pickle=False for security
                processed_entry[key] = buf.getvalue()
            elif isinstance(value, (list, tuple, dict)):
                processed_entry[key] = json.dumps(value)
            else:
                processed_entry[key] = value

        self.data.append(processed_entry)
        
        if len(self.data) >= self.entry_per_file:
            self.save_data()
            
    def save_data(self):
        """
        Save the data to a parquet file. It is recommended to call this method after adding all the data.
        """
        if len(self.data) == 0 or self.schema is None:
            return
        # Convert the data to a pandas dataframe
        df = pd.DataFrame(self.data)
        try:
            table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        except pa.ArrowInvalid as e:
            logger.error(f"Error converting to Arrow Table: {e}")
            logger.error(f"Schema: {self.schema}")
            logger.error(f"Data sample: {df.head(1).to_dict()}")
            raise
        
        # Save the dataframe to parquet
        path = self.root_dir / self.dataset_info["name"] / self.dataset_info["scenes"][0]
        pq.write_to_dataset(table=table,
                            root_path=str(path),
                            schema=self.schema,
                            use_threads=True,
                            compression='zstd',
                            # For datasets with large items like images, a smaller row group size
                            # improves random access performance and reduces memory usage per read.
                            row_group_size=self.row_group_size
        )
        
        # Clear the data buffer after saving
        self.data = []
