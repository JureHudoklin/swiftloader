from .folder_dataset import FolderDataset
from .parquet_dataset import ParquetDataset

from importlib.metadata import version

__version__ = version("swiftloader")