from swiftloader import ParquetDataset, loaders
from swiftloader.parquet_dataset import DataToParquet

from pathlib import Path
from PIL import Image
import io
import json
import pyarrow as pa
from tqdm import tqdm

data_root = Path("/media/jure/ssd/datasets/parquet_datasets")

dataset = ParquetDataset(
    root_dir=data_root,
    datasets_info=[{"name": "industrial_objects", "scenes": ["train"]}],
    dataset_schema=[
        {"field": "image", "dtype": "binary", "loader": loaders.identity_loader},
        {"field": "annotations", "dtype": "string", "loader": loaders.json_loader},
    ],
    batch_size=1,
    shuffle=False,
    drop_last=False,
)
parquetizer = DataToParquet(
    root_dir=data_root,
    dataset_info={"name": "industrial_objects", "scenes": ["train_v3"]},
    schema=pa.schema(
        [
            pa.field("image", pa.binary()),
            pa.field("annotations", pa.string()),
            pa.field("image_annotation", pa.string()),
        ]
    ),
)

for data in tqdm(dataset):
    data = data[0]
    
    image_bytes = data["image"]
    a = data["annotations"]
    annotations = a["annotations"]
    
    img_ann = {
        "width": a["width"],
        "height": a["height"],
        "image_id": a["image_id"],
        "dataset_info": {
            "dataset": "objects365",
            "version": "1.0",
            "date_created": "2024-05-21",
        }
    }
    
    data_new = {
        "image": image_bytes,
        "annotations": json.dumps(annotations),
        "image_annotation": json.dumps(img_ann),
    }
    
    parquetizer.add_entry(data_new)
    
parquetizer.save_data()