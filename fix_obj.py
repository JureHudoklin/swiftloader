from swiftloader import ParquetDataset, loaders
from swiftloader.parquet_dataset import DataToParquet

from pathlib import Path
from PIL import Image
import io
import json
import pyarrow as pa

data_root = Path("/home/jure/datasets/parquet_datasets")

dataset = ParquetDataset(
    root_dir=data_root,
    datasets_info=[{"name": "objects365", "scenes": ["val"]}],
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
    dataset_info={"name": "objects365", "scenes": ["val_"]},
    schema=pa.schema(
        [
            pa.field("image", pa.binary()),
            pa.field("annotations", pa.string()),
            pa.field("image_annotations", pa.string()),
        ]
    ),
)

for data in dataset:
    data = data[0]
    
    image_bytes = data["image"]
    a = data["annotations"]
    annotations = a["annotations"]
    
    img_ann = {
        "width": a["width"],
        "height": a["height"],
        "image_id": a["image_id"],
    }
    
    data_new = {
        "image": image_bytes,
        "annotations": json.dumps(annotations),
        "image_annotations": json.dumps(img_ann),
    }
    
    parquetizer.add_entry(data_new)
    
parquetizer.save_data()