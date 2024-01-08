# SwiftLoader

SwiftLoader is a set of dataloaders opotimized for pytorch.

## Features

- **Classification**: The [`classification.py`](src/swiftloader/classification.py) module provides a dataloader for classification tasks.
- **Object Detection**: The [`object_detection.py`](src/swiftloader/object_detection.py) module provides a dataloader for object detection tasks.
- **Template Based Detection**: The [`template_based_detection.py`](src/swiftloader/template_based_detection.py) module provides a dataloader for template based detection tasks.


## Installation

To install SwiftLoader, run the following command inside the root directory of the repository:

```sh
pip install -e .
```

## Usage
To use the provided dataloaders the dataset should have the following structure:

```
root
├── dataset_1
|   ├── categories.json
│   ├── scene_1
│   │   ├── images
│   │   │   ├── 000000.jpg
│   │   │   ├── 000001.jpg
│   │   │   ├── ...
│   │   ├── annotations
│   │   │   ├── 000000.json
│   │   │   ├── 000001.json
│   │   │   ├── ...
│   │   ├── image_annotations
│   │   │   ├── 000000.jpg
│   │   │   ├── 000001.jpg
│   │   │   ├── ...
...
├── objects
```

The `categories.json` file should contain a list of categories in the dataset. For example:

```json
[
    {"id": 0, "name": "category_1", "supercategory": "category_1"},
    {"id": 1, "name": "category_2", "supercategory": "category_2"},
    ...
]
```
Image annotation files should contain the following fields: `id`, `width`, `height`, `file_name`. Any additinal annotations should be put in the "attributes" field. For example:

```json
{
    "id": 0,
    "width": 1000,
    "height": 1000,
    "file_name": "000000.jpg",
    "attributes": {
        "attribute_1": "value_1",
        "attribute_2": "value_2",
        ...
    }
}
```

Object detection annotation files should contain the following fields: `image_id`, `category_id`, `bbox`. Any additinal annotations should be put in the "attributes" field. For example:


```json
[
    {"image_id": 0, "category_id": 0, "bbox": [0, 0, 100, 100], "attributes": {"attribute_1": "value_1", "attribute_2": "value_2", ...}},
    {"image_id": 0, "category_id": 1, "bbox": [100, 100, 200, 200], "attributes": {"attribute_1": "value_1", "attribute_2": "value_2", ...}},
    ...
]
```

```

To use SwiftLoader, import the necessary modules from the swiftloader package. For example:

```python
from swiftloader import SwiftClassification
from swiftloader import SwiftObjectDetection
from swiftloader import SwiftTemplateObjectDetection
```
```

## License
This project is licensed under the MIT License - see the (LICENSE) file for details.