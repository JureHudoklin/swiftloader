import matplotlib.pyplot as plt
import torch
from functools import partial, reduce
from tqdm import tqdm

import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from swiftloader.parquet_dataset import ParquetDataset
from swiftloader import loaders

if __name__ == "__main__":
    
    
    def compose(*funcs):
        return reduce(lambda f, g: lambda x: f(g(x)), funcs, lambda x: x)
    
    
    f1 = lambda x: x + 1
    f2 = lambda x: x * 2
    f3 = lambda x: x ** 2
    
    f = compose(f1, f2, f3)
    print(f(1))
    
    # a = [torch.rand(3, 512, 512) for _ in range(10)]
    # a = map(lambda x: x, a)

    # print(torch.cat(list(a)))
    # exit()    
    
    base_transforms = T.Resize((512, 512))
    input_transforms = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
        ]
    )
    
    output_transforms = T.Compose([
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], 
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        T.ToDtype(torch.uint8, scale=True),
    ])
    
    dataset = ParquetDataset(
        root_dir="/media/jure/ssd/datasets/parquet_datasets",
        datasets_info=[{"name": "imagenet_1k", "scenes": ["train"]}],
        dataset_schema = [
                {"field": "image", "dtype": "binary", "loader": loaders.image_loader},
                {"field": "image_annotation", "dtype": "string", "loader": loaders.json_loader},
            ],
            batch_size=16,
            drop_last=False,
            shuffle=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=4,
    )
    
    
    
    it = iter(dataloader)
    for data in tqdm(it, total=len(dataloader)):
        pass
    
    exit()
    
    
    mean, std = dataset.get_dataset_mean_std()
    
    print(mean, std)
    exit()
    
    
    for i in range(5,15):
        img, target = dataset[i]
        # print(target)
        img = output_transforms(img)
        img = img.to(torch.uint8)
        fig = plot_switft_dataset(img, None)
        plt.show()
        plt.close(fig)
    
    # dataloader = get_swift_loader(
    #     dataset=dataset,
    #     split="val",
    #     batch_size=2,
    #     num_workers=4,
    #     pin_memory=True,
    #     collate_fn=partial(create_nested_tensor_batch, size_constant=(512, 1024)),
    # )
    
    # itr = iter(dataloader)
    
    # for i in range(len(dataloader)):
    #     batch = next(itr)
    #     fig = plot_switft_dataset_batch(batch.samples, batch.targets, images_format=output_transforms)
    #     plt.show()
