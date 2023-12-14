import matplotlib.pyplot as plt
import torch
from functools import partial

import torchvision.transforms.v2 as T

from swiftloader import SwiftClassification
from swiftloader.util.display import plot_switft_dataset

if __name__ == "__main__":
    
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
    
    dataset = SwiftClassification(
        "/media/jure/ssd/datasets/OBJECTS_DATASET",
        [{"name": "SM_v03"}],
        input_transforms=input_transforms,
        base_transforms=base_transforms,
        attributes=["OK", "zalitost"]
    )
    
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
