# Set matplotlib backend 
# This has to be done before importing pyplot
import matplotlib.pyplot as plt
import torch
from functools import partial

import torchvision.transforms.v2 as T
import json
from swiftloader import SwiftObjectDetection
from swiftloader import SwiftTemplateObjectDetection
from swiftloader.util.display import plot_switft_dataset

if __name__ == "__main__":
    
    base_transforms = T.Resize((512, 512))
    template_transforms = T.Resize((512, 512))
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
    
    dataset = SwiftTemplateObjectDetection(
        "/media/jure/ssd/datasets/OBJECTS_DATASET",
        [{"name": "COCO"}],
        ignore_templates=False,
        input_transforms=input_transforms,
        base_transforms=base_transforms,
        template_transforms=template_transforms,
        input_template_transforms=input_transforms,
    )

    
    for i in range(0, 5):
        img, template, target = dataset[i]
        print(template)
        # for temp in template:
        #     fig = plot_switft_dataset(temp, None)
        #     plt.show()

        img = output_transforms(img)
        img = img.to(torch.uint8)
        fig = plot_switft_dataset(img, target)
        plt.show()
    
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
