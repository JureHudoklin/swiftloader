# Set matplotlib backend 
# This has to be done before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid hanging
import matplotlib.pyplot as plt
import torch
import signal
import sys
import tempfile
import os
from functools import partial
from tqdm import tqdm

import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import json
from swiftloader.folder_dataset import FolderDataset, DataToFolder
from swiftloader import loaders

if __name__ == "__main__":
    
    # Set up signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print('\n\nInterrupted by user (Ctrl+C)')
        plt.close('all')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Interactive Data Annotation Tool")
    print("=" * 40)
    print("This script will:")
    print("1. Load images and annotations from your dataset")
    print("2. Display each bounding box region")
    print("3. Allow you to assign category IDs (0-10) to each annotation")
    print("4. Save the updated annotations to a new dataset")
    print()
    print("Controls:")
    print("- Enter 0-10: Set category_id for the current annotation")
    print("- Enter 's': Skip current annotation (keep existing category_id)")
    print("- Enter 'q': Quit the program")
    print("=" * 40)
    print()

    dataset = FolderDataset(
        root_dir="/home/jure/datasets/folder_datasets",
        datasets_info=[{"name": "TIM_1", "scenes": ["scene_1_annotated"]}],
        dataset_schema=[
            {"field": "image", "dtype": "PIL", "loader": loaders.ImageLoader(out_type="pil")},
            {"field": "image_annotation", "dtype": "json", "loader": loaders.JsonLoader()},
            {"field": "annotations", "dtype": "json", "loader": loaders.JsonLoader()},
            {"field": "mask_full", "dtype": "numpy", "loader": loaders.NumpyLoader()},
        ],
        drop_last=False,
    )
    
    
    data_to_folder = DataToFolder(
        root_dir="/home/jure/datasets/folder_datasets",
        dataset_name="TIM_1",
        scene_name="scene_1_annotated_processed",
    )
    
    for data_idx, data in enumerate(tqdm(dataset, desc="Processing images")):
        image = data["image"]
        image_annotation = data["image_annotation"]
        annotations = data["annotations"]
        
        print(f"\nProcessing image {data_idx + 1}: {len(annotations)} annotations found")
        
        # Skip if no annotations
        if not annotations:
            print("No annotations found, skipping...")
            new_data = {
                "image": image,
                "image_annotation": image_annotation,
                "annotations": annotations,
                "mask_full": data.get("mask_full", None),
            }
            data_to_folder.add_entry(new_data)
            continue
        
        new_annotations = []
        
        ### Display image with interactive bounding box editing ###
        
        # Process each annotation
        for i, annotation in enumerate(annotations):
            bbox = annotation.get("bbox", None)
            if bbox is None:
                print(f"Annotation {i + 1} has no bbox, skipping...")
                new_annotations.append(annotation)
                continue
            
            # Extract bounding box (xywh format)
            x, y, w, h = bbox
            
            # Validate bounding box coordinates
            img_width, img_height = image.size
            if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                print(f"Warning: Bounding box {bbox} is outside image bounds ({img_width}x{img_height})")
            
            # Crop the image to the bounding box (with bounds checking)
            x_crop = max(0, x)
            y_crop = max(0, y)
            x_end = min(img_width, x + w)
            y_end = min(img_height, y + h)
            cropped_image = image.crop((x_crop, y_crop, x_end, y_end))
            
            # Display the cropped image
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.imshow(image)
            plt.title(f"Full Image (Image {data_idx + 1})")
            plt.axis('off')
            
            # Draw rectangle on full image to show current bounding box
            from matplotlib.patches import Rectangle
            ax = plt.gca()
            rect = Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            plt.subplot(2, 1, 2)
            plt.imshow(cropped_image)
            plt.title(f"Cropped Region - Annotation {i + 1}/{len(annotations)}\n"
                     f"Current category_id: {annotation.get('category_id', 'None')}\n"
                     f"BBox: [x={x}, y={y}, w={w}, h={h}]")
            plt.axis('off')
            plt.tight_layout()
            
            # Create visualization and save to temporary file
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.imshow(image)
            plt.title(f"Full Image (Image {data_idx + 1})")
            plt.axis('off')
            
            # Draw rectangle on full image to show current bounding box
            from matplotlib.patches import Rectangle
            ax = plt.gca()
            rect = Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            plt.subplot(2, 1, 2)
            plt.imshow(cropped_image)
            plt.title(f"Cropped Region - Annotation {i + 1}/{len(annotations)}\n"
                     f"Current category_id: {annotation.get('category_id', 'None')}\n"
                     f"BBox: [x={x}, y={y}, w={w}, h={h}]")
            plt.axis('off')
            plt.tight_layout()
            
            # Save the plot to a temporary file and try to open it
            import tempfile
            import os
            temp_file = tempfile.mktemp(suffix='.png')
            try:
                plt.savefig(temp_file, dpi=100, bbox_inches='tight')
                print(f"📊 Visualization saved to: {temp_file}")
                print("   You can open this file to see the image and bounding box.")
                
                # Try to open the image automatically (Linux)
                try:
                    os.system(f"xdg-open {temp_file} &")
                except:
                    pass
                    
            except Exception as e:
                print(f"Warning: Could not save visualization ({e})")
            
            # Print annotation details for text-only mode
            print(f"\n📋 Annotation {i + 1}/{len(annotations)} details:")
            print(f"  - Current category_id: {annotation.get('category_id', 'None')}")
            print(f"  - Bounding box: [x={x}, y={y}, w={w}, h={h}]")
            print(f"  - Image size: {img_width}x{img_height}")
            print(f"  - Cropped region size: {cropped_image.size}")
            
            # Wait for user input
            while True:
                try:
                    print(f"\n🎯 Options:")
                    print(f"  - Enter 0-10: Set category_id for this annotation")
                    print(f"  - Enter 's': Skip this annotation (keep current category_id)")
                    print(f"  - Enter 'q': Quit the program")
                    user_input = input(f"Your choice: ").strip().lower()
                    
                    if user_input == 'q':
                        plt.close('all')
                        # Clean up temp file
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                        print("Exiting...")
                        sys.exit(0)
                    elif user_input == 's':
                        print("⏭️  Skipping this annotation")
                        new_annotations.append(annotation)
                        break
                    elif user_input.isdigit() and 0 <= int(user_input) <= 10:
                        old_category = annotation.get('category_id', 'None')
                        annotation['category_id'] = int(user_input)
                        print(f"✅ Changed category_id from {old_category} to {user_input}")
                        new_annotations.append(annotation)
                        break
                    else:
                        print("❌ Invalid input. Please enter a number 0-10, 's' to skip, or 'q' to quit")
                        continue
                except KeyboardInterrupt:
                    plt.close('all')
                    # Clean up temp file
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                    print("\n\n🛑 Exiting due to Ctrl+C...")
                    sys.exit(0)
                except EOFError:
                    plt.close('all')
                    # Clean up temp file
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                    print("\n\n🛑 Exiting due to EOF...")
                    sys.exit(0)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            
            # Clean up temp file and close plot
            try:
                os.remove(temp_file)
            except:
                pass
            plt.close('all')
        
        # Update the annotations in new_data
        # new_annotations = annotations
        
        #####################
        
        new_data = {
            "image": image,
            "image_annotation": image_annotation,
            "annotations": new_annotations,
            "mask_full": data.get("mask_full", None),
        }
        
        data_to_folder.add_entry(new_data)
        print(f"Saved image {data_idx + 1} with {len(new_annotations)} annotations")
    
    print("\n" + "=" * 40)
    print("Processing completed!")
    print(f"Updated annotations saved to: {data_to_folder.root_dir}/{data_to_folder.dataset_name}/{data_to_folder.scene_name}")
    print("=" * 40)