import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import random

from swiftloader.folder_dataset import FolderDataset, DataToFolder
from swiftloader import loaders

class MaskLabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Instance Mask Labeling Tool")
        self.root.geometry("1400x900")
        
        # Initialize dataset
        self.dataset = FolderDataset(
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
        
        self.data_to_folder = DataToFolder(
            root_dir="/home/jure/datasets/folder_datasets",
            dataset_name="TIM_1",
            scene_name="scene_1_annotated_processed",
        )
        
        # Current data
        self.current_data = None
        self.current_index = 0
        self.dataset_list = list(self.dataset)
        self.z_values = {}  # Dictionary to store z-values for each annotation
        
        # Create GUI
        self.create_gui()
        
        # Load first image
        self.load_image(0)
    
    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for image display
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel for controls
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Image display
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar for zooming/panning
        toolbar_frame = ttk.Frame(left_frame)
        toolbar_frame.pack(fill=tk.X)
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Navigation controls
        nav_frame = ttk.Frame(right_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(nav_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.RIGHT)
        
        self.image_label = ttk.Label(nav_frame, text="Image 1/1")
        self.image_label.pack(side=tk.TOP)
        
        # Z-order controls
        z_frame = ttk.LabelFrame(right_frame, text="Z-Order Assignment")
        z_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Scrollable frame for z-order controls
        canvas_frame = tk.Canvas(z_frame)
        scrollbar = ttk.Scrollbar(z_frame, orient="vertical", command=canvas_frame.yview)
        self.scrollable_frame = ttk.Frame(canvas_frame)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_frame.configure(scrollregion=canvas_frame.bbox("all"))
        )
        
        canvas_frame.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas_frame.configure(yscrollcommand=scrollbar.set)
        
        canvas_frame.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Action buttons
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        # # # ttk.Button(action_frame, text="Calculate Visible Masks", 
        # # #           command=self.calculate_visible_masks).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(action_frame, text="Save Results", 
                  command=self.save_results).pack(fill=tk.X)
        
        # Status
        self.status_label = ttk.Label(right_frame, text="Ready")
        self.status_label.pack(side=tk.BOTTOM, pady=(10, 0))
    
    def load_image(self, index):
        if 0 <= index < len(self.dataset_list):
            self.current_index = index
            self.current_data = self.dataset_list[index]
            self.image_label.config(text=f"Image {index+1}/{len(self.dataset_list)}")
            
            # Initialize z-values for this image
            self.z_values = {}
            for i, annotation in enumerate(self.current_data["annotations"]):
                self.z_values[annotation["id"]] = i + 1
            
            self.display_image()
            self.create_z_controls()
    
    def display_image(self):
        self.ax.clear()
        
        # Display image
        image = np.array(self.current_data["image"])
        self.ax.imshow(image)
        
        # Generate colors for each annotation
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.current_data["annotations"])))
        
        # Display bounding boxes and masks
        for i, annotation in enumerate(self.current_data["annotations"]):
            bbox = annotation["bbox"]  # [x, y, w, h]
            mask_id = annotation["mask_id"]
            
            print(f"Annotation {i}: ID={annotation['id']}, mask_id={mask_id}, bbox={bbox}")
            
            # Draw bounding box
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                   linewidth=3, edgecolor=colors[i], facecolor='none')
            self.ax.add_patch(rect)
            
            # Add label with ID and z-value
            z_val = self.z_values.get(annotation["id"], 0)
            self.ax.text(bbox[0], bbox[1]-20, f"ID:{annotation['id']} Z:{z_val}", 
                        color=colors[i], fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
            
            # Display mask overlay - correct indexing for 3D mask array
            try:
                if mask_id < self.current_data["mask_full"].shape[2]:  # Check third dimension
                    # Correct indexing: mask_full is (height, width, num_masks)
                    mask = self.current_data["mask_full"][:, :, mask_id]
                    print(f"Mask {mask_id} shape: {mask.shape}, dtype: {mask.dtype}, unique values: {np.unique(mask)}")
                    
                    # Create colored mask overlay
                    mask_binary = mask > 0
                    if np.any(mask_binary):
                        # Create RGBA overlay
                        overlay = np.zeros((*mask.shape, 4))
                        overlay[:, :, :3] = colors[i][:3]
                        overlay[:, :, 3] = mask_binary.astype(float) * 0.5
                        self.ax.imshow(overlay)
                        print(f"Displayed mask for annotation {i}")
                    else:
                        print(f"Mask {mask_id} is empty")
                else:
                    print(f"Mask ID {mask_id} out of range (max: {self.current_data['mask_full'].shape[2]-1})")
            except Exception as e:
                print(f"Error displaying mask {mask_id}: {e}")
        
        self.ax.set_title(f"Image {self.current_index+1} - Instance Masks with Z-Order")
        self.ax.axis('off')
        self.canvas.draw()
    
    def create_z_controls(self):
        # Clear existing controls
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Create z-order controls for each annotation
        for i, annotation in enumerate(self.current_data["annotations"]):
            ann_id = annotation["id"]
            
            frame = ttk.Frame(self.scrollable_frame)
            frame.pack(fill=tk.X, pady=2)
            
            # Label
            label = ttk.Label(frame, text=f"Object {ann_id}:")
            label.pack(side=tk.LEFT)
            
            # Z-value spinbox
            z_var = tk.IntVar(value=self.z_values.get(ann_id, i+1))
            spinbox = ttk.Spinbox(frame, from_=1, to=len(self.current_data["annotations"]), 
                                 textvariable=z_var, width=5,
                                 command=lambda aid=ann_id, var=z_var: self.update_z_value(aid, var))
            spinbox.pack(side=tk.RIGHT)
            
            # Bind to update when value changes
            z_var.trace('w', lambda *args, aid=ann_id, var=z_var: self.update_z_value(aid, var))
    
    def update_z_value(self, annotation_id, z_var):
        try:
            self.z_values[annotation_id] = z_var.get()
            self.display_image()  # Refresh display
        except:
            pass
    
    def prev_image(self):
        if self.current_index > 0:
            self.load_image(self.current_index - 1)
    
    def next_image(self):
        if self.current_index < len(self.dataset_list) - 1:
            self.load_image(self.current_index + 1)
    
    # def calculate_visible_masks(self):
    #     if not self.current_data:
    #         return
        
    #     # Sort annotations by z-value (higher z-value = more in front)
    #     sorted_annotations = sorted(self.current_data["annotations"], 
    #                               key=lambda x: self.z_values.get(x["id"], 0))
        
    #     # Create visible masks
    #     image_shape = self.current_data["mask_full"].shape[:2]  # (height, width)
    #     visible_masks = np.zeros((len(self.current_data["annotations"]), *image_shape), dtype=bool)
    #     occupied_pixels = np.zeros(image_shape, dtype=bool)
        
        # # Process from back to front (lowest to highest z-value)
        # for i, annotation in enumerate(sorted_annotations):
        #     mask_id = annotation["mask_id"]
        #     # Correct indexing for 3D mask array
        #     original_mask = self.current_data["mask_full"][:, :, mask_id].astype(bool)
            
        #     # Find index in original annotations list
        #     orig_idx = next(j for j, a in enumerate(self.current_data["annotations"]) if a["id"] == annotation["id"])
            
        #     # Visible part is original mask minus already occupied pixels
        #     visible_mask = original_mask & ~occupied_pixels
        #     visible_masks[orig_idx] = visible_mask
            
        #     # Update occupied pixels
        #     occupied_pixels |= original_mask
        
        # # Store visible masks
        # self.visible_masks = visible_masks
        # self.status_label.config(text="Visible masks calculated")
        
    #     # Display result
    #     self.display_visible_masks()
    
    # def display_visible_masks(self):
    #     # Create a new window to show visible masks
    #     result_window = tk.Toplevel(self.root)
    #     result_window.title("Visible Masks Result")
    #     result_window.geometry("1000x800")
        
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    #     canvas = FigureCanvasTkAgg(fig, master=result_window)
    #     canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    #     # Display original image with full masks
    #     image = np.array(self.current_data["image"])
    #     ax1.imshow(image)
    #     ax1.set_title("Original Masks")
    #     ax1.axis('off')
        
    #     # Display image with visible masks only
    #     ax2.imshow(image)
    #     ax2.set_title("Visible Masks (After Z-Order Processing)")
    #     ax2.axis('off')
        
    #     # Overlay masks
    #     colors = plt.cm.tab10(np.linspace(0, 1, len(self.current_data["annotations"])))
        
    #     for i, annotation in enumerate(self.current_data["annotations"]):
    #         # Original mask on left
    #         mask_id = annotation["mask_id"]
    #         if mask_id < self.current_data["mask_full"].shape[2]:
    #             # Correct indexing for 3D mask array
    #             orig_mask = self.current_data["mask_full"][:, :, mask_id] > 0
    #             if np.any(orig_mask):
    #                 overlay1 = np.zeros((*orig_mask.shape, 4))
    #                 overlay1[:, :, :3] = colors[i][:3]
    #                 overlay1[:, :, 3] = orig_mask.astype(float) * 0.4
    #                 ax1.imshow(overlay1)
            
    #         # Visible mask on right
    #         if hasattr(self, 'visible_masks') and i < len(self.visible_masks):
    #             visible_mask = self.visible_masks[i]
    #             print(f"Visible mask {i}: shape={visible_mask.shape}, has_pixels={np.any(visible_mask)}")
                
    #             if np.any(visible_mask):
    #                 overlay2 = np.zeros((*visible_mask.shape, 4))
    #                 overlay2[:, :, :3] = colors[i][:3]
    #                 overlay2[:, :, 3] = visible_mask.astype(float) * 0.6
    #                 ax2.imshow(overlay2)
        
    #     fig.tight_layout()
    #     canvas.draw()
    
    def save_results(self):
        if not hasattr(self, 'visible_masks'):
            messagebox.showwarning("Warning", "Please calculate visible masks first!")
            return
        
        try:
            # Create mask_vis data
            mask_vis_data = self.visible_masks.astype(np.uint8)
            
            # Save using DataToFolder
            save_data = {
                "image": self.current_data["image"],
                "image_annotation": self.current_data["image_annotation"],
                "annotations": self.current_data["annotations"],
                "mask_full": self.current_data["mask_full"],
                "mask_vis": mask_vis_data,
                "z_values": self.z_values
            }
            
            # Note: You might need to modify this part based on your DataToFolder implementation
            # For now, we'll save as numpy files
            np.save(f"mask_vis_image_{self.current_index}.npy", mask_vis_data)
            
            self.status_label.config(text=f"Results saved for image {self.current_index+1}")
            messagebox.showinfo("Success", f"Visible masks saved for image {self.current_index+1}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MaskLabelingTool(root)
    root.mainloop()