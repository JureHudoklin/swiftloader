import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSpinBox, 
                             QScrollArea, QFrame, QMessageBox, QSplitter,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsTextItem)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QBrush
import numpy as np
from PIL import Image
import cv2

from swiftloader.folder_dataset import FolderDataset, DataToFolder
from swiftloader import loaders


class OptimizedImageWidget(QGraphicsView):
    """Fast image display with mask overlays using QGraphicsView"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Create scene
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Image data
        self.base_image = None
        self.mask_data = None
        self.annotations = []
        self.z_values = {}
        self.colors = []
        
        # Graphics items
        self.image_item = None
        self.overlay_items = []
        self.bbox_items = []
        self.label_items = []
        
        # Enable smooth zooming
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
    def set_image_data(self, image, mask_data, annotations, z_values):
        """Set image data and trigger update"""
        self.base_image = image
        self.mask_data = mask_data
        self.annotations = annotations
        self.z_values = z_values
        
        # Generate colors for annotations
        self.colors = []
        for i in range(len(annotations)):
            hue = (i * 137.5) % 360  # Golden angle spacing
            color = QColor.fromHsv(int(hue), 200, 255)
            self.colors.append(color)
        
        self.update_display()
    
    def update_z_values(self, z_values):
        """Update z-values and trigger efficient redraw"""
        self.z_values = z_values
        self.update_overlays()
    
    def update_display(self):
        """Update the display with new image and overlays"""
        if self.base_image is None:
            return
        
        # Clear existing items
        self.scene.clear()
        self.image_item = None
        self.overlay_items = []
        self.bbox_items = []
        self.label_items = []
        
        # Convert PIL image to QPixmap
        if isinstance(self.base_image, Image.Image):
            img_array = np.array(self.base_image)
        else:
            img_array = self.base_image
            
        # Ensure RGB format
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Add image to scene
        pixmap = QPixmap.fromImage(q_image)
        self.image_item = self.scene.addPixmap(pixmap)
        
        # Add overlays
        self.update_overlays()
        
        # Fit image in view
        self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
    
    def update_overlays(self):
        """Update mask overlays and bounding boxes"""
        if not self.annotations or self.mask_data is None:
            return
        
        # Clear existing overlay items
        for item in self.overlay_items + self.bbox_items + self.label_items:
            self.scene.removeItem(item)
        self.overlay_items = []
        self.bbox_items = []
        self.label_items = []
        
        # Create mask overlay
        height, width = self.mask_data.shape[:2]
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Sort annotations by z-value for proper layering
        sorted_annotations = sorted(enumerate(self.annotations), 
                                  key=lambda x: self.z_values.get(x[1]["id"], 0))
        
        # Draw masks in z-order
        for orig_idx, annotation in sorted_annotations:
            mask_id = annotation["mask_id"]
            if mask_id < self.mask_data.shape[2]:
                mask = self.mask_data[:, :, mask_id] > 0
                color = self.colors[orig_idx]
                
                # Apply mask color
                overlay[mask, 0] = color.red()
                overlay[mask, 1] = color.green()
                overlay[mask, 2] = color.blue()
                overlay[mask, 3] = 128  # Semi-transparent
        
        # Add overlay to scene if it has content
        if np.any(overlay[:, :, 3] > 0):
            overlay_image = QImage(overlay.data, width, height, 4 * width, QImage.Format.Format_RGBA8888)
            overlay_pixmap = QPixmap.fromImage(overlay_image)
            overlay_item = self.scene.addPixmap(overlay_pixmap)
            self.overlay_items.append(overlay_item)
        
        # Add bounding boxes and labels
        for i, annotation in enumerate(self.annotations):
            bbox = annotation["bbox"]  # [x, y, w, h]
            color = self.colors[i]
            
            # Create bounding box
            rect_item = QGraphicsRectItem(bbox[0], bbox[1], bbox[2], bbox[3])
            pen = QPen(color, 3)
            rect_item.setPen(pen)
            rect_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self.scene.addItem(rect_item)
            self.bbox_items.append(rect_item)
            
            # Create label
            z_val = self.z_values.get(annotation["id"], 0)
            label_text = f"ID:{annotation['id']} Z:{z_val}"
            
            # Label background
            bg_rect = QGraphicsRectItem(bbox[0], bbox[1] - 25, len(label_text) * 8, 20)
            bg_rect.setBrush(QBrush(QColor(255, 255, 255, 200)))
            bg_rect.setPen(QPen(Qt.PenStyle.NoPen))
            self.scene.addItem(bg_rect)
            self.label_items.append(bg_rect)
            
            # Label text
            text_item = QGraphicsTextItem(label_text)
            text_item.setPos(bbox[0] + 2, bbox[1] - 22)
            text_item.setDefaultTextColor(QColor(0, 0, 0))
            font = QFont("Arial", 10, QFont.Weight.Bold)
            text_item.setFont(font)
            self.scene.addItem(text_item)
            self.label_items.append(text_item)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.base_image is None:
            return
        
        # Scale factor for zoom
        scale_factor = 1.2
        if event.angleDelta().y() < 0:
            scale_factor = 1.0 / scale_factor
        
        # Apply zoom
        self.scale(scale_factor, scale_factor)
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key.Key_R:  # Reset zoom and pan with 'R' key
            if self.image_item:
                self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
        super().keyPressEvent(event)


class FastMaskLabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fast Instance Mask Labeling Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize dataset
        self.dataset = FolderDataset(
            root_dir="/home/jure/datasets/folder_datasets",
            datasets_info=[{"name": "TIM_1", "scenes": ["scene_2_annotated"]}],
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
            scene_name="scene_2_annotated_vis",
        )
        
        # Current data
        self.current_data = None
        self.current_index = 0
        self.dataset_list = list(self.dataset)
        self.z_values = {}
        self.visible_masks = None
        
        # Update timer for batching z-value changes
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._delayed_update)
        
        self.setup_ui()
        self.load_image(0)
    
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Image display
        self.image_widget = OptimizedImageWidget()
        splitter.addWidget(self.image_widget)
        
        # Right panel - Controls
        right_panel = QWidget()
        right_panel.setMaximumWidth(350)
        right_panel.setMinimumWidth(300)
        right_layout = QVBoxLayout(right_panel)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        right_layout.addLayout(nav_layout)
        
        self.image_label = QLabel("Image 1/1")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.image_label)
        
        # Z-order controls
        z_frame = QFrame()
        z_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        z_layout = QVBoxLayout(z_frame)
        z_layout.addWidget(QLabel("Z-Order Assignment"))
        
        # Scrollable area for z-controls
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        z_layout.addWidget(self.scroll_area)
        
        right_layout.addWidget(z_frame)
        
        # Action buttons
        self.calc_btn = QPushButton("Calculate Visible Masks")
        self.save_btn = QPushButton("Save Results")
        self.calc_btn.clicked.connect(self.calculate_visible_masks)
        self.save_btn.clicked.connect(self.save_results)
        
        right_layout.addWidget(self.calc_btn)
        right_layout.addWidget(self.save_btn)
        
        # Status
        self.status_label = QLabel("Ready")
        right_layout.addWidget(self.status_label)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([1000, 400])  # Initial sizes
    
    def load_image(self, index):
        """Load image at given index"""
        if 0 <= index < len(self.dataset_list):
            self.current_index = index
            self.current_data = self.dataset_list[index]
            self.image_label.setText(f"Image {index+1}/{len(self.dataset_list)}")
            
            # Initialize z-values
            self.z_values = {}
            for i, annotation in enumerate(self.current_data["annotations"]):
                self.z_values[annotation["id"]] = i + 1
            
            self.update_display()
            self.create_z_controls()
    
    def update_display(self):
        """Update the image display"""
        if self.current_data:
            self.image_widget.set_image_data(
                self.current_data["image"],
                self.current_data["mask_full"],
                self.current_data["annotations"],
                self.z_values
            )
    
    def create_z_controls(self):
        """Create z-order control widgets"""
        # Clear existing controls
        for i in reversed(range(self.scroll_layout.count())):
            self.scroll_layout.itemAt(i).widget().deleteLater()
        
        # Create controls for each annotation
        for i, annotation in enumerate(self.current_data["annotations"]):
            ann_id = annotation["id"]
            
            control_widget = QWidget()
            control_layout = QHBoxLayout(control_widget)
            
            label = QLabel(f"Object {ann_id}:")
            spinbox = QSpinBox()
            spinbox.setRange(1, len(self.current_data["annotations"]))
            spinbox.setValue(self.z_values.get(ann_id, i+1))
            
            # Connect with debouncing
            spinbox.valueChanged.connect(lambda v, aid=ann_id: self.update_z_value(aid, v))
            
            control_layout.addWidget(label)
            control_layout.addWidget(spinbox)
            
            self.scroll_layout.addWidget(control_widget)
    
    def update_z_value(self, annotation_id, value):
        """Update z-value with debounced refresh"""
        self.z_values[annotation_id] = value
        
        # Restart timer for debounced update
        self.update_timer.stop()
        self.update_timer.start(100)  # 100ms delay
    
    def _delayed_update(self):
        """Delayed update for batching z-value changes"""
        self.image_widget.update_z_values(self.z_values)
    
    def prev_image(self):
        """Load previous image"""
        if self.current_index > 0:
            self.load_image(self.current_index - 1)
    
    def next_image(self):
        """Load next image"""
        if self.current_index < len(self.dataset_list) - 1:
            self.load_image(self.current_index + 1)
    
    def calculate_visible_masks(self):
        """Calculate visible masks based on z-order"""
        if not self.current_data:
            return
        
        # Sort annotations by z-value (higher z-value = more in front)
        sorted_annotations = sorted(self.current_data["annotations"], 
                                  key=lambda x: self.z_values.get(x["id"], 0), reverse=True)
        
        # Create visible masks
        image_shape = self.current_data["mask_full"].shape[:2]
        visible_masks = np.zeros((len(self.current_data["annotations"]), *image_shape), dtype=bool)
        occupied_pixels = np.zeros(image_shape, dtype=bool)
        
        # Process from back to front (lowest to highest z-value)
        for annotation in sorted_annotations:
            mask_id = annotation["mask_id"]
            original_mask = self.current_data["mask_full"][:, :, mask_id].astype(bool)
            
            # Find index in original annotations list
            orig_idx = next(j for j, a in enumerate(self.current_data["annotations"]) if a["id"] == annotation["id"])
            
            # Visible part is original mask minus already occupied pixels
            visible_mask = original_mask & ~occupied_pixels
            visible_masks[orig_idx] = visible_mask
            
            # Update occupied pixels
            occupied_pixels |= original_mask
        
        # Store results
        self.visible_masks = visible_masks
        self.status_label.setText("Visible masks calculated")
        
        # Show results in a message box
        total_pixels = sum(np.sum(mask) for mask in visible_masks)
        QMessageBox.information(self, "Results", f"Calculated visible masks with {total_pixels} total visible pixels")
    
    def save_results(self):
        """Save the results"""
        if not hasattr(self, 'visible_masks') or self.visible_masks is None:
            QMessageBox.warning(self, "Warning", "Please calculate visible masks first!")
            return
        
        try:
            # Prepare data for saving using data_to_folder.add_entry
            save_data = {
                "annotations": self.current_data["annotations"],  # dict
                "image_annotation": self.current_data["image_annotation"],  # dict
                "image": self.current_data["image"],  # PIL Image
                "mask_full": self.current_data["mask_full"],  # numpy array
                "mask_vis": self.visible_masks.astype(np.uint8).transpose(1, 2, 0),  # numpy array
            }
            
            # Use DataToFolder to save the entry
            self.data_to_folder.add_entry(save_data)
            
            self.status_label.setText(f"Results saved for image {self.current_index+1}")
            QMessageBox.information(self, "Success", f"Results saved using DataToFolder for image {self.current_index+1}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = FastMaskLabelingTool()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()