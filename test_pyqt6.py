#!/usr/bin/env python3

import sys
import os

# Test PyQt6 availability
try:
    from PyQt6.QtWidgets import QApplication, QLabel, QWidget
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPixmap
    print("✓ PyQt6 successfully imported")
    
    # Create minimal test app
    app = QApplication(sys.argv)
    
    widget = QWidget()
    widget.setWindowTitle("PyQt6 Test")
    widget.resize(300, 200)
    
    label = QLabel("PyQt6 is working!", widget)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.resize(300, 200)
    
    print("✓ PyQt6 widgets created successfully")
    print("✓ Ready to run optimized mask labeling tool")
    
    # Don't show the window, just test imports
    # widget.show()
    # sys.exit(app.exec())
    
except ImportError as e:
    print(f"✗ PyQt6 not available: {e}")
    print("Install with: pip install PyQt6")
    sys.exit(1)

print("\nTest completed successfully. You can now run:")
print("python test_3_optimized.py")