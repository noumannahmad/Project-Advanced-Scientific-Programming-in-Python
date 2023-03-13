import os
import sys
sys.path.insert(0, os.path.abspath('./Improved_version_Face_Mask_Detection.py'))

extensions = ['sphinx.ext.autodoc']
autodoc_mock_imports = ['cv2', 'numpy'] # Add other packages if required
