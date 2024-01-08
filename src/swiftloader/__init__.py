from .object_detection import SwiftObjectDetection
from .classification import SwiftClassification
from .template_based_detection import SwiftTemplateObjectDetection

from importlib.metadata import version

__version__ = version("swiftloader")