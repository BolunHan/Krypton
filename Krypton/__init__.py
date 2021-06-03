__version__ = "0.1.2"

from . import Res
from . import Base


def generate_package_structure():
    from Res.ToolKit import TreeNode as node, get_current_path
    import os

    root = node('Krypton')

    for path in os.listdir('.'):
        if path in ['DataCache', 'TestResult', '__pycache__']:
            pass
        else:
            p = root.add_child(node(path))
            if os.path.isdir(os.path.join('.', path)):
                for _ in os.listdir(os.path.join('.', path)):
                    p.add_child(node(_))
    root.TreeView.print()
    return root
