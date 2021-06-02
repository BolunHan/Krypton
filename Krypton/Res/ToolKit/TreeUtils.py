import json
import os
import pickle
import uuid
from collections import OrderedDict
from typing import Optional, Dict, Union, List

from . import GLOBAL_LOGGER
from .Collections import AttrDict

LOGGER = GLOBAL_LOGGER.getChild(os.path.basename(os.path.splitext(__file__)[0]))
__all__ = ['TreeView', 'TreeNode']


class TreeView(object):
    def __init__(self, content, **kwargs):
        self.content = content
        self.sub_view: List['TreeView'] = []

        self.connector = AttrDict({
            'leaves': kwargs.pop('connector.leaves', ['├── ', '└── ']),
            'branch': kwargs.pop('connector.branch', ['│   ', '    ']),
        })

        self.pre: str = kwargs.pop('pre', '')
        self.is_last: bool = kwargs.pop('is_last', True)
        self.is_root: bool = kwargs.pop('is_root', False)

    def append(self, sub_view: 'TreeView'):
        if self.sub_view:
            self.sub_view[-1].is_last = False
            self.sub_view[-1].pre = f'{self.pre}{self.connector.leaves[0]}'  # self.pre + '│   '

        sub_view.pre = f'{self.pre}{self.connector.leaves[1]}'  # self.pre + '    '
        self.sub_view.append(sub_view)

    def _render(self, parent_pre=''):
        if self.is_root:
            tree_view = f'{self.content}\n'
        else:
            tree_view = f'{parent_pre}{self.pre}{self.content}\n'

        for sub_view in self.sub_view:
            if self.is_root:
                sub_pre = ''
            elif self.is_last:
                sub_pre = f'{parent_pre}{self.connector.branch[1]}'
            else:
                sub_pre = f'{parent_pre}{self.connector.branch[0]}'

            tree_view += sub_view._render(parent_pre=sub_pre)

        return tree_view

    def print(self):
        print(str(self))

    def log(self, logger, level):
        logger.log(level, f'\n{str(self)}')

    def __str__(self):
        self.is_root = True
        result = self._render()[:-1]
        self.is_root = False
        return result


class TreeNode(object):
    """The basic node of tree structure"""

    def __init__(self, name: str, node_id: str = None, parent: Optional['TreeNode'] = None):
        self.name = name

        if parent is not None:
            self.parent: Optional['TreeNode'] = parent
            self.parent.add_child(self)
        else:
            self.parent: Optional['TreeNode'] = None

        # validate node_id
        if node_id is None:
            self.id = str(uuid.uuid4())
        elif isinstance(node_id, str):
            if '/' in node_id:
                raise ValueError('node_id contains invalid symbol "/".')
            self.id = node_id
        else:
            raise TypeError(f'node_id must be string, you have {node_id}')

        self.child: Dict[str, 'TreeNode'] = OrderedDict()

    def __repr__(self):
        return f'<TreeNode>{{name: {self.name}, path: {self.index_path}, parent: {None if self.parent is None else self.parent.name}, child: {[node.name for node in self.child.values()]}}}'

    def __str__(self):
        return str(self.name)

    def __getitem__(self, item: Union[int, str, List[int]]):
        if isinstance(item, int):
            return list(self.child.values())[item]
        elif isinstance(item, str):
            path_id = item.split('/')

            # absolute path
            if path_id[0] == '':
                if len(path_id) == 2:
                    return self.root
                else:
                    node = self
                    for node_id in path_id[2:]:
                        node = node.child.get(node_id)
                    return node
            # relative path
            else:
                node = self
                for node_id in path_id:
                    node = node.child.get(node_id)
                return node
        elif isinstance(item, list):
            node = self
            for index in item:
                node = node[index]
            return node
        else:
            raise ValueError(f'Invalid index type {type(item)}, use int, str or List[int]')

    def __contains__(self, child: 'TreeNode'):
        return child.id in self.child

    def __len__(self):
        """return number of children node"""
        return len(self.child)

    def __bool__(self):
        """always return True for exist node"""
        return True

    @classmethod
    def from_json(cls, json_message: Union[str, bytes, bytearray]) -> 'TreeNode':
        json_dict = json.loads(json_message)

        path = json_dict.pop('<@AbsolutePath>', None)

        root = cls(
            name=json_dict.pop('name'),
            node_id=json_dict.pop('id'),
            parent=json_dict.pop('parent')
        )

        for key, value in json_dict.items():
            if key == 'child':
                for json_str in value:
                    node = cls.from_json(json_str)

                    root.child[node.id] = node
                    node.parent = root
            elif '<@pickle_dump>' in key:
                setattr(root, key, pickle.loads(value.replace('<@pickle_dump>', '')))
            else:
                setattr(root, key, value)

        if path is None:
            return root
        else:
            return root[path]

    def to_json(self, with_parent: bool = True) -> str:
        data_dict = {}

        if self.parent is None or not with_parent:
            root = self
        else:
            root = self.root

        for key, value in root.__dict__.items():
            if key == 'child':
                data_dict[key] = [node.to_json(with_parent=False) for node in value.values()]
            elif key == 'parent':
                data_dict[key] = None
            else:
                # noinspection PyBroadException
                try:
                    _ = json.dumps(value)
                    data_dict[key] = value
                except Exception as _:
                    data_dict[key] = f'<@pickle_dump>{pickle.dumps(value)}'

        if with_parent:
            data_dict['<@AbsolutePath>'] = self.path

        return json.dumps(data_dict)

    def pipe(self, func: callable, *args, **kwargs):
        return func(self, *args, **kwargs)

    def find_index(self, node_id: str) -> int:
        return list(self.child).index(node_id)

    def find_child(self, node_id: str) -> 'TreeNode':
        """get a child node of current node by node_id"""
        node = self.child.get(node_id, None)
        return node

    def pop_child(self, node_id: str) -> 'TreeNode':
        """pop a child node from current node by node_id"""
        node = self.child.pop(node_id)
        node.parent = None
        return node

    def add_child(self, node: 'TreeNode') -> 'TreeNode':
        """add a child node to current node"""
        if not isinstance(node, TreeNode):
            raise ValueError('TreeNode only add another TreeNode obj as child')

        node.parent = self
        self.child[node.id] = node
        return node

    def update_child(self, node: 'TreeNode', node_id: str = None) -> 'TreeNode':
        """update child of this node with new node."""
        if node_id is None:
            node_id = node.id

        if node_id not in self.child:
            raise KeyError(f'No child with id {node_id}')

        # old_node = self.child[node_id]

        node.id = node_id
        self.add_child(node)

        return node

    def items(self):
        """iterate through child items"""
        return self.child.items()

    def _tree_view(self, pre: str = '', is_last: bool = True, is_root: bool = True):
        """dump tree to string"""
        if is_root:
            tree_view = str(self) + '\n'
        else:
            if is_last:
                tree_view = pre + '└── ' + str(self) + '\n'
            else:
                tree_view = pre + '├── ' + str(self) + '\n'

        child_id_list = list(self.child.keys())

        for i in range(len(child_id_list)):
            node = self.child[child_id_list[i]]
            child_is_root = False

            if i == len(child_id_list) - 1:
                child_is_last = True
            else:
                child_is_last = False

            if is_root:
                child_pre = pre + ''
            else:
                if is_last:
                    child_pre = pre + '    '
                else:
                    child_pre = pre + '│   '
            tree_view += node._tree_view(pre=child_pre, is_last=child_is_last, is_root=child_is_root)

        return tree_view

    @property
    def level(self):
        """get depth of this tree"""
        if self.parent is None:
            return 1
        else:
            return self.parent.level + 1

    @property
    def tree_view(self) -> str:
        """get structure of this tree, use print(node.tree_view)"""
        LOGGER.warning(DeprecationWarning('.tree_view deprecated, use .TreeView instead'))
        return self._tree_view()[:-1]

    # noinspection PyPep8Naming
    @property
    def TreeView(self):
        tree_view = TreeView(content=str(self))
        for child in self.child.values():
            tree_view.append(child.TreeView)
        return tree_view

    @property
    def path(self) -> str:
        """return path string (from its root to current node)"""
        if self.parent:
            return f'{self.parent.path}/{self.id}'
        else:
            return f'/{self.id}'

    @property
    def index_path(self) -> List[int]:
        if self.parent:
            return self.parent.index_path + [self.parent.find_index(node_id=self.id)]
        else:
            return []

    @property
    def root(self):
        """get the root node (the most left) of itself, if exist"""
        if self.parent:
            return self.parent.root
        else:
            return self

    @property
    def leaves(self):
        """get leaf nodes of (the most right) of itself"""
        leaves = []
        if self.child:
            for node in self.child.values():
                leaves.extend(node.leaves)
        else:
            leaves = [self]
        return leaves

    @property
    def sub_nodes(self):
        """get all nodes at right side, WITHOUT ITSELF"""
        sub_nodes = []
        if self.child:
            for node in self.child.values():
                sub_nodes.append(node)
                sub_nodes.extend(node.sub_nodes)
        else:
            sub_nodes = []
        return sub_nodes
