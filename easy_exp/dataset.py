
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Iterator
import json

class Dataset(ABC):
    """数据集基类"""
    def __init__(self, data=None):
        self.data = data

    @staticmethod
    def from_json(file_path: str) -> "Dataset":
        """从JSON文件加载数据并创建数据集实例
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            数据集实例
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset(data)
        
    def __iter__(self) -> Iterator[Any]:
        """迭代数据集中的样本
        
        Returns:
            数据迭代器
        """
        data = self.data
        if isinstance(data, dict):
            yield from data.items()
        elif isinstance(data, list):
            yield from data
        else:
            yield data
            
    def __len__(self) -> int:
        """获取数据集大小
        
        Returns:
            数据集中的样本数量
        """
        data = self.data
        if isinstance(data, dict):
            return len(data)
        elif isinstance(data, list):
            return len(data)
        return 1
