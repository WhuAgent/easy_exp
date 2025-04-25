
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseDataset(ABC):
    """数据集基类"""
    @abstractmethod
    def get_data(self) -> Any:
        """获取数据"""
        pass

class ExampleDataset(BaseDataset):
    """示例数据集实现"""
    def get_data(self) -> Dict[str, Any]:
        return {
            "train_X": [[1, 2], [3, 4]],
            "train_y": [0, 1],
            "test_X": [[5, 6]],
            "test_y": [1]
        }
