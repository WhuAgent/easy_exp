
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseMetric(ABC):
    """评估指标基类"""
    @abstractmethod
    def compute(self, predictions: Any, targets: Any) -> Dict[str, float]:
        """计算评估指标"""
        pass

class ExampleMetric(BaseMetric):
    """示例评估指标实现"""
    def compute(self, predictions: List[int], data: Dict[str, Any]) -> Dict[str, float]:
        correct = sum(1 for p, t in zip(predictions, data["test_y"]) if p == t)
        return {"accuracy": correct / len(predictions)}
