
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseModel(ABC):
    """模型基类"""
    @abstractmethod
    def predict(self, data: Any) -> Any:
        """模型预测"""
        pass

class ExampleModel(BaseModel):
    """示例模型实现"""
    def predict(self, data: Dict[str, Any]) -> List[int]:
        return [0 if sum(x) < 3 else 1 for x in data["test_X"]]
