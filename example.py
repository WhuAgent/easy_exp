import re
import sys
import json
from datetime import datetime
from typing import Dict, Any, List

from easy_exp.dataset import BaseDataset
from easy_exp.model import BaseModel
from easy_exp.metric import BaseMetric
from easy_exp.exp_runner import BaseExpRunner

from utils.message import SystemMessage, UserMessage
from utils.chat import chat_llm


DEBUG = True if sys.gettrace() is not None else False


class ProblemDataset(BaseDataset):
    """数学问题数据集类"""
    
    def __init__(self, file_path: str):
        self.problems = self._load_problems(file_path)
        
    def _load_problems(self, file_path: str) -> List[Dict[str, Any]]:
        """从JSON文件加载问题数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def get_data(self) -> List[Dict[str, Any]]:
        """获取所有问题数据"""
        return self.problems
        
    def __len__(self) -> int:
        return len(self.problems)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.problems[idx]


class ProblemModel(BaseModel):
    def predict(self, problem: str) -> str:
        start = datetime.now()
        messages = [
            SystemMessage("You are a professional mathematician. You are given a problem and you need to solve it. Please put the answer in \\boxed{}"),
            UserMessage(problem)
        ]

        response = chat_llm(messages,
                            model="qwen2.5-32b-instruct",
                            api_key="sk-cf690968fd414e058f7cb0d2d3273c22",
                            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

        end = datetime.now()

        print("------------------------System------------------------")
        print(messages[0].content)
        print()
        print()
        print("------------------------User------------------------")
        print(messages[1].content)
        print()
        print()
        print("------------------------Response------------------------")
        print(response.content)
        print()
        print()

        return response.content, (end - start).total_seconds(),  response.token_cost


class ProblemMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        
        self.count = 0
        self.correct_count = 0
        self.accuracy = 0
        
        self.total_time = 0
        self.avg_time = 0
        
        self.total_token_cost = 0
        self.avg_token_cost = 0


    def check(self, problem, answer, solution):
        if answer is None:
            return False, 0
        system_prompt = "You are an experienced mathematics teacher with a strong grasp of logical reasoning and precise calculations, capable of quickly identifying the core of mathematical problems and evaluating the consistency between answers and solution processes."
        user_prompt = "Here is the math problem:\n\n{problem} \n\n with standard solution:\n\n{solution}\n\n The student's answer is:\n\n{answer}\n\n Please check whether the answer is correct or not. Please answer in True or False directly without any additional explanations."
        messages = [
            SystemMessage(system_prompt),
            UserMessage(user_prompt.format(problem=problem, solution=solution, answer=answer))
        ]

        response = chat_llm(messages,
                            model="qwen2.5-32b-instruct",
                            api_key="sk-cf690968fd414e058f7cb0d2d3273c22",
                            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

        if re.search("true", response.content.strip().lower()):
            return True, response.token_cost
        else:
            return False, response.token_cost

    def compute(self, problem, answer, progress, time, token_cost):
        from utils.math import get_answer
        answer = get_answer(answer)
        solution = get_answer(progress)

        print("------------------------Agent Answer------------------------")
        print(answer)
        print("------------------------Standard Answer------------------------")
        print(solution)
        print("------------------------Solution Progress------------------------")
        print(progress)
        print()
        print()

        correct, _ = self.check(problem, answer, solution)
        
        self.count += 1
        self.correct_count += correct
        self.accuracy = self.correct_count / self.count
        self.total_time += time
        self.avg_time = self.total_time / self.count
        self.total_token_cost += token_cost
        self.avg_token_cost = self.total_token_cost / self.count

        return {
            "is_correct": correct,
            "correct_num": self.correct_count,
            "accuracy": self.accuracy,
            "time": time,
            "average_time": self.avg_time,
            "cost": token_cost,
            "average_cost": self.avg_token_cost
        }


class MATHExpRunner(BaseExpRunner):
    def exp_one_step(self, step, data, model, metric):
        problem = data["problem"]
        answer, time, token_cost = model.predict(problem)
        results = metric.compute(problem, answer, data["solution"], time, token_cost)
        
        if results.get("is_correct", False):
           results.update({"log_path": f"true/{data['path']}"})
        else:
            results.update({"log_path": f"false/{data['path']}"})
        
        return results


if __name__ == "__main__":
    dataset = ProblemDataset("sampled_problems.json")
    model = ProblemModel()
    metric = ProblemMetric()
    
    project = "agent-network-math"
    method_name = "test_easy_exp_restore"
    config = {
        "method": method_name,
        "level": 5,
        "type": "all",
        "judge": "gpt-4"
    }
    
    wandb_enabled = False if DEBUG else True
    wandb_enabled = True
    
    # 请在这里补全实验代码
    with MATHExpRunner(project=project,
                       name=method_name,
                       config=config, 
                       wandb_enabled=wandb_enabled,
                       restore_from="92xi1igb"
                       ) as runner:
        runner.run(dataset, model, metric)

