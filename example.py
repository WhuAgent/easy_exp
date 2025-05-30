import re
import sys
from datetime import datetime

from easy_exp.dataset import Dataset
from easy_exp.model import BaseModel
from easy_exp.metric import BaseMetric
from easy_exp.exp_runner import BaseExpRunner

from easy_exp.llm.message import SystemMessage, UserMessage
from easy_exp.llm import llm


DEBUG = True if sys.gettrace() is not None else False


class ProblemModel(BaseModel):
    def predict(self, problem: str) -> str:
        llm.init()
        start = datetime.now()
        messages = [
            SystemMessage("You are a professional mathematician. You are given a problem and you need to solve it. Please put the answer in \\boxed{}"),
            UserMessage(problem)
        ]

        response = llm.chat_llm(messages, model="qwen2.5-32b-instruct")

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
        print(response)
        print()
        print()
        
        token_cost = llm.report()

        return response.content, (end - start).total_seconds(), token_cost


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
            return False
        system_prompt = "You are an experienced mathematics teacher with a strong grasp of logical reasoning and precise calculations, capable of quickly identifying the core of mathematical problems and evaluating the consistency between answers and solution processes."
        user_prompt = "Here is the math problem:\n\n{problem} \n\n with standard solution:\n\n{solution}\n\n The student's answer is:\n\n{answer}\n\n Please check whether the answer is correct or not. Please answer in True or False directly without any additional explanations."
        messages = [
            SystemMessage(system_prompt),
            UserMessage(user_prompt.format(problem=problem, solution=solution, answer=answer))
        ]

        response = llm.chat_llm(messages, model="qwen-turbo-latest")

        if re.search("true", response.content.strip().lower()):
            return True
        else:
            return False

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

        correct = self.check(problem, answer, solution)
        
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
    dataset = Dataset.from_json("sampled_problems.json")
    model = ProblemModel()
    metric = ProblemMetric()
    
    project = "agent-network-math"
    method_name = "test_easy_dataset"
    config = {
        "method": method_name,
        "level": 5,
        "type": "all",
        "judge": "gpt-4"
    }
    
    wandb_enabled = False if DEBUG else True
    
    # 请在这里补全实验代码
    with MATHExpRunner(project=project,
                       name=method_name,
                       config=config, 
                       wandb_enabled=wandb_enabled,
                    #    restore_from="92xi1igb"
                       ) as runner:
        runner.run(dataset, model, metric)

