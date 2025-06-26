
"""
实验框架核心模块
支持wandb记录的可选集成
"""
from contextlib import redirect_stdout
from datetime import datetime
import json
import shutil
import click
import wandb
import sys
from typing import Callable, Any, Dict, Optional
from functools import wraps
import os
from tqdm import tqdm

from easy_exp.dataset import Dataset


class BaseExpRunner:
    def __init__(self, 
                 project: str, 
                 name: str, 
                 config: Dict[str, Any] = None, 
                 wandb_enabled: bool = True,
                 restore_from: str = None):
        self.project = project
        self.name = name
        self.config = config or {}
        self.wandb_enabled = wandb_enabled
        self.restore_from = restore_from
        self.restored_data = None
        
    def __enter__(self):
        """上下文管理器入口， 开始实验"""
        if os.path.exists("exp_log"):
            shutil.rmtree("exp_log")
        os.makedirs("exp_log", exist_ok=True)
        
        if self.wandb_enabled:
            run = wandb.init(project=self.project, name=self.name, config=self.config)
            if self.restore_from is not None:
                self.restored_data = self.restore_from_run(run.entity, self.project, self.restore_from)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出， 结束实验"""
        if self.wandb_enabled:
            wandb.finish()
    
    def restore_from_run(self, entity, project: str, run_id: str):
        click.echo(click.style(f"Restoring from run: {entity}/{project}/{run_id}", fg='yellow'))
        api = wandb.Api()
        
        run = api.run(f"{entity}/{project}/{run_id}")

        history = run.history()
        
        data = []

        for i, row in history.iterrows():
            res_dict = {}
            for idx in row.index:
                if not idx.startswith('_'):
                    res_dict[idx] = row[idx]
            data.append(res_dict)
        
        click.echo(click.style(f"Downloading files...", fg='yellow'))
        cnt = 0
        for file in run.files():
            if file.name.startswith("exp_log"):
                cnt += 1
        dowload_cnt = 0
        for file in run.files():
            if file.name.startswith("exp_log"):
                click.echo(click.style(f"Downloading file({dowload_cnt}/{cnt}): {file.name}", fg='yellow'))
                file.download()
                dowload_cnt += 1

        return data


    def record(self, results: Dict[str, Any], restore_flag):
        """记录单步指标"""
        final_log_path = None
        if "log_path" in results:
            final_log_path = os.path.normpath(os.path.join("exp_log", results.get("log_path")))
            if not restore_flag:
                os.makedirs(os.path.dirname(final_log_path), exist_ok=True)
                try:
                    if os.path.exists("temp.log"):
                        shutil.move("temp.log", final_log_path)
                except Exception as e:
                    print(f"Failed to move log file: {e}")
        
        if self.wandb_enabled:
            wandb.log(results)
            if final_log_path and os.path.exists(final_log_path):
                wandb.save(final_log_path)

    def exp_one_step(self, step: int, data: Any, model: Any, metric) -> Optional[Dict[str, Any]]:
        """子类需要重写的实验逻辑方法"""
        raise NotImplementedError("Subclasses must implement exp_one_step() method")

    def run(self, dataset: Dataset, model, metric, ):
        """运行实验的入口方法，自动处理所有循环"""
        for i, data in tqdm(enumerate(dataset), total=len(dataset)):
            try:
                print() # 打印空行
                data_str = json.dumps(data, indent=4)
                click.echo(click.style(f"Evaluating: \n{data_str}", fg='blue'))
                
                if os.path.exists("temp.log"): # 清理上次的log
                    os.remove("temp.log")

                DEBUG = True if sys.gettrace() is not None else False

                if DEBUG:
                    if self.restored_data and i < len(self.restored_data):
                        click.echo(click.style(f"Restore from existing data...", fg='yellow'))
                        results = self.restored_data[i]
                        metric.record(**results)
                        restore_flag = True
                    else:
                        results = self.exp_one_step(i, data, model, metric)
                        restore_flag = False
                else:
                    with open("temp.log", "w", encoding="UTF-8") as f, redirect_stdout(f):
                        if self.restored_data and i < len(self.restored_data):
                            click.echo(click.style(f"Restore from existing data...", fg='yellow'))
                            results = self.restored_data[i]
                            metric.record(**results)
                            restore_flag = True
                        else:
                            results = self.exp_one_step(i, data, model, metric)
                            restore_flag = False
                
                if results is not None:
                    results_str = json.dumps(results, indent=4)
                    click.echo(click.style(f"Results: \n{results_str}", fg='green'))
                    self.record(results, restore_flag)
                else:
                    click.echo(click.style("No results", fg='red'))
            except KeyboardInterrupt as e:
                if self.wandb_enabled:
                    wandb.finish()
                break
            except Exception as e:
                import traceback
                error_trace_back = traceback.format_exc()
                with open("error.log", "a", encoding="UTF-8") as f:
                    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    f.write("While Evaluating: \n")
                    f.write(data_str)
                    f.write("\n")
                    f.write(error_trace_back)
                    f.write("\n")
                    f.write("\n")
                click.echo(click.style(f"Error processing step {i}: {str(e)}", fg='red'))
                continue

