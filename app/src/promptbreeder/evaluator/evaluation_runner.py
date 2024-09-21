import csv
import json
import os
import pathlib
import shutil
import subprocess
import time
from datetime import datetime

import pandas as pd
import yaml


class EvaluationRunner:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(self.config_path, "r") as file:
            self.cfg = yaml.safe_load(file)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # gen file
        self.exp_abbr_name = self.cfg["gen"]["exp_abbr_name"]
        self.dataset_class_name = self.cfg["gen"]["dataset_class_name"]
        self.dataset_postprocess_func = self.cfg["gen"]["dataset_postprocess_func"]
        self.pred_postprocess_func = self.cfg["gen"]["pred_postprocess_func"]
        self.input_columns = self.cfg["gen"]["input_columns"]
        self.output_column = self.cfg["gen"]["output_column"]
        self.evaluator = self.cfg["gen"]["evaluator"]
        self.user_prompt = json.dumps(self.cfg["gen"]["user_prompt"])
        self.bot_prompt = json.dumps(self.cfg["gen"]["bot_prompt"])

        # eval file
        self.model_abbr = self.cfg["eval"]["model"]["abbr"]
        self.model_path = self.cfg["eval"]["model"]["path"]
        self.model_type = self.cfg["eval"]["model"]["type"]
        self.query_per_second = self.cfg["eval"]["model"]["query_per_second"]
        self.max_out_len = self.cfg["eval"]["model"]["max_out_len"]
        self.max_seq_len = self.cfg["eval"]["model"]["max_seq_len"]
        self.batch_size = self.cfg["eval"]["model"]["batch_size"]
        self.temperature = self.cfg["eval"]["model"]["temperature"]
        self.api_base = self.cfg["eval"]["model"]["api_base"]

        # execute evaluation
        self.run_script_path = self.cfg["execute"]["run_script_path"]
        self.root_output_directory = self.cfg['execute']['output_directory']
        self.task_name = self.cfg["execute"]["task_name"]
        self.experiment_version = f"{self.exp_abbr_name}/EXP_{timestamp}"
        self.output_directory = f"{self.root_output_directory}/{self.task_name}/{self.experiment_version}"
        self.result_directory = f"{self.output_directory}/results"


        # record results
        # self.result_directory = f"{self.cfg['record']['result_directory']}/{self.exp_abbr_name}/{self.model_abbr}"
        self.performance_metric = self.cfg["record"]["performance_metric"]

    def gen_py_code(self, data_path, task_prompt, few_shot_text=""):
        input_columns_text = ", ".join([f'"{col}"' for col in self.input_columns])
        self.sys_prompt = (
            task_prompt + "Here are some examples for demonstration:\n" + few_shot_text
        )

        gen_py_code = f"""
import os

import importlib

from oocl_opencompass.datasets import (
    {self.dataset_class_name},
    {self.dataset_postprocess_func},
    {self.pred_postprocess_func},
)

from oocl_opencompass.evaluator import {self.evaluator}
from oocl_opencompass.inferencer import OoclInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

reader_cfg = dict(input_columns=[{input_columns_text}], output_column="{self.output_column}")

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role="SYSTEM",
                    fallback_role="HUMAN",
                    prompt={json.dumps(self.sys_prompt)},
                )
            ],
            round=[
                dict(
                    role="HUMAN",
                    prompt={self.user_prompt},
                ),
                dict(role="BOT", prompt={self.bot_prompt}),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=OoclInferencer,
        previous_output_json_filepath=os.getenv("REUSED_OUTPUT_PATH"),
    ),
)

eval_cfg = dict(
    evaluator=dict(type={self.evaluator}),
    pred_role="BOT",
    pred_postprocessor=dict(type={self.pred_postprocess_func}),
    dataset_postprocessor=dict(type={self.dataset_postprocess_func}),
)


eval_datasets = [
    dict(
        type={self.dataset_class_name},
        abbr="{self.exp_abbr_name}",
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
        datat_path="{data_path}",
    ),
]

"""
        # print("Generated gen.py Code:\n", gen_py_code)
        return gen_py_code

    def eval_py_code(self):
        eval_py_code = f"""
from mmengine.config import read_base
from opencompass import LocalRunner, NaivePartitioner, OpenICLInferTask

with read_base():
    from promptbreeder.evaluator.generated_modules.gen import eval_datasets

datasets = [*eval_datasets]

from oocl_opencompass.models import AzureOpenAI, QwenAPI, TgiAPI

meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
    reserved_roles=[
        dict(role="SYSTEM", api_role="SYSTEM"),
    ],
)


models = [
    dict(
        abbr="{self.model_abbr}",
        path="{self.model_path}",
        type={self.model_type},
        meta_template=meta_template,
        query_per_second={self.query_per_second},
        max_out_len={self.max_out_len},
        max_seq_len={self.max_seq_len},
        batch_size={self.batch_size},
        temperature={self.temperature},
        api_base="{self.api_base}",
    )
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=8, task=dict(type=OpenICLInferTask)),
)
"""
        # print("Generated eval.py Code:\n", eval_py_code)
        return eval_py_code

    def create_and_save_code(self, data_path, task_prompt, few_shot_text):
        dir_name = os.path.dirname(self.cfg["gen"]["save_path"])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        gen_code = self.gen_py_code(data_path, task_prompt, few_shot_text)
        with open(self.cfg["gen"]["save_path"], "w", encoding="utf-8") as f:
            f.write(gen_code)

        eval_code = self.eval_py_code()
        with open(self.cfg["eval"]["save_path"], "w", encoding="utf-8") as f:
            f.write(eval_code)

    def execute_evaluation(self):
        command = [
            "python",
            self.run_script_path,
            self.cfg["eval"]["save_path"],
            "-n",
            self.task_name,
            "-w",
            self.root_output_directory,
            "-v",
            self.experiment_version,
            "--debug",
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        print(result.stdout)

        if result.returncode == 0:
            print("Subprocess ran successfully!")
        else:
            print(f"Subprocess execution failed with return code: {result.returncode}")
            print(f"Standard Error Output:\n{result.stderr}")

    def record_performance(self, task_prompt, data_path):
        while not os.path.isdir(self.output_directory):
            print(f"Waiting for the directory {self.output_directory} to be created...")
            time.sleep(5)
        # Get the latest folder under output_directory
        list_of_folders = os.listdir(self.output_directory)
        latest_folder = max(
            [
                folder
                for folder in list_of_folders
                if os.path.isdir(os.path.join(self.output_directory, folder))
                and folder != "results"
            ],
            key=lambda folder: os.path.getctime(
                os.path.join(self.output_directory, folder)
            ),
        )
        latest_folder_path = os.path.join(self.output_directory, latest_folder)

        # Copy config files to each experiment folder
        config_files_folder = os.path.join(latest_folder_path, "config_files")
        os.makedirs(config_files_folder, exist_ok=True)
        for py_file in [
            self.cfg["gen"]["save_path"],
            self.cfg["eval"]["save_path"],
            self.config_path,
        ]:
            shutil.copy(py_file, config_files_folder)

        # Locate the CSV file under 'summary' folder
        summary_csv_path = os.path.join(
            latest_folder_path, "summary", f"summary_{latest_folder}.csv"
        )
        df_performance = pd.read_csv(summary_csv_path)

        # Retrieve/Calculate metric from summary file
        performance_score = df_performance[
            df_performance["metric"] == self.performance_metric
        ].iloc[0][self.model_abbr]
        invalid_format_count = df_performance[
            df_performance["metric"] == "invalid_format_count"
        ].iloc[0][self.model_abbr]
        question_count = df_performance[
            df_performance["metric"] == "question_count"
        ].iloc[0][self.model_abbr]
        follow_format_rate = (
            0
            if question_count == 0
            else (1 - invalid_format_count / question_count) * 100
        )

        output = {
            "System Prompt": json.dumps(self.sys_prompt),
            "User Prompt": self.user_prompt,
            "Model": self.model_abbr,
            "Eval Data": data_path,
            "Result Output Path": latest_folder_path,
            "Score": performance_score,
            "Follow Format Rate": follow_format_rate,
        }

        pathlib.Path(self.result_directory).mkdir(parents=True, exist_ok=True)
        result_file_path = os.path.join(self.result_directory, "results.csv")
        write_mode = "w" if not os.path.exists(result_file_path) else "a"

        with open(result_file_path, write_mode, newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=output.keys())
            if write_mode == "w":  # Write header only in write mode
                writer.writeheader()
            writer.writerow(output)

        return output

    def evaluate(self, data_path, task_prompt, few_shot_text):
        print("#" * 5, f"Start Evaluation on [{data_path}]", "#" * 5)
        self.create_and_save_code(data_path, task_prompt, few_shot_text)
        self.execute_evaluation()
        performance_result_dict = self.record_performance(task_prompt, data_path)
        return performance_result_dict
