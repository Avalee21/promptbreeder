import json
import random

import numpy as np
import pandas as pd
import yaml
from promptbreeder.models.llm import LLM
from promptbreeder.selector.selector import Selector
from sklearn.metrics.pairwise import cosine_similarity


class MutationOperator:
    def __init__(self, config_path, target_score):
        self.config_path = config_path
        with open(self.config_path, "r") as file:
            self.cfg = yaml.safe_load(file)

        self.thinking_styles = pd.read_csv(self.cfg["thinking_styles_file_path"])[
            "Thinking Style"
        ].tolist()
        self.mutation_prompts = pd.read_csv(self.cfg["mutation_prompts_file_path"])[
            "Prompt"
        ].tolist()
        self.task_description = self.cfg["task_description"]
        self.target_score = target_score

        with open(self.cfg["exemplars"]["exemplars_path"]) as f:
            self.exemplars = json.load(f)
        self.context_list_max_size = self.cfg["exemplars"]["context_list_max_size"]

        self.selector = Selector()
        self.model = LLM("/workspace/promptbreeder/config/llm.yml").get_llm(
            self.cfg["llm_name"]
        )

    def construct_few_shot_text(self, context_list):
        placeholders = self.cfg["exemplars"]["value_names"]
        new_context_list = []

        for context in context_list:
            text = self.cfg["exemplars"]["text"]
            for idx, placeholder in enumerate(placeholders, start=1):
                value = context.get(placeholder)
                if value is not None:
                    placeholder = "{value" + str(idx) + "}"
                    text = text.replace(placeholder, str(value))
            new_context_list.append(text)

        few_shot_text = "\n\n".join(new_context_list)
        print("#" * 100 + "\n", few_shot_text, "\n" + "#" * 100)
        return few_shot_text

    def zero_order_prompt_generation(self) -> str:
        task_prompt = self.model.generate(
            [f"""INSTRUCTION: {self.task_description} INSTRUCTION MUTANT: """]
        )[0]
        mutation_prompt = ""
        return mutation_prompt, task_prompt

    def first_order_prompt_generation(
        self, mutation_prompt: str, task_prompt: str, thinking_style=""
    ) -> str:
        task_prompt = self.model.generate(
            [
                f"""{mutation_prompt} {thinking_style} INSTRUCTION: {task_prompt} INSTRUCTION MUTANT: """
            ]
        )[0]
        return mutation_prompt, task_prompt

    def EDA_mutation(self, prompt_units):
        # Calculate the embeddings and filter the prompt units based on cosine similarity
        embeddings = np.array([prompt_unit.embedding for prompt_unit in prompt_units])
        embeddings_similarity = cosine_similarity(embeddings)

        filtered_indices = []
        for i in range(len(embeddings)):
            if not any(embeddings_similarity[i, :i] > 0.95):
                filtered_indices.append(i)

        filtered_prompt_units = [prompt_units[i] for i in filtered_indices]
        random.shuffle(filtered_prompt_units)

        shuffled_task_prompts = "\n".join(
            [unit.task_prompt for unit in filtered_prompt_units]
        )

        task_prompt = self.model.generate(
            [f"""{shuffled_task_prompts} Continue this list with a new response: """]
        )[0]
        mutation_prompt = ""
        return mutation_prompt, task_prompt

    def EDA_rank_index_mutation(self, prompt_units):
        # Calculate the embeddings and filter the prompt units based on cosine similarity
        embeddings = np.array([prompt_unit.embedding for prompt_unit in prompt_units])
        embeddings_similarity = cosine_similarity(embeddings)

        filtered_indices = []
        for i in range(len(embeddings)):
            if not any(embeddings_similarity[i, :i] > 0.95):
                filtered_indices.append(i)

        filtered_prompt_units = np.array(prompt_units)[filtered_indices]

        # Rank the filtered prompt units by ascending order of fitness
        ranked_prompt_units = sorted(
            filtered_prompt_units, key=lambda prompt_unit: prompt_unit.score
        )

        mutation_prompt = self.selector.select("Uniform Random", self.mutation_prompts)

        # Create a prefix to deceive the LLM
        prefix = f"INSTRUCTION: {mutation_prompt}\n"
        prefix += "A List of Responses in descending order of score.\n"
        prefix += f"{len(ranked_prompt_units)} is the best response. It resembles {len(ranked_prompt_units)-1} more than it does (1)\n"

        task_prompt = self.model.generate(
            [
                prefix
                + "\n".join(
                    [
                        f"{i+1}. {unit.task_prompt}"
                        for i, unit in enumerate(ranked_prompt_units)
                    ]
                )
                + "Continue this list with a better response: "
            ]
        )[0]
        return mutation_prompt, task_prompt

    def lineage_based_mutation(self, elites):
        task_prompt = self.model.generate(
            [
                "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY:\n"
                + "\n".join([elite.task_prompt for elite in elites])
                + "Continue this list with a better response: "
            ]
        )[0]
        mutation_prompt = ""
        return mutation_prompt, task_prompt

    def zero_order_hyper_mutation(self, task_prompt: str) -> str:
        thinking_style = self.selector.select("Uniform Random", self.thinking_styles)
        # TODO: How does it work?
        mutation_prompt = self.model.generate(
            inputs=[
                f"""<INSERT> {thinking_style} INSTRUCTION: {self.task_description} INSTRUCTION MUTANT: {task_prompt}"""
            ]
            # [f"""{self.task_description} {thinking_style}"""]
        )[0]
        mutation_prompt, task_prompt = self.first_order_prompt_generation(
            mutation_prompt, task_prompt, thinking_style
        )
        return mutation_prompt, task_prompt

    def first_order_hyper_mutation(self, mutation_prompt, task_prompt):
        hyper_mutation_prompt = (
            "Please summarize and improve the following instruction:"
        )
        mutation_prompt = self.model.generate(
            [
                f"""{hyper_mutation_prompt} INSTRUCTION: {mutation_prompt} INSTRUCTION MUTANT: """
            ]
        )[0]
        _, task_prompt = self.first_order_prompt_generation(
            mutation_prompt, task_prompt
        )
        return mutation_prompt, task_prompt

    def lamarckian_mutation(self):
        exemplars = self.selector.select(
            "Uniform Random Multiple", self.exemplars, n=self.context_list_max_size
        )
        task_prompt = self.model.generate(
            [
                f"""I instructed my friend to <INSERT>.The friend read the instruction and wrote an output for every one of the inputs.\nHere are the input-output pairs:\n{self.construct_few_shot_text(exemplars)}"""
            ]
        )[0]
        mutation_prompt = ""
        return mutation_prompt, task_prompt

    def prompt_crossover(self, prompt_units):
        mutation_prompt = ""
        if random.random() < 0.1:
            selected_unit = self.selector.select(
                "Roulette Wheel",
                prompt_units,
                [prompt_unit.score for prompt_unit in prompt_units],
            )
            task_prompt = selected_unit.task_prompt
        else:
            task_prompt = ""
        return mutation_prompt, task_prompt

    def context_shuffling(self, unit):
        if unit.score <= self.target_score:
            new_context = self.selector.select(
                "Uniform Random",
                [
                    exemplar
                    for exemplar in self.exemplars
                    if exemplar not in unit.context_list
                ],
            )
            if len(unit.context_list) < self.context_list_max_size:
                unit.context_list.append(new_context)
                print(f"Length of unit.context_list: {len(unit.context_list)}")
            else:
                # With 10% chance, resample the entire context list
                if random.random() < 0.1:
                    unit.context_list = self.selector.select(
                        "Uniform Random Multiple",
                        self.exemplars,
                        n=self.context_list_max_size,
                    )
                else:
                    replace_index = self.selector.select(
                        "Uniform Random", list(range(0, len(unit.context_list) - 1))
                    )
                    unit.context_list[replace_index] = new_context

            unit.few_shot_text = self.construct_few_shot_text(unit.context_list)


"""
python /workspace/promptbreeder/promptbreeder/evolution/mutation_operator.py
"""
