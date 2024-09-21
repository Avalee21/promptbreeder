import copy
import csv
import os

import numpy as np
import pandas as pd
import yaml
from promptbreeder.models.llm import LLM
from promptbreeder.selector.selector import Selector
from transformers import pipeline


class Population:
    def __init__(
        self, mutation_operator, evaluation_runner, valid_data_path, feature_extractor
    ):
        self.units = []
        self.elites = []
        self.mutation_operator = mutation_operator
        self.evaluation_runner = evaluation_runner
        self.output_file_path = (
            f"{self.evaluation_runner.result_directory}/prompt_units.csv"
        )
        self.valid_data_path = valid_data_path
        self.selector = Selector()
        self.feature_extractor = feature_extractor
        self.mutation_methods = [
            self.mutation_operator.zero_order_prompt_generation,
            self.mutation_operator.zero_order_hyper_mutation,
            # self.mutation_operator.lamarckian_mutation,
            self.mutation_operator.first_order_prompt_generation,
            self.mutation_operator.first_order_hyper_mutation,
            self.mutation_operator.context_shuffling,
            # self.EDA_mutation,
            # self.EDA_rank_index_mutation,
            self.lineage_based_mutation,
            self.prompt_crossover,
        ]
        self.mutation_index = 0

    def EDA_mutation(self):
        mutation_prompt, task_prompt = self.mutation_operator.EDA_mutation(self.units)
        return mutation_prompt, task_prompt

    def EDA_rank_index_mutation(self):
        mutation_prompt, task_prompt = self.mutation_operator.EDA_rank_index_mutation(
            self.units
        )
        return mutation_prompt, task_prompt

    def lineage_based_mutation(self):
        mutation_prompt, task_prompt = self.mutation_operator.lineage_based_mutation(
            self.elites
        )
        return mutation_prompt, task_prompt

    def prompt_crossover(self):
        mutation_prompt, task_prompt = self.mutation_operator.prompt_crossover(
            self.units
        )
        return mutation_prompt, task_prompt

    def add_unit(self, mutation_prompt, task_prompt, parent_prompt):
        print("\nAdding prompt unit ...")
        self.units.append(
            PromptUnit(
                mutation_prompt,
                task_prompt,
                parent_prompt,
                self.evaluation_runner,
                self.valid_data_path,
                self.feature_extractor,
            )
        )

    def binary_tournament(self):
        for _ in range(len(self.units)):
            unit1, unit2 = self.selector.select(
                "Uniform Random Multiple", self.units, n=2
            )
            if unit1.score != unit2.score:
                break
        else:
            print("Max attempts reached. All units might have the same score.")
            unit1, unit2 = self.selector.select(
                "Uniform Random Multiple", self.units, n=2
            )

        winner, loser = (unit1, unit2) if unit1.score > unit2.score else (unit2, unit1)

        mutation_method = self.selector.select("Uniform Random", self.mutation_methods)
        # mutation_method = self.mutation_operator.context_shuffling
        # mutation_method = self.mutation_methods[self.mutation_index]

        mutated_winner = copy.deepcopy(winner)
        is_updated = mutated_winner.mutate(mutation_method)

        self.mutation_index = (self.mutation_index + 1) % len(self.mutation_methods)

        if is_updated and (mutated_winner.score > loser.score):
            # Replace the loser with the mutated copy of the winner
            loser_index = self.units.index(loser)
            self.units[loser_index] = mutated_winner

            self.update_elites()

    def update_elites(self):
        best_score = max(self.units, key=lambda unit: unit.score).score
        best_units = [
            unit
            for unit in self.units
            if unit.score == best_score and unit not in self.elites
        ]
        self.elites.extend(best_units)

    # def find_best_unit(self):
    #     max_score = max(self.units, key=lambda unit: unit.score).score
    #     best_units = [unit for unit in self.units if unit.score == max_score]

    #     if len(best_units) == 1:
    #         return best_units[0]
    #     else:
    #         return min(
    #             best_units, key=lambda x: len(str(x.task_prompt) + str(x.few_shot_text))
    #         )

    def find_best_units(self, test_all_units=False):
        seen = set()
        deduped_units = []

        for unit in self.units:
            unit_hash = hash(str(unit.task_prompt) + str(unit.few_shot_text))

            if unit_hash not in seen:
                seen.add(unit_hash)
                deduped_units.append(unit)

        if test_all_units:
            return deduped_units
        else:
            max_score = max(deduped_units, key=lambda unit: unit.score).score
            return [unit for unit in deduped_units if unit.score == max_score]

    def write_to_csv(self):
        """
        Write the population data to a CSV file.
        """
        headers = [
            "Mutation Prompt",
            "Task Prompt",
            "Parent Prompt",
            "Mutation Method",
            "Few Shot Context",
            "Generation",
            "Score",
        ]
        data = [
            [
                unit.mutation_prompt,
                unit.task_prompt,
                unit.parent_prompt,
                unit.mutation_method,
                unit.few_shot_text,
                unit.generation,
                unit.score,
            ]
            for unit in self.units
        ]

        with open(self.output_file_path, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(headers)
            csv_writer.writerows(data)

    @staticmethod
    def load_from_file(
        file_path,
        mutation_operator,
        evaluation_runner,
        valid_data_path,
        feature_extractor,
    ):
        print("Loadig prompt units from file ...")
        df = pd.read_csv(file_path, encoding="utf-8")
        df["Few Shot Context"].fillna("", inplace=True)

        population = Population(
            mutation_operator, evaluation_runner, valid_data_path, feature_extractor
        )
        for _, row in df.iterrows():
            unit = PromptUnit(
                row["Mutation Prompt"],
                row["Task Prompt"],
                row["Parent Prompt"],
                evaluation_runner,
                valid_data_path,
                feature_extractor,
                row["Few Shot Context"],
                row["Generation"],
                row["Score"],
            )
            population.units.append(unit)
        population.update_elites()
        return population


class PromptUnit:
    def __init__(
        self,
        mutation_prompt,
        task_prompt,
        parent_prompt,
        evaluation_runner,
        valid_data_path,
        feature_extractor,
        few_shot_text="",
        generation=0,
        score=None,
    ):
        self.mutation_prompt = mutation_prompt
        self.task_prompt = task_prompt
        self.parent_prompt = parent_prompt
        self.mutation_method = ""

        self.evaluation_runner = evaluation_runner
        self.valid_data_path = valid_data_path
        self.feature_extractor = feature_extractor

        self.few_shot_text = few_shot_text or ""
        self.context_list = []  # TODO
        self.generation = generation or 0
        self.score = (
            self.evaluate_fitness(self.valid_data_path) if score is None else score
        )
        self.embedding = self.get_embeddings(self.task_prompt)

    def get_embeddings(self, text):
        return np.mean(
            self.feature_extractor(text, **{"truncation": True, "max_length": 512})[0],
            axis=0,
        )

    def display_prompt_unit_info(self):
        attributes_to_display = [
            "mutation_prompt",
            "task_prompt",
            "mutation_method",
            "parent_prompt",
            "few_shot_text",
            "generation",
            "score",
        ]
        for name in attributes_to_display:
            print(f"{name}: {getattr(self, name)}")

    def evaluate_fitness(self, data_path):
        performance_result_dict = self.evaluation_runner.evaluate(
            data_path, self.task_prompt, self.few_shot_text
        )
        return performance_result_dict["Score"]

    def adapt_inputs(self, method):
        if method.__name__ in [
            "first_order_prompt_generation",
            "first_order_hyper_mutation",
        ]:
            return (self.mutation_prompt, self.task_prompt)

        elif method.__name__ in [
            "zero_order_hyper_mutation",
        ]:
            return (self.task_prompt,)

        elif method.__name__ in [
            "zero_order_prompt_generation",
            "lamarckian_mutation",
            "EDA_mutation",
            "EDA_rank_index_mutation",
            "lineage_based_mutation",
            "prompt_crossover",
        ]:
            return ()

    def mutate(self, mutation_method):
        print(f"\n\nSelected method: {mutation_method.__name__}")
        self.mutation_method = mutation_method.__name__

        is_updated = False
        old_task_prompt = self.task_prompt
        old_few_shot_text = self.few_shot_text

        if mutation_method.__name__ != "context_shuffling":
            input_args = self.adapt_inputs(mutation_method)
            (
                mutation_prompt,
                task_prompt,
            ) = mutation_method(*input_args)

            if mutation_prompt:
                self.mutation_prompt = mutation_prompt
            if task_prompt:
                self.task_prompt = task_prompt
        else:
            mutation_method(self)
            new_few_shot_text = self.few_shot_text
            if old_few_shot_text == new_few_shot_text:
                print(
                    "Attention! The few_shot_text has NOT been updated in context_shuffling!"
                )

        if str(self.task_prompt) + str(self.few_shot_text) != str(
            old_task_prompt
        ) + str(old_few_shot_text):
            is_updated = True

            self.parent_prompt = old_task_prompt
            self.score = self.evaluate_fitness(self.valid_data_path)
            self.embedding = self.get_embeddings(self.task_prompt)
            self.generation += 1

        return is_updated


class PromptInitializer:
    def __init__(self, config_path, mutation_operator, evaluation_runner):
        self.config_path = config_path
        with open(self.config_path, "r") as file:
            self.cfg = yaml.safe_load(file)

        self.valid_data_path = self.cfg["valid_data_path"]
        self.prompt_units_file = self.cfg["prompt_units_file_path"]
        self.num_prompts = self.cfg["num_prompts"]
        self.load_from_file = self.cfg["load_from_file"]

        self.selector = Selector()
        self.llm_instances = LLM("/workspace/promptbreeder/config/llm.yml")
        self.mutation_operator = mutation_operator
        self.evaluation_runner = evaluation_runner
        self.feature_extractor = pipeline(
            "feature-extraction",
            model="bert-base-uncased",
            tokenizer="bert-base-uncased",
        )

    def initialize_prompts(self, is_baseline=False):
        if is_baseline:
            population = Population(
                self.mutation_operator,
                self.evaluation_runner,
                self.valid_data_path,
                self.feature_extractor,
            )
            population.add_unit(
                "None",  # mutation_prompt,
                self.mutation_operator.task_description,  # task_prompt,
                self.mutation_operator.task_description,
            )
            population.write_to_csv()
            return population

        if self.load_from_file and os.path.exists(self.prompt_units_file):
            population = Population.load_from_file(
                self.prompt_units_file,
                self.mutation_operator,
                self.evaluation_runner,
                self.valid_data_path,
                self.feature_extractor,
            )
        else:
            population = Population(
                self.mutation_operator,
                self.evaluation_runner,
                self.valid_data_path,
                self.feature_extractor,
            )

            for i in range(self.num_prompts):
                mutation_prompt = self.selector.select(
                    "Uniform Random", self.mutation_operator.mutation_prompts
                )
                thinking_style = self.selector.select(
                    "Uniform Random", self.mutation_operator.thinking_styles
                )
                (
                    mutation_prompt,
                    task_prompt,
                ) = self.mutation_operator.first_order_prompt_generation(
                    mutation_prompt,
                    self.mutation_operator.task_description,
                    thinking_style,
                )

                population.add_unit(
                    mutation_prompt,
                    task_prompt,
                    self.mutation_operator.task_description,
                )

                population.write_to_csv()

        return population
