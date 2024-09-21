import yaml

from promptbreeder.evaluator.evaluation_runner import EvaluationRunner
from promptbreeder.evolution.mutation_operator import MutationOperator
from promptbreeder.initializer.prompt_initializer import PromptInitializer


class AutoPromptOptimizer:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(self.config_path, "r") as file:
            self.cfg = yaml.safe_load(file)

        self.is_baseline = self.cfg["is_baseline"]
        self.strategy = self.cfg["strategy"]
        self.num_iterations = self.cfg["num_iterations"]
        self.target_score = self.cfg["target_score"]
        self.test_all_units = self.cfg["test_all_units"]

        self.test_data_path = self.cfg["test_data_path"]

        self.mutation_operator = MutationOperator(
            self.cfg["mutation_operator_config_path"], self.target_score
        )
        self.evaluation_runner = EvaluationRunner(
            self.cfg["evaluation_runner_config_path"]
        )
        self.prompt_initializer = PromptInitializer(
            self.cfg["prompt_initializer_config_path"],
            self.mutation_operator,
            self.evaluation_runner,
        )

    def initialize_prompts(self):
        population = self.prompt_initializer.initialize_prompts(self.is_baseline)
        return population

    def optimize_prompts(self, population):
        if self.is_baseline:
            return population
        if self.strategy == "Score":
            while not any(unit.score >= self.target_score for unit in population.units):
                population.binary_tournament()
                population.write_to_csv()
        else:
            for i in range(self.num_iterations):
                population.binary_tournament()
                population.write_to_csv()
        return population

    # def eval_best_unit(self, population):
    #     best_unit = population.find_best_unit()
    #     best_unit.display_prompt_unit_info()
    #     score = best_unit.evaluate_fitness(self.test_data_path)
    #     print("The performance score of test set: ", score)

    def eval_best_units(self, population):
        best_units = population.find_best_units(self.test_all_units)

        for best_unit in best_units:
            best_unit.display_prompt_unit_info()
            score = best_unit.evaluate_fitness(self.test_data_path)
            print("The performance score of test set: ", score)


if __name__ == "__main__":
    config_path = "/workspace/promptbreeder/config/auto_prompt_optimizer.yml"
    auto_prompt_optimizer = AutoPromptOptimizer(config_path)
    population = auto_prompt_optimizer.initialize_prompts()
    population = auto_prompt_optimizer.optimize_prompts(population)
    # auto_prompt_optimizer.eval_best_unit(population)
    auto_prompt_optimizer.eval_best_units(population)


"""
python /workspace/promptbreeder/promptbreeder/auto_prompt_optimization.py
"""
