import random


class Selector:
    def __init__(self):
        pass

    def roulette_wheel_selection(self, population, fitness_scores):
        total_fit = sum(fitness_scores)
        selected = random.uniform(0, total_fit)
        cum_sum = 0
        for i, fitness in enumerate(fitness_scores):
            cum_sum += fitness
            if cum_sum > selected:
                return population[i]

    def uniform_random_selection(self, population):
        return random.choice(population)

    def uniform_random_selection_multiple(self, population, k=1):
        return random.sample(population, k=k)

    def independent_binomial_selection(self, population, max_length):
        # TODO
        selection_probability = max_length / len(population)
        selected_population = [
            ex for ex in population if random.random() < selection_probability
        ]
        return selected_population

    def top_selection(self, population, fitness_scores, n):
        sorted_population = [
            x for _, x in sorted(zip(fitness_scores, population), reverse=True)
        ]
        return sorted_population[:n]

    def select(
        self, selection_method, population, fitness_scores=None, n=1, max_length=1
    ):
        if selection_method == "Roulette Wheel":
            return self.roulette_wheel_selection(population, fitness_scores)
        elif selection_method == "Uniform Random":
            return self.uniform_random_selection(population)
        elif selection_method == "Uniform Random Multiple":
            return self.uniform_random_selection_multiple(population, n)
        elif selection_method == "Independent Bernoulli":
            return self.independent_binomial_selection(population, max_length)
        elif selection_method == "Top":
            return self.top_selection(population, fitness_scores, n)
        else:
            raise ValueError("Invalid selection method")


if __name__ == "__main__":
    # Example usage
    population = [
        "Individual 1",
        "Individual 2",
        "Individual 3",
        "Individual 4",
        "Individual 5",
    ]
    fitness_scores = [5, 3, 9, 6, 8]

    selector = Selector()

    selected_individual = selector.select("Roulette Wheel", population, fitness_scores)
    print("Selected individual (Roulette Wheel):", selected_individual)

    selected_individual = selector.select("Uniform Random", population)
    print("Selected individual (Uniform Random):", selected_individual)

    selected_individuals = selector.select("Top", population, fitness_scores, n=3)
    print("Selected individuals (Top):", selected_individuals)

"""
python /workspace/promptbreeder/promptbreeder/selector/selector.py
"""
