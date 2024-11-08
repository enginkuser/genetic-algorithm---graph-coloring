from itertools import count

from GraphColoringProblem import *


class GeneticSolver:
    """
    You are free to define new methods and change the inside of "solve" method
    """

    problem: GraphColoringProblem       #: Given problem with the graph
    graph: Graph                        #: Given problem graph
    selection_method: str               #: Selection method *'Tournament'* or *'Roulette Wheel'*
    population_size: int                #: Number of individuals in the population
    mutation_rate: float                #: Probability of mutating individuals
    elitism_ratio: float                #: Ratio of population for elitism
    max_iterations: int                 #: Number of generations
    rnd: random.Random                  #: Random generator

    def __init__(self, problem: GraphColoringProblem,
                 selection_method: str,
                 population_size: int,
                 mutation_rate: float,
                 elitism_ratio: float,
                 max_iteration: int,
                 seed: int):
        """
        Constructor

        :param problem: Given problem with the graph
        :param selection_method: Selection method *'Tournament'* or *'Roulette Wheel'*
        :param population_size: Number of individuals in the population
        :param mutation_rate: Probability of mutating individuals
        :param elitism_ratio: Ratio of population for elitism
        :param max_iteration: Number of generations
        :param seed: Random seed
        """

        # Store the variables and parameters
        self.problem = problem
        self.graph = self.problem.graph
        self.selection_method = selection_method
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism_ratio = elitism_ratio
        self.max_iteration = max_iteration
        self.rnd = random.Random(seed)

        # Validate the parameters
        assert self.selection_method in ['Tournament', 'Roulette Wheel'], "Invalid selection method."
        assert 0 <= self.elitism_ratio <= 1, f"Invalid elitism ratio {self.elitism_ratio} must be in [0, 1]."
        assert 0 <= self.mutation_rate <= 1, f"Invalid mutation rate {self.mutation_rate} must be in [0, 1]."
        assert self.population_size > 0, f"Invalid population size {self.population_size} must be a positive integer."
        assert self.max_iteration > 0, f"Invalid max. generations {self.max_iteration} must be a positive integer."

    #################################################################
    #this creates a population including totally random solutions in number of population_size
    def create_population(self) -> list:
        population = []
        for i in range(self.population_size):
            population.append(self.problem.random_solution)
        return population

    #this calculates fitness score of a given solution
    def fitness(self, individual: Solution) -> float:
        weight_feasibility_ratio=4
        weight_normalized_objective_ratio=1

        #Feasibility ratio calculation
        count_different_adjacent_nodes = 0
        count_connection = 0
        for node in self.graph:
            for adjacent_node in self.graph[node]:
                count_connection += 1
                if individual[node] != individual[adjacent_node]:
                    count_different_adjacent_nodes += 1
        feasibility_ratio = (count_different_adjacent_nodes / 2) / (count_connection / 2)

        # Normalized objective score ratio calculation
        objective_min = len(self.graph) * min(COLORS.values())
        objective_max = len(self.graph) * max(COLORS.values())
        objective = GraphColoringProblem.objective(individual)
        normalized_objective_ratio = (objective - objective_min) / (objective_max - objective_min)

        #fitness calculation
        fitness = ((weight_feasibility_ratio*feasibility_ratio +
                    weight_normalized_objective_ratio*normalized_objective_ratio)
                    /
                    (weight_feasibility_ratio+weight_normalized_objective_ratio))
        return fitness


    #this selects and return a random solution from the given population
    def selection(self, population: list) -> Solution:

        if self.selection_method == 'Tournament':
            #randomly add 3 individual to the competition
            tournament_group = self.rnd.sample(population, k=3 )
            max_value = self.fitness(tournament_group[0])
            tournament_winner = tournament_group[0]

            for individual in tournament_group:
                fitness_score = self.fitness(individual)
                if max_value < fitness_score:
                    max_value = fitness_score
                    tournament_winner = individual
            return tournament_winner

        elif self.selection_method == 'Roulette Wheel':
            sum_fitness_scores = 0
            probability_list = []

            for individual in population:
                fitness_score = self.fitness(individual)
                sum_fitness_scores += fitness_score
                probability_list.append(fitness_score)

            for i in range(len(population)):
                probability_list[i]=probability_list[i]/sum_fitness_scores

            selection = self.rnd.choices(population, weights=probability_list, k=1)
            return selection[0]

    #this creates a new solution using two parent solution
    def crossover(self, parent1: Solution, parent2: Solution) -> Solution:
        new_individual = {}

        #uniform crossover
        for node in range(len(parent1)):
            if self.rnd.random() < 0.5:
                new_individual[node] = parent1[node]
            else:
                new_individual[node] = parent2[node]
        return new_individual

    #for a given gen (node), if a random value is lower than mutation probability, mutate the gen randomly
    def mutate(self, individual: Solution) -> Solution:
        for node in individual:
            if self.rnd.random() < self.mutation_rate:
                individual[node] = random.choice(list(COLORS.keys()))
        return individual

    def solve(self, discover_state_space: bool) -> Solution:

        population = self.create_population() #population
        #descending order in terms of fitness scores
        population = sorted(population, key=self.fitness, reverse=True)

        last_fitness = 0
        count_same_fitness = 0
        count_restart_population = 0
        best_reached_fitness = 0.0
        best_reached_solution = None

        for i in range(self.max_iteration):
            #put the elites in the future generation
            n_elites = round(self.population_size*self.elitism_ratio)
            new_population = population[:n_elites]

            #create new generation with the size of self.population_size
            while len(new_population) < self.population_size:
                #Choose two different parents and cross them
                while True:
                    parent1 = self.selection(population)
                    parent2 = self.selection(population)
                    if parent1 != parent2:
                        new_individual = self.crossover(parent1, parent2)
                        break
                new_individual=self.mutate(new_individual)
                new_population.append(new_individual)

            population = new_population
            population = sorted(population, key=self.fitness, reverse=True)

            if self.fitness(population[0]) > best_reached_fitness:
                best_reached_fitness = self.fitness(population[0])
                best_reached_solution = population[0]

            if self.fitness(population[0]) == last_fitness:
                count_same_fitness += 1
            else:
                count_same_fitness = 0
            last_fitness = self.fitness(population[0])

            print("iteration: ", i)
            print("Solution: ", population[0])
            print("Feasibility:", self.problem.feasibility(population[0]))
            print("Objective:", self.problem.objective(population[0]))
            print("Fitness Score: ", self.fitness(population[0]))
            print("--------------")
            print("No fitness value change in the last ", count_same_fitness, " iterations")

            # in local optima, initialize a random population to discover other part of state space
            if discover_state_space:
                print("# of times population restarted: ", count_restart_population)
                print("best reached fitness: ", best_reached_fitness)

                if count_restart_population == 5 and count_same_fitness == 25: #if we change population 5 times and it converges, it's enough, finish the operation
                    break
                elif count_same_fitness == 25: #if we have the same fitness value in the last 20 iteration
                    population = self.create_population()
                    population = sorted(population, key=self.fitness, reverse=True)
                    count_restart_population +=1
            else:
                if count_same_fitness == 25:
                    break
            print("")
        return best_reached_solution






















"""
    def fitness(self, solution: Solution) -> float:
        #Penalty ratio calculation
        count_same_adjacent_nodes = 0
        count_connection = 0
        for node in self.graph:
            for adjacent_node in self.graph[node]:
                count_connection += 1
                if solution[node] == solution[adjacent_node]:
                    count_same_adjacent_nodes += 1
        penalty_ratio = (count_same_adjacent_nodes / 2) / (count_connection / 2)
        # Örneğin 0.4 oran demek bağlantıların %40'ı aynı renkte demek.

        # Normalized objective score ratio calculation
        objective_min = len(self.graph)*min(COLORS.values())
        objective_max = len(self.graph)*max(COLORS.values())
        current_objective=GraphColoringProblem.objective(solution)
        normalized_objective_ratio = (current_objective - objective_min) / (objective_max - objective_min)

        #fitness calculation, higher better
        fitness_score = normalized_objective_ratio * (1-penalty_ratio)
        fitness_score_penalty_based = (1-penalty_ratio)
        fitness_penalty_more_important = current_objective/(penalty_ratio+0.001)
        return fitness_penalty_more_important
        
        
    def solve(self) -> Solution:

        # Initialize population
        population = self.create_population()

        for iteration in range(self.max_iteration):
            # Evaluate fitness and sort by highest fitness
            population = sorted(population, key=self.fitness, reverse=True)

            # Elitism: Keep the top performing solutions
            elites = population[:int(self.elitism_ratio * self.population_size)]

            # Generate new population through crossover and mutation
            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent1 = self.selection(population)
                parent2 = self.selection(population)
                offspring = self.crossover(parent1, parent2)
                self.mutate(offspring)
                new_population.append(offspring)

            population = new_population

        # Return the best solution found
        return max(population, key=self.fitness)
"""