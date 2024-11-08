from GeneticSolver import GeneticSolver
from GraphColoringProblem import * #The class and its methods are imported

if __name__ == '__main__':
    # Define parameters
    FILE_PATH = "dataset5.pkl" #graph data file
    RANDOM_SEED = 342347345345  # Define a random seed to regenerate consistent outcomes

    # Define the most promising values for each parameter after hyperparameter tuning.

    BEST_SELECTION_METHOD ="Roulette Wheel"  # 'Tournament' or 'Roulette Wheel'

    if BEST_SELECTION_METHOD == "Tournament":
        BEST_POPULATION_SIZE = 20
        BEST_MUTATION_RATE = 0.04
        BEST_ELITISM_RATIO = 0.1
        BEST_MAX_ITERATIONS = 1000
        DISCOVER_STATE_SPACE = True
    elif BEST_SELECTION_METHOD == "Roulette Wheel":
        BEST_POPULATION_SIZE = 60
        BEST_MUTATION_RATE = 0.04
        BEST_ELITISM_RATIO = 0.2
        BEST_MAX_ITERATIONS = 500
        DISCOVER_STATE_SPACE = False

    # Define the problem
    graph = GraphColoringProblem.read(FILE_PATH) #Static method
    #graph = GraphColoringProblem.generate_map(node_size=5, path=None, seed=1234)
    problem = GraphColoringProblem(graph)

    # Solve via Genetic Algorithm
    solver = GeneticSolver(problem,
                           selection_method=BEST_SELECTION_METHOD,
                           population_size=BEST_POPULATION_SIZE,
                           mutation_rate=BEST_MUTATION_RATE,
                           elitism_ratio=BEST_ELITISM_RATIO,
                           max_iteration=BEST_MAX_ITERATIONS,
                           seed=RANDOM_SEED)

    solution = solver.solve(discover_state_space=DISCOVER_STATE_SPACE)
    print("")
    print("Solution: ", solution)
    print("Feasibility:", problem.feasibility(solution))
    print("Objective:", problem.objective(solution))
    print("Fitness Score: ", solver.fitness(solution))
    print("")
    problem.draw(solution, name="Graph Coloring Solution")