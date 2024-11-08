"""
DO NOT CHANGE THIS Python File
"""

import random
import pickle
from typing import Dict, Set, TypeVar, Optional

import networkx as nx
from matplotlib import pyplot as plt

# Global definition of colors with scores
COLORS = {
    "Blue": 1,
    "Green": 2,
    "Red": 3,
    "Yellow": 4
}

# Type Variables
Graph = TypeVar(bound=Dict[str, Set[int]], name='Graph')
Solution = TypeVar(bound=Dict[str, str], name='Solution')
#Data types to use in variables

class GraphColoringProblem:
    """
        This class involves some helper function for the **Graph Coloring Problem**
    """
    #varName: str #Bu class'da string tipinde varName var diye belirtiyoruz.
    graph: Graph

    def __init__(self, graph: Graph):
        """
        Constructor

        :param graph: Problem graph to store
        """

        self.graph = graph

    def feasibility(self, solution: Solution) -> bool:
        """
        This method validates the feasibility of a given graph and solution.

        :param solution: Solution
        :return: Feasibility
        """
        assert len(self.graph) == len(solution), \
            f"Given solution and graph are inconsistent ({len(solution)} nodes vs. {len(self.graph)} nodes)."

        assert set(self.graph.keys()) == set(solution.keys()), \
            f"Given solution and graph must have the same nodes."

        # Iterate over all nodes in the graph
        for node in self.graph:
            for neighbor in self.graph[node]:
                # Check the adjacency coloring constraint
                if solution[node] == solution[neighbor]:
                    return False

        return True

    @staticmethod
    def objective(solution: Solution) -> float:
        """
        This method evaluates the objective of a given solution.

        :param solution: Solution
        :return: Value of the objective function (i.e., the total score)
        """
        total_score: float = 0

        for node, color in solution.items():
            total_score += COLORS[color]

        return total_score

    def draw(self, solution: Solution, name: Optional[str] = None):
        """
        This method demonstrates the given graph and solution.

        :param solution: Solution
        :param name: Name of the plot, optional.
        """
        # Define the graph
        network_graph = nx.Graph()

        for node in self.graph:
            for neighbor in self.graph[node]:
                network_graph.add_edge(node, neighbor)

        # Define node colors
        colors = [solution[node].lower() for node in network_graph.nodes()]

        # Draw
        pos = nx.spring_layout(network_graph)
        nx.draw(network_graph, pos, with_labels=True,
                node_color=colors,
                edge_color='gray', node_size=500, font_size=12
                )

        if name is not None:
            plt.title(name)

        plt.show()

    @staticmethod
    def generate_map(node_size: int, path: Optional[str] = None, seed: int = 1234) -> Graph:
        """
        This method randomly generates a map with a given node size and random seed parameters

        :param node_size: The number of nodes will be on the graph
        :param path: File path to save the graph as a *Pickle* file
        :param seed: Random seed
        :return: Randomly generated graph
        """
        rnd = random.Random(seed)

        # Define nodes
        graph = {
            node: set()
            for node in range(node_size)
        }

        # Assign colors
        assigned_colors = {
            node: rnd.choice(list(COLORS.keys()))
            for node in range(node_size)
        }

        # Assign edges
        for node in range(node_size):
            # Determine candidate nodes which has different color
            candidates = [candidate
                          for candidate in graph
                          if candidate != node and assigned_colors[candidate] != assigned_colors[node]
                          ]

            # If insufficient number of candidates can be found, regenerate map with different seed.
            if len(candidates) < 2:
                return GraphColoringProblem.generate_map(node_size, path, seed + 1)

            # Remove duplicates
            candidates = list(set(candidates) - graph[node])

            if len(candidates) < 2:  # Prevent redundancy
                continue

            # Randomly determine the number of edges among candidates, at least two edges
            number_of_edges = rnd.randint(2, len(candidates))

            # Assign edges
            adjacency = rnd.choices(candidates, k=number_of_edges)

            graph[node].update(adjacency)

            # Directed edges
            for adjacent in adjacency:
                graph[adjacent].add(node)

        if path is not None:
            # Save the graph via Pickle
            with open(path, 'wb') as f:
                pickle.dump(graph, f)

        return graph

    def save(self, file_path: str):
        """
        This method saves the graph to a given file path as a *Pickle* file.

        :param file_path: Target file path
        """

        # Save the graph via Pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self.graph, f)

    @staticmethod
    def read(file_path: str) -> Graph:
        """
        This function reads the graph from a given file path which is a *Pickle* file.

        :param file_path: Target pickle file path
        :return: Read graph
        """

        # Read the graph via Pickle
        with open(file_path, 'rb') as f:
            graph = pickle.load(f)

            return graph

    @property
    def random_solution(self) -> Solution:
        """
        This method randomly generates a solution.

        **Note**: It does not guarantee that the solution is valid.

        :return: Randomly generated solution, i.e. []
        """
        # Define random generator (object)
        rnd = random.Random()

        # Randomly assign colors (solution is dictionary)
        solution: Solution = {}

        for node in self.graph:
            solution[node] = rnd.choice(list(COLORS.keys()))
            #node: 0, 1, 2, 3...

        return solution

    @property
    def clone(self) -> Graph:
        """
        This method provides a clone of the graph.

        :return: Clone of the graph
        """
        return self.graph.copy()
