# Copyright © 2021 Giovanni Squillero <squillero@polito.it>
# Free for personal or classroom use; see 'LICENCE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
from math import sqrt
from typing import Any
import numpy as np
import numpy.random as random
import networkx as nx
import matplotlib.pyplot as plt


NUM_CITIES = 35
STEADY_STATE = NUM_CITIES*10
MUTATION_FACTOR = 100 - NUM_CITIES
LAMBDA = 50 #surviving individuals
SIGMA = 200 #offspring individuals at each tweak
MAX_STEPS = 5000

class Tsp:

    def __init__(self, num_cities: int, seed: Any = None) -> None:
        if seed is None:
            seed = num_cities
        self._num_cities = num_cities
        self._graph = nx.DiGraph()
        np.random.seed(seed)
        for c in range(num_cities):
            self._graph.add_node(c, pos=(np.random.random(), np.random.random()))

    def distance(self, n1, n2) -> int:
        pos1 = self._graph.nodes[n1]['pos']
        pos2 = self._graph.nodes[n2]['pos']
        return round(1_000_000 / self._num_cities * sqrt((pos1[0] - pos2[0])**2 +
                                                         (pos1[1] - pos2[1])**2))

    def evaluate_solution(self, solutionr: np.array) -> float:
        best_cost = 100000000000 #maxint
        best_pos = 0
        costs = np.array(range(LAMBDA))
        solution = np.copy(solutionr)
        for index1 in range(LAMBDA-1):
            total_cost = 0
            
            
            tmp = solution[index1].tolist() + [solution[index1,0]]

            for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):

                total_cost += self.distance(n1, n2)

            costs[index1] = total_cost

            if total_cost < best_cost:
                
                best_cost = total_cost
                best_pos = index1
        return best_cost, best_pos, costs

    def evaluate_individual(self, solution: np.array) -> float:
        total_cost = 0
        tmp = solution.tolist() + [solution[0]]
        for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
            total_cost += self.distance(n1, n2)
        return total_cost

    def plot(self, path: np.array = None) -> None:
        if path is not None:
            self._graph.remove_edges_from(list(self._graph.edges))
            tmp = path.tolist() + [path[0]]
            for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
                self._graph.add_edge(n1, n2)
        plt.figure(figsize=(12, 5))
        nx.draw(self._graph,
                pos=nx.get_node_attributes(self._graph, 'pos'),
                with_labels=True,
                node_color='pink')
        if path is not None:
            plt.title(f"Current path: {self.evaluate_individual(path):,}")
        plt.show()

    @property
    def graph(self) -> nx.digraph:
        return self._graph


def tweak_CC(problem,solution, solution_pos, solution_costs):
    new_solution = solution.copy()
    #new_costs = solution_costs.copy() #keep parents
    new_costs = np.ones((LAMBDA))*10000000 #don't keep parents
    #cycle crossover
    for i in range(SIGMA):
        parent1_i = random.randint(0, LAMBDA)
        parent2_i = random.randint(0, LAMBDA)

        parent1 = solution[parent1_i]
        parent2 = solution[parent2_i]

        

        locus1 = random.randint(0, NUM_CITIES)
        locus2 = random.randint(0, NUM_CITIES)

        #we need locus2 > locus1
        if locus2 < locus1:
            tmp = locus2
            locus2 = locus1
            locus1 = tmp



        offspring = parent1.copy()

        #values of parent2 not present in parent1[locus1:locus2] (ordered)
        
        to_be_copied = parent2[np.isin(parent2,parent1[locus1:locus2+1], invert= True)]

        #with a certain probability we apply a minor mutation
        if(random.randint(0,100)>MUTATION_FACTOR):
            np.random.shuffle(to_be_copied)

        #copy into offspring
        offspring[0:locus1] = to_be_copied[0:locus1]
        offspring[locus2+1:] = to_be_copied[locus1 : ]

        #if offspring is better than any of the current individual in the new solution, replace
        offspring_cost = problem.evaluate_individual(offspring)
        worse_costs_indexes = np.nonzero(new_costs>offspring_cost) #this returns a tuple for some reason

        if worse_costs_indexes[0].size !=0:
            
            victim_tmp = random.randint(0, worse_costs_indexes[0].size)
            victim_index = worse_costs_indexes[0][victim_tmp]

            new_solution[victim_index] = offspring
            new_costs[victim_index] = offspring_cost
        elif random.randint(0,100)>MUTATION_FACTOR:
            #otherwise we kill a random individual with a certain probability
            victim_index = random.randint(0, LAMBDA)

            new_solution[victim_index] = offspring
            new_costs[victim_index] = offspring_cost

    return new_solution



def tweak2(solution: np.array, *, pm: float = .1) -> np.array:
    new_solution = solution.copy()
    
    p = None
    while p is None or p < pm:
        i1 = random.randint(0, solution.shape[0])
        i2 = random.randint(0, solution.shape[0])
        temp = new_solution[i1]
        new_solution[i1] = new_solution[i2]
        new_solution[i2] = temp
        p = np.random.random()
    return new_solution


def execute(num_cities):

    NUM_CITIES=num_cities
    STEADY_STATE=10*num_cities
    MUTATION_FACTOR = 100 - NUM_CITIES*2

    problem = Tsp(NUM_CITIES)
    asd = [range(NUM_CITIES)]
    solution = [np.array(range(NUM_CITIES))]
    for i in range(LAMBDA-1):
        solution= np.append(solution, asd, 0)
        np.random.shuffle(solution[i])
    
    solution_cost, solution_pos, solution_costs = problem.evaluate_solution(solution)
    
    if __name__ == '__main__':
        problem.plot(solution[solution_pos])
        print(f"Current path: {solution_cost}")

    history = [(0, solution_cost)]
    steady_state = 0
    step = 0
    while steady_state < STEADY_STATE:
        step += 1
        steady_state += 1
        new_solution = tweak_CC(problem, solution, solution_pos, solution_costs)
        new_solution_cost, new_solution_pos, new_solution_costs = problem.evaluate_solution(new_solution)
        solution = new_solution
        solution_costs = new_solution_costs
        if new_solution_cost < solution_cost:
            solution_cost = new_solution_cost
            solution_pos = new_solution_pos
            history.append((step, solution_cost))
            steady_state = 0
        if step > MAX_STEPS:
            break
    if __name__ == '__main__':
        problem.plot(solution[solution_pos])
        print(f"Current path: {solution_cost}, number of steps: {step}")
    
    return solution_cost
    


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().setLevel(level=logging.INFO)
    execute(NUM_CITIES)