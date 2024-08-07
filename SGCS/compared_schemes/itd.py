import numpy as np
from libs.util import read_ltsp_data
from libs.ltsp_info import LTSPInfo
from libs.steep_improve_route import steep_improve_route
import random
import time


class UAVItd(object):
    def __init__(self, ltsp_file, big_theta, population_size, survival_rate, mutation_rate, max_steep_iter):
        self.dimension, _, _, _ = read_ltsp_data(ltsp_file)
        self.big_theta = big_theta
        self.alpha = self.big_theta / self.dimension
        self.info = LTSPInfo(ltsp_file, self.alpha)
        self.population_size = population_size  # Number of individuals in the population
        self.survival_rate = survival_rate  # Number of individuals to retain
        self.mutation_rate = mutation_rate  # Mutation probability
        self.max_steep_iter = max_steep_iter  # Maximum number of iterations for the steep improve method
        self.routes = self.random_init()
        self.costs, self.is_valids, self.fitness = self.refresh_info(current_iter=0)
        self.best_solution = self.info.make_solution(self.routes[0])  # Initialize `best_solution`

    # Initialize the path
    def random_init(self):
        routes = []
        for _ in range(self.population_size):
            inspected = 0
            result = [self.info.start_node]
            unserviced = list(range(self.info.dimension))
            del unserviced[unserviced.index(self.info.start_node)]
            while inspected < self.info.needed_inspection:
                node = random.choice(unserviced)
                inspected += self.info.flow[node]
                result.append(node)
                del unserviced[unserviced.index(node)]
            result = steep_improve_route(result, self.info.dists, self.max_steep_iter)  # FIXME 10.23的初始化没有这一步
            routes.append(result)
        return routes

    # Update the population's costs, is_valids, fitness, etc.
    # The calculation of penalties in fitness requires `current_iter`
    def refresh_info(self, current_iter):
        # Calculate the cost, validity, and fitness of each path
        costs = []
        is_valids = []
        fitness_lst = []
        for i, route in enumerate(self.routes):
            costs.append(self.info.compute_cost(route))
            inspected, is_valid = self.info.compute_inspected(route)
            is_valids.append(is_valid)
            fitness = self.compute_fitness(costs[i], inspected, current_iter)
            fitness_lst.append(fitness)
        return costs, is_valids, fitness_lst

    # fitness=cost+penalty, penalty=function(inspected, current_iter)
    def compute_fitness(self, cost, inspected, current_iter):
        penalty = 0
        if inspected < self.info.needed_inspection:
            rate = (self.info.needed_inspection - inspected) / self.info.needed_inspection
            # Calculate the average distance between two points to standardize the dimensions of penalty and cost
            cost_factor = np.average(self.info.dists)
            penalty = cost_factor*current_iter*rate
        fitness = cost + penalty
        return fitness

    # Sort by fitness and select the top `survival_rate` individuals as the ones with reproductive capability
    def choose_survivals(self):
        sort_index = np.argsort(np.array(self.fitness)).copy()
        selected_index = sort_index[0:int(self.survival_rate * len(sort_index))]
        parents = []
        parents_fitness = []
        for index in selected_index:
            parents.append(self.routes[index])
            parents_fitness.append(self.fitness[index])
        return parents, parents_fitness

    # `choose_couple` is used to select two individuals for mating
    # First, use the genetic algorithm to select sites
    # Then, apply the Steep Improve Method to optimize the TSP problem
    # Use single-point crossover to produce new individuals
    @staticmethod
    def choose_couple(parents, parents_fitness):
        fitness = 1.0 / np.array(parents_fitness)
        sum_score = sum(fitness)
        score_ratio = [sub * 1.0 / sum_score for sub in fitness]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        index1 = 0
        index2 = 0
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(parents[index1]), list(parents[index2])

    # Single-point crossover, suitable for crossing between two individuals with different gene lengths
    @staticmethod
    def one_point_crossover(x, y):
        x_t = [t for t in range(1, len(x))]
        x_point = random.choice(x_t)
        x_cross = x[x_point:]
        y_t = [t for t in range(1, len(y))]
        y_point = random.choice(y_t)
        y_cross = y[y_point:]
        # cross
        x1 = x[:x_point]
        new_x = x1 + y_cross
        y1 = y[:y_point]
        new_y = y1 + x_cross
        # Genes may be duplicated
        new_x = set(new_x)
        new_x = list(new_x)
        new_y = set(new_y)
        new_y = list(new_y)
        return new_x, new_y

    # Mutation
    # Randomly select a position and add a new site from uncovered sites
    def mutation(self, gene):
        unserviced = [t for t in range(self.info.dimension)]
        for i in gene:
            del unserviced[unserviced.index(i)]
        if len(unserviced) > 0:
            node_sel = random.choice(unserviced)
            gene.append(node_sel)
        return list(gene)

    def step(self, current_iter):
        # Select a subset of top individuals as candidate parents
        parents, parents_fitness = self.choose_survivals()
        # Update `best_solution`
        cur_best_solution = self.info.make_solution(parents[0])
        if cur_best_solution.cost < self.best_solution.cost and cur_best_solution.is_valid:
            self.best_solution = cur_best_solution
        # Generate new population routes
        new_routes = parents.copy()
        # Generate a new population
        inner_iter = 0  # If unable to generate valid individuals, stop crossover after max_inner_iter
        max_inner_iter = 10000
        while len(new_routes) < self.population_size and inner_iter < max_inner_iter:
            # Select parents using roulette wheel method
            gene_x, gene_y = self.choose_couple(parents, parents_fitness)
            # Crossover
            gene_x_new, gene_y_new = self.one_point_crossover(gene_x, gene_y)
            # Mutate
            if np.random.rand() < self.mutation_rate:
                gene_x_new = self.mutation(gene_x_new)
            if np.random.rand() < self.mutation_rate:
                gene_y_new = self.mutation(gene_y_new)
            # Use SIA to solve TSP sub-problem
            gene_x_new = steep_improve_route(gene_x_new, self.info.dists, self.max_steep_iter)
            gene_y_new = steep_improve_route(gene_y_new, self.info.dists, self.max_steep_iter)
            # Generate new solution
            x_s = self.info.make_solution(gene_x_new)
            y_s = self.info.make_solution(gene_y_new)
            # Calculate fitness = cost + penalty
            x_fitness = self.compute_fitness(x_s.cost, x_s.inspected, current_iter)
            y_fitness = self.compute_fitness(y_s.cost, y_s.inspected, current_iter)
            # Insert the better individuals into the population
            if x_fitness < y_fitness and (gene_x_new not in new_routes) and x_s.is_valid:
                new_routes.append(gene_x_new)
            elif x_fitness >= y_fitness and (gene_y_new not in new_routes) and y_s.is_valid:
                new_routes.append(gene_y_new)
            inner_iter += 1
        if len(new_routes) < self.population_size:
            while len(new_routes) < self.population_size:
                # If unable to generate valid individuals, copy parent individuals to maintain population size
                route_sel = random.choice(parents)
                new_routes.append(route_sel)
        self.routes = new_routes
        self.costs, self.is_valids, self.fitness = self.refresh_info(current_iter)
        return self.best_solution


class ITD(object):
    def __init__(self, ltsp_file, big_theta, pop_size, sur_rate, mut_rate, max_stp_iter, num_iters):
        self.name = "itd"
        self.uav = UAVItd(ltsp_file, big_theta, pop_size, sur_rate, mut_rate, max_stp_iter)
        self.num_iters = num_iters
        # for save results
        self.costs_arr = np.zeros((self.num_iters,))  # Record the path_len value after each iteration
        # train
        self.train()
        # res
        self.path_len = self.uav.best_solution.cost
        # actual_verification_rate
        self.actual_verification_rate = self.uav.best_solution.inspected / sum(self.uav.info.flow)

    def train(self):
        current_iter = 0
        count = 0
        while current_iter < self.num_iters:
            sol = self.uav.step(current_iter)
            self.costs_arr[current_iter] = sol.cost
            print(f"ITD_Iter: [{current_iter + 1}/{self.num_iters}] path_len: {sol.cost:.2f}")
            if current_iter > 100:
                if np.absolute(self.costs_arr[current_iter] - self.costs_arr[current_iter - 1]) < 1e-4:
                    count += 1
            else:
                count = 0
            if count > 100:
                break
            current_iter += 1


if __name__ == "__main__":
    ltsp_file_path = "../source_data/cvrp/uni_norm/n40.vrp"
    # algorithm
    theta_par = 10
    g_pop_size = 50
    g_sur_rate = 0.8  # The larger this value, the slower the population update
    g_mut_rate = 0.05
    g_max_stp_iter = 500
    uav_itd = UAVItd(ltsp_file_path, theta_par, g_pop_size, g_sur_rate, g_mut_rate, g_max_stp_iter)
    # train
    num_of_iters = 500
    cur_iter = 0
    while cur_iter < num_of_iters:
        start_time = time.time()
        solution = uav_itd.step(cur_iter)
        end_time = time.time()
        d = end_time - start_time
        print(f"Iter: [{cur_iter + 1}/{num_of_iters}] path_len: {solution.cost:.2f}, {d:.2f} s")
        cur_iter += 1
    print(f'cost: {uav_itd.best_solution.cost:.2f}')
    ac_verification_rate = uav_itd.best_solution.inspected / sum(uav_itd.info.flow)
    print(f"actual verification rate: {ac_verification_rate:.4f}")
    # Plot the experimental path
    uav_itd.info.vis_solution(uav_itd.best_solution, save_path=None, show=True)
