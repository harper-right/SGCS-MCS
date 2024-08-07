from libs.ltsp_info import LTSPInfo
from libs.steep_improve_route import steep_improve_route
import random
import numpy as np


# Flow_Greedy starts from the origin,
# selects stations in descending order of passenger flow until the needed_inspection is satisfied
# After selecting the stations, use the Steep Improve Algorithm in TSP_2_Opt to plan the TSP path
class UAVFlow(object):
    def __init__(self, info, max_steep_iter, mutation_rate):
        self.info = info
        self.max_steep_iter = max_steep_iter  # Maximum iterations for the steep improve method
        self.mutation_rate = mutation_rate
        self.route = self.flow_greedy_init()
        self.best_solution = self.step()

    def flow_greedy_init(self):
        inspected = 0
        route = [self.info.start_node]  # From start point
        unserviced = list(np.argsort(self.info.flow))
        unserviced = unserviced[::-1]
        del unserviced[unserviced.index(self.info.start_node)]
        while inspected < self.info.needed_inspection:
            next_sel = unserviced[0]
            if random.random() > (1 - self.mutation_rate):  # With a certain probability, randomly
                next_sel = random.choice(unserviced)
            inspected += self.info.flow[next_sel]
            route.append(next_sel)
            del unserviced[unserviced.index(next_sel)]
        return route

    # the steep_improve_route is needed to solve the TSP problem
    def step(self):
        self.route = steep_improve_route(self.route, self.info.dists, self.max_steep_iter)
        best_solution = self.info.make_solution(self.route)
        return best_solution


# Reputation + Flow_Greedy under the specified inspection rate alpha
class RepFlow(object):
    def __init__(self, ltsp_file, alpha, max_steep_iter, mutation_rate):
        self.name = 'rep_flow'
        # for uav
        self.alpha = alpha  # Determine the number of workers that need to be inspected
        ltsp_info = LTSPInfo(ltsp_file, alpha)
        self.uav = UAVFlow(ltsp_info, max_steep_iter, mutation_rate)
        # res
        self.path_len = self.uav.best_solution.cost


if __name__ == "__main__":
    # ltsp_info
    verification_rate = 0.3  # alpha
    lp_info = LTSPInfo("../source_data/cvrp/uni_norm/n40.vrp", verification_rate)
    # flow_greedy_algorithm
    max_stp_iter = 500
    mut_rate = 0.0
    uav_flow = UAVFlow(lp_info, max_stp_iter, mut_rate)
    print(f"task number: {lp_info.dimension}")
    print(f'cost: {uav_flow.best_solution.cost:.2f}')
    # Plot the path
    uav_flow.info.vis_solution(uav_flow.best_solution, save_path=None, show=True)
