import random
from libs.ltsp_info import LTSPInfo
from libs.steep_improve_route import steep_improve_route


# Randomly select sites until the "needed_inspection" requirement is met.
# After selecting the sites, use the Steep Improve Algorithm in the TSP_2_Opt to plan the TSP path.
class UAVRandom(object):
    def __init__(self, info, max_steep_iter):
        self.info = info
        self.max_steep_iter = max_steep_iter  # The maximum number of iterations for the "steep improve" method
        self.route = self.random_init()  # Initialize path
        self.best_solution = self.step()

    def random_init(self):
        inspected = 0
        route = [self.info.start_node]
        unserviced = list(range(self.info.dimension))
        del unserviced[unserviced.index(self.info.start_node)]
        while inspected < self.info.needed_inspection:
            node = random.choice(unserviced)
            inspected += self.info.flow[node]
            route.append(node)
            del unserviced[unserviced.index(node)]
        return route

    # "steep_improve_route" solves the Traveling Salesman Problem (TSP).
    def step(self):
        self.route = steep_improve_route(self.route, self.info.dists, self.max_steep_iter)
        best_solution = self.info.make_solution(self.route)
        return best_solution


# Reputation + Rand
class RepRand(object):
    def __init__(self, ltsp_file, alpha, max_steep_iter):
        self.name = 'rep_rand'
        # for uav
        self.alpha = alpha
        ltsp_info = LTSPInfo(ltsp_file, alpha)
        self.uav = UAVRandom(ltsp_info, max_steep_iter)
        self.path_len = self.uav.best_solution.cost


if __name__ == "__main__":
    # ltsp_info
    verification_rate = 0.3  # alpha
    lp_info = LTSPInfo("../source_data/cvrp/uni_norm/n40.vrp", verification_rate)
    # rand_algorithm
    max_stp_iter = 500
    uav_rand = UAVRandom(lp_info, max_stp_iter)
    print(f"task number: {lp_info.dimension}")
    print(f'cost: {uav_rand.best_solution.cost:.2f}')
    # Plot the path
    uav_rand.info.vis_solution(uav_rand.best_solution, save_path=None, show=True)
