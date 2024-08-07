from libs.ltsp_info import LTSPInfo
import random
import math


# LocationGreedy starts from the starting point, and each time selects the node with the nearest next hop distance,
# until the "needed_inspection" requirement is satisfied.
class UAVLoc(object):
    def __init__(self, info, mutation_rate):
        self.info = info
        self.mutation_rate = mutation_rate
        self.route = self.location_greedy_init()
        self.best_solution = self.step()

    def location_greedy_init(self):
        inspected = 0
        route = [self.info.start_node]  # Starting from the starting point
        unserviced = list(range(self.info.dimension))
        del unserviced[unserviced.index(self.info.start_node)]
        while inspected < self.info.needed_inspection:
            next_sel = unserviced[0]
            if random.random() < (1 - self.mutation_rate):
                cur_node = route[-1]
                tmp_dis = math.inf
                for node in unserviced:
                    if self.info.dists[cur_node][node] < tmp_dis:
                        next_sel = node
                        tmp_dis = self.info.dists[cur_node][node]
            else:
                next_sel = random.choice(unserviced)
            inspected += self.info.flow[next_sel]
            route.append(next_sel)
            del unserviced[unserviced.index(next_sel)]
        return route

    # One-step point selection and path planning, directly return the result
    def step(self):
        self.best_solution = self.info.make_solution(self.route)
        return self.best_solution


# Reputation + Location_Greedy
class RepLoc(object):
    def __init__(self, ltsp_file, alpha, mutation_rate):
        self.name = 'rep_loc'
        # for uav
        self.alpha = alpha
        ltsp_info = LTSPInfo(ltsp_file, alpha)
        self.uav = UAVLoc(ltsp_info, mutation_rate)
        self.path_len = self.uav.best_solution.cost


if __name__ == "__main__":
    # ltsp_info
    verification_rate = 0.3  # alpha
    lp_info = LTSPInfo("../source_data/cvrp/uni_norm/n40.vrp", verification_rate)
    # location_greedy_algorithm
    mut_rate = 0
    uav_loc = UAVLoc(lp_info, mut_rate)
    # step
    best_solution = uav_loc.step()
    print(f"task number: {lp_info.dimension}")
    print(f'cost: {uav_loc.best_solution.cost:.2f}')
    # Plot the path
    uav_loc.info.vis_solution(uav_loc.best_solution, save_path=None, show=True)
