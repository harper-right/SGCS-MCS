import numpy as np
from libs.ltsp_info import LTSPInfo
from libs.steep_improve_route import steep_improve_route


class UAVVirEa(object):
    def __init__(self, info, max_steep_iter, phi):
        self.info = info
        self.phi = phi  # phi is a rate phi*task_num is the number of tasks to be visited
        self.max_steep_iter = max_steep_iter  # Maximum iterations for the steep improve method
        self.route = self.init_route()
        self.best_solution = self.step()  # best_solution

    # Initialize the path
    def init_route(self):
        route = [self.info.start_node]
        sorted_indices = np.argsort(-self.info.flow)
        to_visit_num = int(self.phi * self.info.dimension)
        for i in range(to_visit_num):
            route.append(sorted_indices[i])
        return route

    def step(self):
        refreshed_route = steep_improve_route(self.route, self.info.dists, self.max_steep_iter)
        solution = self.info.make_solution(refreshed_route)
        return solution


class VirEa(object):
    def __init__(self, ltsp_file, alpha, max_steep_iter, phi):
        self.name = "vir_ea"
        # for uav
        self.alpha = alpha
        ltsp_info = LTSPInfo(ltsp_file, alpha)
        self.uav = UAVVirEa(ltsp_info, max_steep_iter, phi)
        # res
        self.path_len = self.uav.best_solution.cost
        # actual_verification_rate
        self.actual_verification_rate = self.uav.best_solution.inspected / sum(self.uav.info.flow)


if __name__ == "__main__":
    # ltsp_info
    verification_rate = 0.3
    lp_info = LTSPInfo("../source_data/cvrp/uni_norm/n40.vrp", verification_rate)
    # algorithm
    max_stp_iter = 500
    phi_par = 0.2
    uav_ve = UAVVirEa(lp_info, max_stp_iter, phi_par)
    print(f"task number: {lp_info.dimension}")
    print(f'cost: {uav_ve.best_solution.cost:.2f}')
    ac_verification_rate = uav_ve.best_solution.inspected / sum(uav_ve.info.flow)
    print(f"actual verification rate: {ac_verification_rate:.2f}")
    # Plot the path
    uav_ve.info.vis_solution(uav_ve.best_solution, save_path=None, show=True)
