import numpy as np
from libs.ltsp_info import LTSPInfo
from libs.steep_improve_route import steep_improve_route


# binary search and greedy algorithm for UAV baseline data collection, with max_path constraint.
class UAVUitde(object):
    def __init__(self, info, max_steep_iter, uav_max_path):
        self.info = info
        self.max_steep_iter = max_steep_iter
        self.max_path = uav_max_path
        self.sorted_indices = np.argsort(-self.info.flow)
        self.best_solution = self.step()

    # Use steep_improve_route
    def can_verify(self, mid):
        current_to_visit_lst = [0]
        can_verify_bool = False
        for i in range(mid + 1):
            task_index = self.sorted_indices[i]
            current_to_visit_lst.append(task_index)
        refreshed_route = steep_improve_route(current_to_visit_lst, self.info.dists, self.max_steep_iter)
        route_len = self.info.compute_cost(refreshed_route)
        if route_len <= self.max_path:
            can_verify_bool = True
        return can_verify_bool, refreshed_route

    # max_verified_workers
    def step(self):
        task_num = len(self.info.coords) - 1
        low, high = 0, task_num - 1
        max_verified_workers = 0
        best_solution = self.info.make_solution([0])
        while low <= high:
            mid = (low + high) // 2
            can_verify_bool, trajectory = self.can_verify(mid)
            if can_verify_bool:
                solution = self.info.make_solution(trajectory)
                if solution.inspected > max_verified_workers:
                    best_solution = solution
                low = mid + 1
            else:
                high = mid - 1
        return best_solution


class UITDE(object):
    def __init__(self, ltsp_file, alpha, max_steep_iter, uav_max_path):
        self.name = 'uitde'
        # for uav
        self.alpha = alpha
        ltsp_info = LTSPInfo(ltsp_file, alpha)
        self.uav = UAVUitde(ltsp_info, max_steep_iter, uav_max_path)
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
    max_path = 200
    uav_uitde = UAVUitde(lp_info, max_stp_iter, max_path)
    print(f"task number: {lp_info.dimension}")
    print(f'cost: {uav_uitde.best_solution.cost:.2f}')
    ac_verification_rate = uav_uitde.best_solution.inspected / sum(uav_uitde.info.flow)
    print(f"actual verification rate: {ac_verification_rate:.2f}")
    # Plot the path
    uav_uitde.info.vis_solution(uav_uitde.best_solution, save_path=None, show=True)
