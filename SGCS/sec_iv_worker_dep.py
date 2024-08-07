import numpy as np
import matplotlib.pyplot as plt
import time
import os
import xlwt


# Section iv
# Simulations of "discussion on worker-dependent verification"
class Workers(object):
    def __init__(self):
        self.num = 30
        self.omg = self.get_par_with_dis(1.0, 5.0, "UNI")  # omega_i
        self.c = np.zeros((self.num, 3))  # c[i] = [ c_{-1}[i], c_{0}[i], c_{1}[i] ]
        self.c[:, 1] = self.get_par_with_dis(0.2, 1.0, "NORM")  # c_0[i]
        self.c[:, 2] = self.get_par_with_dis(2.0, 6.0, "NORM")  # c_1[i]
        self.gamma = 0.5

    def get_par_with_dis(self, a, b, dis_name):
        if dis_name == 'NORM':
            par = np.around(np.random.normal(loc=(a + b) / 2, scale=(b - a) / 6, size=(self.num,)), decimals=2)
            par[par > b] = b
            par[par < a] = a
        elif dis_name == 'EXP':
            par = np.around(np.random.exponential(scale=1, size=(self.num,)), decimals=2)
            par = par + a
            par[par > b] = b
        else:  # 'UNI', and others
            par = np.around(np.random.uniform(low=a, high=b, size=(self.num,)), decimals=2)
        return par

    def get_workers_response(self, alpha_i, p):
        ui = np.zeros(self.c.shape)
        ui[:, 0] = p[0] - self.c[:, 0]
        ui[:, 1] = p[1] - self.c[:, 1] + (1 - alpha_i * (1 + self.omg * self.gamma)) * (p[2] - p[1])
        ui[:, 2] = p[2] - self.c[:, 2]
        x_star = np.argmax(ui, axis=1)
        return ui, x_star

    # Get the alpha_{i}^{+} of each worker under a certain pricing strategy p
    def get_alpha_plus(self, p):
        # If there is a reputation mechanism, the minimum verification rate required
        p1 = p[2]  # Platform rewards for high-quality perception data
        c1 = self.c[:, 2]  # c_1[i]
        lower = c1.copy()
        lower[lower > p1] = p1
        alpha_i_plus = 1 / (1 + self.omg * self.gamma) * (lower - self.c[:, 1]) / p1
        return alpha_i_plus


class Platform(object):
    def __init__(self, workers):
        self.workers = workers
        # Make sure x_i=-1 is not the optimal strategy for the worker; xi is either 0 or 1
        self.p = np.array([0.0, 0.0, np.max(workers.c)+1e-4])
        # The platform records the quality of workers’ historical data
        self.Qt = np.ones(workers.num)  # Qt will adjust according to the quality inspection results.
        self.delta_Q = 0.05  # Qt updates each time
        # Initialize the platform's verification rate for each worker
        self.theta = 0.1
        self.alpha_t = self.get_alpha_t()
        # Total rounds
        self.total_t = 100

    # Count the number of workers that choose strategy xi=1
    @staticmethod
    def get_n1(x_star):
        return np.sum(x_star[:] == 2)

    def get_alpha_t(self):
        alpha_t = self.theta + (1.0 - self.theta) * np.sqrt(1.0 - self.Qt)
        return alpha_t

    # Observe how alpha_t changes as Qt decreases
    def test_for_alpha_t_update(self):
        qt = 1.0
        delta_q = 0.01
        total_t = int(qt / delta_q)
        alp_t_lst = []
        for slot in range(total_t):
            alp_t = self.theta + (1.0 - self.theta) * np.sqrt(1.0 - qt)
            alp_t_lst.append(alp_t)
            qt -= delta_q
        plt.plot(alp_t_lst)
        plt.xlabel("slot")
        plt.ylabel(r"$\alpha_i(t)$")
        plt.show()

    # Worker-Dependent Verification Rates (WDVR) strategy
    def worker_dependent_verification(self):
        good_rto_res = list()  # Record the number of workers who submitted high-quality data in each round
        inspect_rto_res = list()  # Records the number of times the platform inspects workers in each time slice

        for slot in range(self.total_t):
            # Workers respond to the platform's strategy
            ui, x_star = self.workers.get_workers_response(self.alpha_t, self.p)
            # The platform conducts quality inspection of workers’ perception data with a certain probability
            inspect_cnt = 0  # Record the total number of times the platform inspects workers in this round
            for idx in range(self.workers.num):
                if np.random.rand() < self.alpha_t[idx]:
                    inspect_cnt += 1  # Data quality inspection for workers
                    if x_star[idx] == 1:  # If workers submit low-quality data
                        self.Qt[idx] = max(0, self.Qt[idx] - self.delta_Q)  # Lowering Qt
            # According to the test results, update alpha_t
            self.alpha_t = self.get_alpha_t()
            # Utility Indicator for Computing Platforms
            good_ratio = self.get_n1(x_star) / self.workers.num
            inspect_ratio = inspect_cnt / self.workers.num

            # Record the metrics for this round
            good_rto_res.append(good_ratio)
            inspect_rto_res.append(inspect_ratio)

        return np.array(good_rto_res), np.array(inspect_rto_res)

    # Fixed Verification Rates (FVR) strategy
    def fixed_verification(self):
        # 通过公式计算工人的alpha_i^{+}
        alpha_i_plus = self.workers.get_alpha_plus(self.p)
        max_alpha_i_plus = np.max(alpha_i_plus) + 1e-4
        alpha_for_fvr = np.ones(self.workers.num) * max_alpha_i_plus

        # for save
        good_rto_res = list()
        inspect_rto_res = list()

        for slot in range(self.total_t):
            ui, x_star = self.workers.get_workers_response(alpha_for_fvr, self.p)
            inspect_cnt = 0
            for idx in range(self.workers.num):
                if np.random.rand() < max_alpha_i_plus:
                    inspect_cnt += 1
            good_ratio = self.get_n1(x_star) / self.workers.num
            inspect_ratio = inspect_cnt / self.workers.num

            good_rto_res.append(good_ratio)
            inspect_rto_res.append(inspect_ratio)

        return np.array(good_rto_res), np.array(inspect_rto_res)


# Comparison between different strategies, results of repeated experiments
def vis_app_cmp(rep_res, app_list, y_label, fig_name, save_dir):
    avg_res = np.average(np.array(rep_res), axis=0)  # Average of repeated experiments
    plt.figure()
    for app_idx in range(len(app_list)):
        plt.plot(avg_res[app_idx, :], label=app_list[app_idx])
    plt.xlabel('Round')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{fig_name}.pdf"), bbox_inches='tight', pad_inches=0.02)
    plt.show()


if __name__ == "__main__":
    # # test for alpha_t_update
    # worker_entities = Workers()
    # platform = Platform(worker_entities)
    # platform.test_for_alpha_t_update()

    # See the effect of worker-dependent verification
    rep_time = 1000  # default 500

    # for save
    theta_lst = [0.05, 0.1, 0.2]  # for WDVR strategy, [0.05, 0.1, 0.3]
    rep_good_ratio = list()
    rep_inspect_ratio = list()
    rep_opt_inspect_ratio = list()  # Theoretical optimal value

    for rep in range(rep_time):
        # for save
        good_ratio_lst = list()  # shape=(len(app_lst), total_t)
        inspect_ratio_lst = list()  # shape=(len(app_lst), total_t)
        opt_inspect_ratio = list()

        # initialize workers
        worker_entities = Workers()

        # WDVR strategy
        for wdvr_idx in range(len(theta_lst)):
            platform = Platform(worker_entities)
            platform.theta = theta_lst[wdvr_idx]
            platform.alpha_t = platform.get_alpha_t()
            good_rtos, inspect_rtos = platform.worker_dependent_verification()
            # for good_ratio
            good_ratio_lst.append(good_rtos)
            # for inspect_cnt
            inspect_ratio_lst.append(inspect_rtos)
            # for opt_inspect_ratio
            alpha_plus_i = worker_entities.get_alpha_plus(platform.p)
            opt_inspect_ratio.append(np.average(alpha_plus_i))

        # FVR strategy
        platform = Platform(worker_entities)
        good_rtos, inspect_rtos = platform.fixed_verification()
        # save
        good_ratio_lst.append(good_rtos)
        inspect_ratio_lst.append(inspect_rtos)

        # save this rep results
        rep_good_ratio.append(good_ratio_lst)
        rep_inspect_ratio.append(inspect_ratio_lst)
        rep_opt_inspect_ratio.append(opt_inspect_ratio)

        print(f"repeated experiments, [{rep+1}/{rep_time}]")

    # app_lst = [r"WDVR, $\theta=0.1$", r"WDVR, $\theta=0.2$", r"WDVR, $\theta=0.3$", "FVR"]
    app_lst = list()
    for theta in theta_lst:
        app_lst.append(rf"WDVR, $\theta={theta}$")
    app_lst.append("FVR")

    result_dir = f"./results/{time.strftime('%m_%d_%H_%M')}_sec_iv_discuss"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # See how data quality changes over time, indicated by the results of workers' strategy selection
    vis_app_cmp(rep_good_ratio, app_lst, "Good Ratio", "quality-time", result_dir)

    # View the changes in the platform's inspection cost over time
    vis_app_cmp(rep_inspect_ratio, app_lst, "Average Verification Rate", "verification_cost-time", result_dir)

    # SAVE in .xls
    file_name = f"origin_plt_data.xls"  # file_name
    workbook = xlwt.Workbook()
    sheet_lst = ["Good Ratio", "Average Inspect Ratio"]
    sheet_res_dir = dict()
    sheet_res_dir["Good Ratio"] = np.average(np.array(rep_good_ratio), axis=0).T
    sheet_res_dir["Average Inspect Ratio"] = np.average(np.array(rep_inspect_ratio), axis=0).T
    for s, sheet_name in enumerate(sheet_lst):
        res_arr = sheet_res_dir.get(sheet_name)
        print(res_arr.shape)
        sheet = workbook.add_sheet(sheet_name)
        sheet.write(0, 0, "slot")
        time_lst = np.arange(res_arr.shape[0])
        for i, time in enumerate(time_lst):
            sheet.write(i + 1, 0, float(time))
        for j, app_name in enumerate(app_lst):
            sheet.write(0, j + 1, app_name)
        for i in range(res_arr.shape[0]):
            for j in range(res_arr.shape[1]):
                sheet.write(i + 1, j + 1, float(res_arr[i][j]))
        if sheet_name == "Average Inspect Ratio":
            opt_ins_rto = np.average(np.array(rep_opt_inspect_ratio))
            sheet.write(0, len(app_lst) + 1, "Optimal")
            for i in range(len(time_lst)):
                sheet.write(i + 1, len(app_lst) + 1, float(opt_ins_rto))
    workbook.save(os.path.join(result_dir, file_name))
