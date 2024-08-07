from libs.game_info import GameInfo
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import scipy.stats as st
import time
import os


# Verification of Theorem 2 of SGCS, fig(b),
# under the constraint of alpha_plus, the worker's strategic behavior has the lowest benefit
class WorkerUtility(object):
    def __init__(self, game_info):
        self.info = game_info
        self.info.m = 150  # set worker number for clearer scatter plot; too many dots will overlap
        self.info.omg, self.info.c = self.info.refresh_workers()
        self.p1_arr = np.linspace(self.info.a, self.info.b, num=100)
        self.dist = self.get_dist()
        self.g_arr, self.in_arr, self.pay_arr, self.n0_arr, self.n1_arr, self.a_star_arr = self.step()
        self.best_idx = self.get_best_idx()
        self.ui, self.x_star = self.get_worker_utility()

    def get_alpha_star(self, p):
        # If there is a reputation mechanism, the minimum verification rate required
        p1 = p[2]
        c2 = self.info.c[:, 2]
        lower = c2.copy()
        lower[lower > p1] = p1
        alpha_i = 1 / (1 + self.info.omg * self.info.gamma) * (lower - self.info.c[:, 1]) / p1
        alpha_star = np.max(alpha_i)
        alpha_star += 1e-6
        return alpha_star

    def get_dist(self):
        # Exponential distribution
        if self.info.c1_dis == 'EXP':
            dist = st.expon(loc=self.info.a, scale=0.4)
        # Normal distribution
        elif self.info.c1_dis == 'NORM':
            mean = (self.info.b + self.info.a) / 2
            std = (self.info.b - self.info.a) / 6
            dist = st.norm(loc=mean, scale=std)
        # Uniform distribution
        else:
            dist = st.uniform(loc=self.info.a, scale=(self.info.b - self.info.a))
        return dist

    def get_g(self, p):
        v1 = self.info.v[2]
        p1 = p[2]
        n1 = self.dist.cdf(p1) * self.info.m
        income = self.info.xi * np.log(n1 * v1 + 1)
        payment = n1 * p1
        g = income - payment
        return g, income, payment, n1

    def step(self):
        g_arr = np.zeros(self.p1_arr.shape)
        in_arr = np.zeros(g_arr.shape)
        pay_arr = np.zeros(g_arr.shape)
        n0_arr = np.zeros(g_arr.shape, dtype='int64')
        n1_arr = np.zeros(g_arr.shape, dtype='int64')
        a_star_arr = np.zeros(g_arr.shape)
        p = np.array([0.0, 0.0, 0.0])
        for i in range(self.p1_arr.shape[0]):
            p[2] = self.p1_arr[i]
            a_star_arr[i] = self.get_alpha_star(p)
            g_arr[i], in_arr[i], pay_arr[i], n1_arr[i] = self.get_g(p)
        return g_arr, in_arr, pay_arr, n0_arr, n1_arr, a_star_arr

    def get_best_idx(self):
        best_idx = np.argmax(self.g_arr)
        return best_idx

    def get_worker_utility(self):
        best_p1 = self.p1_arr[self.best_idx]
        p = np.array([0.0, 0.0, best_p1])
        best_alpha = self.a_star_arr[self.best_idx]
        ui = np.zeros(self.info.c.shape)
        ui[:, 0] = p[0] - self.info.c[:, 0]
        ui[:, 1] = p[1] - self.info.c[:, 1] + (1 - best_alpha * (1 + self.info.omg * self.info.gamma)) * (p[2] - p[1])
        ui[:, 2] = p[2] - self.info.c[:, 2]
        x_star = np.argmax(ui, axis=1)
        return ui, x_star


if __name__ == "__main__":
    # get workers
    game_par_path = "./source_data/par_game.yaml"
    info = GameInfo(game_par_path)
    workers = WorkerUtility(info)
    ui_arr = workers.ui
    c_arr = workers.info.c

    # res path
    par_name = 'thm_workers'
    result_dir = f"./results/{time.strftime('%m_%d_%H_%M')}_{par_name}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Plot
    x = c_arr[:, 1]
    y = c_arr[:, 2]
    app_lst = [r"$x_i=-1$", r"$x_i=0$", r"$x_i=1$"]
    fig = plt.figure()
    # Define a coordinate system
    ax = fig.add_subplot(projection='3d')
    for j, app in enumerate(app_lst):
        ax.scatter(x, y, ui_arr[:, j], label=app)
    plt.legend(loc=(0.66, 0.26))
    # adjust the image perspective first
    ax.view_init(elev=10, azim=-30)
    ax.set_facecolor('white')
    # Adjust the distance between the label and the scale axis
    ax.set_xlabel(r"$c_0$", labelpad=-7)
    ax.set_ylabel(r"$c_1$", labelpad=-5)
    ax.set_zlabel("Each worker's utility", labelpad=-7)
    # Set the distance between the axis and the scale
    plt.tick_params(axis='x', pad=-5, labelsize=8)
    plt.tick_params(axis='y', pad=-5, labelsize=8)
    plt.tick_params(axis='z', pad=-3, labelsize=8)
    # Set the x-axis scale change interval
    x_major_locator = MultipleLocator(0.2)
    ax.xaxis.set_major_locator(x_major_locator)
    # # Set the y-axis scale change interval
    # ax.yaxis.set_ticks([0.3, 0.35, 0.5, 0.75, 0.9])
    # save
    fig_name = 'Theorem_c' + '.pdf'
    plt.savefig(os.path.join(result_dir, fig_name), dpi=1200)
    # Display the image
    plt.show()
