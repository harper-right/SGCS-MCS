from libs.game_info import GameInfo
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as st


# Obtain the optimal solution for the Stackelberg Game under any distribution condition,
# using probability density distribution (pdf) and cumulative density distribution (cdf)
class GamePDF(object):
    def __init__(self, game_info):
        self.info = game_info
        self.p1_arr = self.info.p1_arr
        self.dist = self.get_dist()
        self.g_arr, self.in_arr, self.pay_arr, self.n0_arr, self.n1_arr, self.a_star_arr = self.step()
        self.best_idx = self.get_best_idx()

    def get_alpha_star(self, p):
        # Minimum verification rate required if a reputation mechanism is in place
        p1 = p[2]
        c2 = self.info.c[:, 2]
        lower = c2.copy()
        lower[lower > p1] = p1
        alpha_i = 1 / (1 + self.info.omg * self.info.gamma) * (lower - self.info.c[:, 1]) / p1
        alpha_star = np.max(alpha_i)
        # Increase alpha by delta to avoid critical values
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

    def get_worker_omg(self):
        return self.info.omg


if __name__ == "__main__":
    # game_info
    parameter_file = "par_game.yaml"
    info = GameInfo(os.path.join("../data_cvrp/", parameter_file))
    # pdf solve equation for stackelberg game
    game_eq = GamePDF(info)
    b_idx = game_eq.best_idx
    print(f"p1*: {game_eq.p1_arr[b_idx]:.4f}, G*: {game_eq.g_arr[b_idx]:.2f}")
    print(f"a*: {game_eq.a_star_arr[b_idx]:.2f}\n")
    # Plot experimental results, G vs p1
    plt.plot(game_eq.p1_arr, game_eq.g_arr)
    plt.xlabel(r'$p_1$')
    plt.ylabel(r'$G$')
    plt.title(f'Theoretical, p1*: {game_eq.p1_arr[b_idx]:.2f}, G*: {game_eq.g_arr[b_idx]:.2f}')
    plt.show()
