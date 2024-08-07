from libs.game_info import GameInfo
from libs.game_pdf_eq import GamePDF
import numpy as np
import os
import matplotlib.pyplot as plt


# Using Monte Carlo method, we can obtain the optimal solution of Stackelberg Game.
class GameMonteCarlo(object):
    def __init__(self, game_info):
        self.info = game_info
        self.p1_arr = self.info.p1_arr
        self.g_arr, self.in_arr, self.pay_arr, self.n0_arr, self.n1_arr, self.a_star_arr = self.step()
        self.best_idx = self.get_best_idx()

    def get_alpha_star(self, p):
        # If there is a reputation mechanism, the minimum verification rate required
        p1 = p[2]
        c2 = self.info.c[:, 2]
        lower = c2.copy()
        lower[lower > p1] = p1
        alpha_i = 1 / (1 + self.info.omg * self.info.gamma) * (lower - self.info.c[:, 1]) / p1
        alpha_star = np.max(alpha_i)
        # Add delta to alpha to avoid critical values
        alpha_star += 1e-6
        return alpha_star

    def get_worker_response(self, p, alpha):
        ui = np.zeros(self.info.c.shape)
        ui[:, 0] = p[0] - self.info.c[:, 0]
        ui[:, 1] = p[1] - self.info.c[:, 1] + (1 - alpha * (1 + self.info.omg * self.info.gamma)) * (p[2] - p[1])
        ui[:, 2] = p[2] - self.info.c[:, 2]
        x_star = np.argmax(ui, axis=1)
        return x_star

    def get_g(self, p, alpha, x_star):
        v_sum = 0.0
        p_x_sum = 0.0
        p_y_sum = 0.0
        y_lst = [0, 2, 2]
        for idx in range(x_star.shape[0]):
            x = x_star[idx]
            y = y_lst[x]
            v_sum += self.info.v[x]
            p_x_sum += p[x]
            p_y_sum += p[y]
        income = self.info.xi * np.log(v_sum + 1)
        payment = alpha * p_x_sum + (1 - alpha) * p_y_sum
        g = income - payment
        return g, income, payment

    def step(self):
        # save for res
        g_arr = np.zeros(self.p1_arr.shape)
        in_arr = np.zeros(g_arr.shape)
        pay_arr = np.zeros(g_arr.shape)
        n0_arr = np.zeros(g_arr.shape, dtype='int64')
        n1_arr = np.zeros(g_arr.shape, dtype='int64')
        a_star_arr = np.zeros(g_arr.shape)
        # Initialize the platform's pricing strategy
        p = np.array([0.0, 0.0, 0.0])
        for i in range(self.p1_arr.shape[0]):
            # PLATFORM's strategy, including pricing strategy and testing strategy
            p[2] = self.p1_arr[i]
            alpha = self.get_alpha_star(p)
            a_star_arr[i] = alpha
            # WORKER's response
            x_star = self.get_worker_response(p, alpha)
            # Calculate the data profit G brought by the current platform's strategy (p1, alpha)
            g_arr[i], in_arr[i], pay_arr[i] = self.get_g(p, alpha, x_star)
            # n
            n0_arr[i] = np.sum(x_star == 1)
            n1_arr[i] = np.sum(x_star == 2)
        return g_arr, in_arr, pay_arr, n0_arr, n1_arr, a_star_arr

    def get_best_idx(self):
        best_idx = np.argmax(self.g_arr)
        return best_idx


if __name__ == "__main__":
    # game_info
    parameter_file = "par_game.yaml"
    info = GameInfo(os.path.join("../source_data/", parameter_file))
    # pdf for stackelberg game
    game_eq = GamePDF(info)
    b_idx = game_eq.best_idx
    print('PDF Equation: ...')
    print(f"p1*: {game_eq.p1_arr[b_idx]:.4f}, n1*: {game_eq.n1_arr[b_idx]:.2f}")
    print(f"a*: {game_eq.a_star_arr[b_idx]:.2f}\n")
    # Plotting experimental results, G vs p1
    plt.plot(game_eq.p1_arr, game_eq.g_arr)
    plt.xlabel(r'$p_1$')
    plt.ylabel(r'$G$')
    plt.title(f'Theoretical, p1*: {game_eq.p1_arr[b_idx]:.2f}, n1*: {game_eq.n1_arr[b_idx]:.2f}')
    plt.show()

    # monte carlo for stackelberg game, repeat experiments
    rep_time = 500
    best_p1_arr = np.zeros((rep_time,))
    best_n1_arr = np.zeros((rep_time,), dtype='int64')
    print('Monte Carlo for best_p1, best_n1: ...')
    for rep in range(rep_time):
        info.omg, info.c = info.refresh_workers()
        game = GameMonteCarlo(info)
        best_p1_arr[rep] = game.p1_arr[game.best_idx]
        best_n1_arr[rep] = game.n1_arr[game.best_idx]
        if (rep + 1) % 100 == 0:
            print(f"[{rep+1} / {rep_time}] ...")
    print('Monte Carlo is DONE :)')
    avg_best_p1, avg_best_n1 = np.average(best_p1_arr), np.average(best_n1_arr)

    # Draw Monte Carlo experiment results, best_p1, best_n1
    # best_p1
    plt.plot(best_p1_arr, label='monte-carlo')
    plt.hlines(avg_best_p1, 1, rep_time, colors='r', label='best_p1_eq')
    plt.xlabel('rep_time')
    plt.title(f'p1_monte-carlo, avg p1: {avg_best_p1:.2f}')
    plt.show()
    # best_n1
    plt.plot(best_n1_arr, label='monte-carlo')
    plt.hlines(avg_best_n1, 1, rep_time, colors='r', label='avg_n1_eq')
    plt.xlabel('rep_time')
    plt.title(f'n1_monte-carlo, avg n1: {avg_best_n1:.2f}')
    plt.show()
