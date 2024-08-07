import yaml
import os
import numpy as np


# Read Stackelberg Game parameters from a .yaml file
class GameInfo(object):
    def __init__(self, parameter_path):
        with open(parameter_path) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        self.m = int(self.config['game'].get('m'))
        self.xi = float(self.config['game'].get('xi'))
        self.gamma = float(self.config['game'].get('gamma'))
        self.v0 = float(self.config['game'].get('v0'))
        self.v1 = float(self.config['game'].get('v1'))
        self.v = np.array([0, self.v0, self.v1])
        self.min_omg = float(self.config['game'].get('min_omg'))
        self.max_omg = float(self.config['game'].get('max_omg'))
        self.min_c0 = float(self.config['game'].get('min_c0'))
        self.max_c0 = float(self.config['game'].get('max_c0'))
        self.a = float(self.config['game'].get('a'))  # a is min_c1
        self.b = float(self.config['game'].get('b'))  # b is max_c1
        self.omg_dis = self.config['game'].get('omg_dis')
        self.c0_dis = self.config['game'].get('c0_dis')
        self.c1_dis = self.config['game'].get('c1_dis')
        self.omg, self.c = self.refresh_workers()
        # self.task_dim = int(self.config['game'].get('task_dim'))
        self.p1_arr = np.linspace(self.a, self.b, num=100)  # Brute force test to find the optimal value for p1

    def get_par_with_dis(self, dis, a, b):
        if dis == 'NORM':
            par = np.around(np.random.normal(loc=(a + b) / 2, scale=(b - a) / 6, size=(self.m,)), decimals=2)
            par[par > b] = b
            par[par < a] = a
        elif dis == 'EXP':
            par = np.around(np.random.exponential(scale=0.4, size=(self.m,)), decimals=2)
            par = par + a
            par[par > b] = b
        else:  # 'UNI', and others
            par = np.around(np.random.uniform(low=a, high=b, size=(self.m,)), decimals=2)
        return par

    # Generate workers' omega
    def refresh_omg(self):
        omg = self.get_par_with_dis(self.omg_dis, self.min_omg, self.max_omg)
        return omg

    # Generate workers' cost
    def refresh_c(self):
        c1 = self.get_par_with_dis(self.c0_dis, self.min_c0, self.max_c0)
        c2 = self.get_par_with_dis(self.c1_dis, self.a, self.b)
        c = np.zeros((self.m, 3))
        c[:, 1] = c1
        c[:, 2] = c2
        return c

    # Generate worker population
    def refresh_workers(self):
        omg = self.refresh_omg()
        c = self.refresh_c()
        return omg, c


if __name__ == "__main__":
    # game_info
    parameter_file = "par_game.yaml"
    par_path = os.path.join("../source_data/", parameter_file)
    info = GameInfo(par_path)
    print(f'worker_num: {info.m}')
