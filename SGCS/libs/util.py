import os
import numpy as np


# Get the name of each folder in the directory
def get_folder_names(path):
    folder_names = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            folder_names.append(folder)
    return folder_names


# Get the name of each file in the directory
# If ends are specified (e.g., ".vrp", ".txt"), get only filenames ending with those extensions
def get_file_names(path, ends=None):
    file_names = []  # List to store filenames
    for file_name in os.listdir(path):
        if ends is None:
            file_names.append(file_name)
        else:
            if file_name.endswith(ends):
                file_names.append(file_name)
    return file_names


# for TSP
def read_tsp_data(data_file):
    with open(data_file, 'r') as file:
        lines = file.readlines()
    # Find the line number where node coordinates start
    node_coord_start = lines.index("NODE_COORD_SECTION\n") + 1
    # Find the line number where node coordinates end
    node_coord_end = lines.index("EOF\n")
    # Extract node coordinates
    coords_lst = []
    for line in lines[node_coord_start:node_coord_end]:
        _, x, y = line.strip().split()
        coords_lst.append((float(x), float(y)))
    dimension = len(coords_lst)
    return dimension, coords_lst


# for CVRP
def read_cvrp_data(data_file):
    with open(data_file) as f:
        content = [line.rstrip("\n") for line in f.readlines()]
    dimension = int(content[3].split()[-1])
    print(content[5].split()[-1])
    capacity = int(content[5].split()[-1])
    coords = [(-1, -1) for _ in range(dimension)]
    demand = [-1 for _ in range(dimension)]
    # Line number where node coordinates start
    node_coord_start = 7
    for i in range(node_coord_start, dimension + node_coord_start):
        nid, xc, yc = [float(x) for x in content[i].split()]
        coords[int(nid)-1] = (xc, yc)
    # Line number where node demands start
    node_demand_start = node_coord_start + dimension + 1
    for i in range(node_demand_start, dimension + node_demand_start):
        nid, dem = [int(x) for x in content[i].split()]
        demand[nid-1] = dem
    return dimension, capacity, coords, demand


# for ltsp
def read_ltsp_data(data_file):
    with open(data_file) as f:
        content = [line.rstrip("\n") for line in f.readlines()]
    dimension = int(content[3].split()[-1])
    capacity = int(content[5].split()[-1])
    coords = [(-1, -1) for _ in range(dimension)]
    demand = [-1 for _ in range(dimension)]
    # Line number where node coordinates start
    node_coord_start = 7
    for i in range(node_coord_start, dimension + node_coord_start):
        nid, xc, yc = [float(x) for x in content[i].split()]
        coords[int(nid) - 1] = (xc, yc)
    # Line number where node demands start
    node_demand_start = node_coord_start + dimension + 1
    for i in range(node_demand_start, dimension + node_demand_start):
        nid, dem = [int(x) for x in content[i].split()]
        demand[nid - 1] = dem
    return dimension, capacity, coords, np.array(demand)


# FIXME for Stackelberg Game
# Given worker information `game_info` and platform pricing strategy `p`,
# compute the critical value of the verification rate $\alpha^+$
def get_alpha_plus(game_info, p):
    # Minimum verification rate required if a reputation mechanism is in place
    p1 = p[2]
    c2 = game_info.c[:, 2]
    lower = c2.copy()
    lower[lower > p1] = p1
    alpha_i = 1 / (1 + game_info.omg * game_info.gamma) * (lower - game_info.c[:, 1]) / p1
    alpha_plus = np.max(alpha_i)
    # Increase alpha by delta to avoid hitting the critical value
    alpha_plus += 1e-6
    return alpha_plus


# Workers' response when the platform provides pricing `p` and verification rate `alpha`
def get_worker_response(info, p, alpha):
    ui = np.zeros(info.c.shape)
    ui[:, 0] = p[0] - info.c[:, 0]
    ui[:, 1] = p[1] - info.c[:, 1] + (1 - alpha * (1 + info.omg * info.gamma)) * (p[2] - p[1])
    ui[:, 2] = p[2] - info.c[:, 2]
    x_star = np.argmax(ui, axis=1)
    return x_star


# Platform's data profit $G$ under the strategy $(p, \alpha)$
def get_g(info, p, alpha, x_star):
    v_sum = 0.0
    p_x_sum = 0.0
    p_y_sum = 0.0
    y_lst = [0, 2, 2]
    for idx in range(x_star.shape[0]):
        x = x_star[idx]
        y = y_lst[x]
        v_sum += info.v[x]
        p_x_sum += p[x]
        p_y_sum += p[y]
    if v_sum > 0:
        log_v_sum = np.log(v_sum + 1)
    else:
        log_v_sum = -1.0 * np.log(1 - v_sum)
    income = info.xi * log_v_sum
    payment = alpha * p_x_sum + (1 - alpha) * p_y_sum
    g = income - payment
    return g, income, payment
