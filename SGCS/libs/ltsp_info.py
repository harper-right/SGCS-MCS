from libs.util import *
import random
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


class Solution(object):
    def __init__(self, route=None, cost=0, inspected=0, is_valid=False):
        self.route = route
        if route is None:
            self.route = []  # List type to store the order of nodes visited by the UAV
        self.cost = cost  # the total path length flown by the UAV
        self.inspected = inspected  # Number of workers inspected by the UAV
        self.is_valid = is_valid


class LTSPInfo(object):
    def __init__(self, data_file, inspection_rate):
        # Number of task, task coordinates (including start), `coords`, and traffic flow of each task
        self.dimension, self.capacity, self.coords, self.flow = read_ltsp_data(data_file)
        self.dists = self.compute_dists()  # List recording the pairwise distances between tasks
        self.start_node = 0  # Starting point of the UAV's route
        self.needed_inspection = int(inspection_rate*sum(self.flow))
        random.seed()

    def compute_dists(self):
        coords_arr = np.array(self.coords)
        distances = distance.pdist(coords_arr)
        dist_matrix = distance.squareform(distances)
        dists = [row for row in dist_matrix]
        return dists

    # Calculate the min and max x and y values of a route to obtain a bounding box
    def bounding_box(self, route):
        x_min = min(self.coords[node][0] for node in route)
        x_max = max(self.coords[node][0] for node in route)
        y_min = min(self.coords[node][1] for node in route)
        y_max = max(self.coords[node][1] for node in route)
        return x_min, x_max, y_min, y_max

    def make_solution(self, route):
        if route[0] != self.start_node:
            return None
        cost = 0
        inspected = 0
        is_valid = False
        for i in range(0, len(route)):
            n1, n2 = route[i], route[(i+1) % len(route)]
            cost += self.dists[n1][n2]
            inspected += self.flow[n2]
        if inspected >= self.needed_inspection:
            is_valid = True
        solution = Solution(route=route, cost=cost, inspected=inspected, is_valid=is_valid)
        return solution

    # Calculate the flight path length of the route
    def compute_cost(self, route):
        cost = 0
        for i in range(len(route)):
            n1, n2 = route[i], route[(i + 1) % len(route)]
            cost += self.dists[n1][n2]
        return cost

    # Calculate the number of workers the UAV can inspect along the route
    def compute_inspected(self, route):
        inspected = 0
        is_valid = True
        for i in range(len(route)-1):
            n2 = route[i+1]
            inspected += self.flow[n2]
        if inspected < self.needed_inspection:
            is_valid = False
        return inspected, is_valid

    # Plot the route map, highlight the selected sites, and display the flight path
    def vis_solution(self, solution, save_path=None, show=False):
        route = solution.route
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # First, plot the traffic at each task
        x = [coord[0] for coord in self.coords]
        y = [coord[1] for coord in self.coords]
        z = self.flow
        # Sort the traffic and calculate the index ranges for each percentile group
        sorted_indices = np.argsort(z)
        total = len(sorted_indices)
        quarter = int(total * 0.25)
        half = int(total * 0.5)
        three_quarters = int(total * 0.75)
        # Generate color array
        colors = ['skyblue', 'lightgreen', 'salmon', 'darkorange']
        for i in range(total):
            index = sorted_indices[i]
            if i < quarter:
                color = colors[0]
            elif i < half:
                color = colors[1]
            elif i < three_quarters:
                color = colors[2]
            else:
                color = colors[3]
            ax.bar3d(x[index], y[index], 0, 1, 1, z[index], alpha=0.9, color=color, edgecolor='gray')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('')
        ax.set_zticks([])  # Remove Z-axis tick labels
        ax.w_zaxis.line.set_visible(False)  # Remove Z-axis axis line
        ax.grid(False)  # Remove the grids
        # Set view angle
        ax.view_init(elev=70, azim=60)

        # Plot the positions of selected sites on the XOY plane
        selected_station_coords = [self.coords[i] for i in route]
        selected_x = [coord[0] for coord in selected_station_coords]
        selected_y = [coord[1] for coord in selected_station_coords]
        selected_z = [0] * len(selected_x)
        s = 5000 / self.dimension
        ax.scatter(selected_x[1:], selected_y[1:], selected_z[1:], color='red', s=s, label='Selected Stations')
        # Represent the starting point in purple
        ax.scatter([selected_x[0]], [selected_y[0]], [0], color='blue', s=s, alpha=0.5, label='Start')

        # Plot the path
        # Return to the starting point
        selected_x.append(selected_x[0])
        selected_y.append(selected_y[0])
        ax.plot(selected_x, selected_y, color='black', linewidth=1, linestyle='-')
        ax.set_title(f"Best Route, cost: {solution.cost:.2f}, valid: {solution.is_valid}")

        # Plot the XOY plane
        # Create grid point coordinates
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x = np.linspace(x_min, x_max, 2)
        y = np.linspace(y_min, y_max, 2)
        x_plane, y_plane = np.meshgrid(x, y)
        # Define the plane equation for the XOY plane (Z=0)
        z_plane = np.zeros_like(x_plane)

        # Plot the plane
        ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.1, edgecolor="black")

        # Set axes, tick labels, and view angle
        ax.axis("off")
        ax.grid(False)

        # Save the figure.
        if save_path:
            plt.savefig(os.path.join(save_path, "route.png"))
        if show:
            plt.show()
        plt.close()


if __name__ == "__main__":
    alpha = 0.3
    task_num = 80
    li = LTSPInfo(f"../source_data/cvrp/uni_norm/n{task_num}.vrp", alpha)
    print(f"Needed inspected number: {li.needed_inspection}")
    plt.hist(li.flow)
    plt.show()
