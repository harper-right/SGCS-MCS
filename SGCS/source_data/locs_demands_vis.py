from libs.util import get_file_names
from libs.ltsp_info import LTSPInfo
import matplotlib.pyplot as plt
import numpy as np
import os


# Plot the distribution of sites and workers
if __name__ == '__main__':
    # Read in data
    trajectory_name = "mobility"  # "t_drive", "roma"
    file_names = get_file_names(f"./{trajectory_name}", ends=".vrp")

    # res path
    result_dir = f"./vis_locs_demands/{trajectory_name}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for file_name in file_names:
        file_path = os.path.join(f"./{trajectory_name}", file_name)
        ci = LTSPInfo(file_path, inspection_rate=0.3)
        # Treat customers as task sites, with their demand as the task's traffic
        coords = ci.coords
        flow = ci.flow

        # How many tasks are needed if selecting sites greedily by traffic size?
        sorted_indices = np.argsort(flow)[::-1]
        alpha = 0.3
        needed_inspected = int(alpha*sum(flow))
        selected_station_coords = []
        inspected_num = 0
        place = 0
        while inspected_num < needed_inspected:
            station_idx = sorted_indices[place]
            selected_station_coords.append(coords[station_idx])
            inspected_num += flow[station_idx]
            place += 1

        print(f"Selected station number: {len(selected_station_coords)}")
        print(f"selected rate: {len(selected_station_coords)/len(flow)}")

        # Plot the distribution of sites and traffic
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = [coord[0] for coord in coords]
        y = [coord[1] for coord in coords]
        z = flow
        # Plot the traffic of each site as a bar chart
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
            ax.bar3d(x[index], y[index], 0, 1, 1, z[index], alpha=0.9, color=color, edgecolor="grey")

        # Plot the selected sites' positions on the XOY plane
        selected_x = [coord[0] for coord in selected_station_coords]
        selected_y = [coord[1] for coord in selected_station_coords]
        print(f'dimension: {ci.dimension}')
        s = 5000 / ci.dimension
        ax.scatter(selected_x, selected_y, [0] * len(selected_x), color='red', s=s, label='Selected Stations')

        # Mark the starting point
        start_coord = (ci.coords[0])
        ax.scatter([start_coord[0]], [start_coord[1]], [0], color='blue', s=s, alpha=0.5, label='Start')

        # Plot the XOY plane
        # Create grid point coordinates
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x = np.linspace(x_min, x_max, 2)
        y = np.linspace(y_min, y_max, 2)
        X, Y = np.meshgrid(x, y)
        # Define the plane equation, here for the XOY plane (Z=0)
        Z = np.zeros_like(X)

        # Plot the plane
        ax.plot_surface(X, Y, Z, alpha=0.07, edgecolor="black")

        # Set axis labels, ticks, and view angle
        ax.axis("off")
        ax.grid(False)  # Remove grid lines

        # Set view angle
        ax.view_init(elev=70, azim=20)

        # save and show
        fig_name = f"{file_name[:-4]}.png"
        plt.savefig(os.path.join(result_dir, fig_name), bbox_inches='tight')
        # plt.savefig(os.path.join(result_dir, f"{file_name[:-4]}.png"), bbox_inches='tight')
        # plt.show()
