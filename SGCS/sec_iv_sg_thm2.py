from libs.util import get_worker_response
from libs.util import get_g
from libs.game_info import GameInfo
from libs.game_pdf_eq import GamePDF
import numpy as np
import pandas as pd
import os
import time
import xlwt
import matplotlib.pyplot as plt


# In Section IV, we verify the minimum valid inspection rate alpha_plus defined by Theorem 2.
# Verify the correctness of Theorem 2's solution to alpha_plus,
# and the necessity of the platform to monitor the quality of sensing data.
if __name__ == '__main__':
    # game_info
    g_info = GameInfo(os.path.join("./source_data/", "par_game.yaml"))
    g_info.m = 600  # set worker number
    # index_i
    par_name = 'sec_iv_thm2'
    par_arr = np.linspace(0.0, 1.0, num=101)  # alpha_arr
    result_dir = f"./results/{time.strftime('%m_%d_%H_%M')}_{par_name}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # index_j
    app_lst = ['g', 'm_minus_1', 'm0', 'm1']  # m_m1 is m_minus_1
    # index_k
    rep_time = 100
    # save res in arr
    res_arr = np.zeros((rep_time, par_arr.shape[0], len(app_lst)))
    thm_name = ["best_alpha", "best_g"]
    thm_res_arr = np.zeros((rep_time, len(thm_name)))
    # experiments
    for k in range(rep_time):
        g_info.omg, g_info.c = g_info.refresh_workers()  # Regenerate the worker group
        # get best p1
        game_eq = GamePDF(g_info)
        b_idx = game_eq.best_idx
        best_p1 = game_eq.p1_arr[b_idx]
        # initial p_arr
        p_arr = np.array([0.0, 0.0, best_p1])
        # best alpha in THEOREM
        thm_res_arr[k, 0] = game_eq.a_star_arr[b_idx]
        thm_res_arr[k, 1] = game_eq.g_arr[b_idx]
        # MONTE CARLO
        for i, alpha in enumerate(par_arr):
            x_arr = get_worker_response(g_info, p_arr, alpha)
            res_arr[k, i, 0], _, _ = get_g(g_info, p_arr, alpha, x_arr)
            res_arr[k, i, 1] = np.sum(x_arr == 0)  # M_{-1}
            res_arr[k, i, 2] = np.sum(x_arr == 1)  # M_{0}
            res_arr[k, i, 3] = np.sum(x_arr == 2)  # M_{1}
    # save res in .npy
    print(f"Experiment is done!")
    np.save(os.path.join(result_dir, f'M_and_G.npy'), res_arr)
    # get theoretical best_p1, best_g1
    best_alpha = np.average(thm_res_arr[:, 0])
    best_g = np.average(thm_res_arr[:, 1])
    # Save the results of theoretical calculations in .xls
    file_name = f"{par_name}_thm_for_plt.xls"  # file_name
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("THEORETICAL")
    sheet.write(0, 0, "best_alpha")
    sheet.write(0, 1, float(best_alpha))
    sheet.write(1, 0, "best_g")
    sheet.write(1, 1, float(best_g))
    workbook.save(os.path.join(result_dir, file_name))

    # Save the results of the line data in .xls
    # CHANGE Array to DataFrame
    rep_time, parameter_num, approach_num = res_arr.shape
    reshaped_arr = res_arr.reshape(parameter_num * rep_time, approach_num)  # shape=(rep_time*a_arr, app)
    df = pd.DataFrame(reshaped_arr, columns=app_lst)
    df[par_name] = np.tile(par_arr, rep_time) # Add x-axis
    # PLOT
    x = df[par_name].unique()
    y_mean = df.groupby(par_name).mean()
    y_std = df.groupby(par_name).std()
    # plot line
    color_lst = ['#3862A7', '#D9763E', '#5CAB6E', '#C55356']  # 'b', 'o', 'g', 'r'
    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()  # Clone the first subplot's settings and share the X axis
    # Plot a graph of workers' strategy choices
    for j, app in enumerate(app_lst[1:]):
        ax1.plot(x, y_mean[app], color_lst[j], label=app)
        ax1.fill_between(x, y_mean[app] - y_std[app], y_mean[app] + y_std[app], color=color_lst[j], alpha=0.2)
    # label_font
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'M')
    # # Draw a graph of the platform’s revenue
    # app = "g"
    # ax2.plot(x, y_mean[app], color_lst[-1], label=app)
    # ax2.fill_between(x, y_mean[app] - y_std[app], y_mean[app] + y_std[app], color=color_lst[-1], alpha=0.2)
    # Plotting the theoretical optimal value of alpha
    plt.vlines(best_alpha, 0, g_info.m + 1, colors='r', label='best_alpha', ls='--')
    # legend_font
    legend = plt.legend(loc='best')
    frame = legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')  # Set the legend background to be transparent
    # grid
    plt.grid()
    # show
    plt.savefig(os.path.join(result_dir, f"{par_name}.jpg"), bbox_inches='tight', pad_inches=0.02)
    plt.show()
    # SAVE in .xls
    file_name = f"{par_name}_for_plt.xls"  # file_name
    workbook = xlwt.Workbook()
    sheet_lst = ["mean", "low_bound", "up_bound"]
    sheet_res_dir = dict()
    sheet_res_dir["mean"] = y_mean.values  # Save average value
    sheet_res_dir["low_bound"] = y_mean.values - y_std.values  # Save the lower bound
    sheet_res_dir["up_bound"] = y_mean.values + y_std.values  # Save the upper bound
    for s, sheet_name in enumerate(sheet_lst):
        res_arr = sheet_res_dir.get(sheet_name)
        sheet = workbook.add_sheet(sheet_name)
        sheet.write(0, 0, par_name)
        for i, par in enumerate(par_arr):
            sheet.write(i + 1, 0, float(par))
        for j, app_name in enumerate(app_lst):
            sheet.write(0, j + 1, app_name)
        for i in range(res_arr.shape[0]):
            for j in range(res_arr.shape[1]):
                sheet.write(i + 1, j + 1, float(res_arr[i][j]))
    workbook.save(os.path.join(result_dir, file_name))
    print(f"The alpha^+ {best_alpha}")
    print(f'{par_name} is DONE! :)')
