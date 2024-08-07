from libs.util import get_alpha_plus
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


# In Section IV, we verify the optimal p_1 defined by Theorem 1
if __name__ == '__main__':
    # game_info
    g_info = GameInfo(os.path.join("./source_data/", "par_game.yaml"))
    g_info.m = 600  # worker number setting
    # index_i
    par_name = 'sec_iv_thm1'
    par_arr = g_info.p1_arr  # horizontal axis
    result_dir = f"./results/{time.strftime('%m_%d_%H_%M')}_{par_name}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # index_j
    app_lst = ['M_minus_1', 'M_0', 'M_1', 'G', 'income', 'payment']
    # index_k
    rep_time = 500
    # save res in arr
    res_arr = np.zeros((rep_time, par_arr.shape[0], len(app_lst)))
    # save p1 in Theorem
    thm_p1_arr = np.zeros((rep_time,))  # Save best_p1
    # experiments
    for k in range(rep_time):
        g_info.omg, g_info.c = g_info.refresh_workers()  # Regenerate the worker group
        # get best p1
        game_eq = GamePDF(g_info)
        b_idx = game_eq.best_idx
        best_p1 = game_eq.p1_arr[b_idx]
        # save best p1 in THEOREM
        thm_p1_arr[k] = best_p1
        for i, par in enumerate(par_arr):
            # initial p_arr
            p_arr = np.array([0.0, 0.0, par])
            alpha_plus = get_alpha_plus(g_info, p_arr)
            # MONTE CARLO
            x_arr = get_worker_response(g_info, p_arr, alpha_plus)
            g, income, payment = get_g(g_info, p_arr, alpha_plus, x_arr)
            # save M
            res_arr[k, i, 0] = np.sum(x_arr == 0)  # M_{-1}
            res_arr[k, i, 1] = np.sum(x_arr == 1)  # M_{0}
            res_arr[k, i, 2] = np.sum(x_arr == 2)  # M_{1}
            # save G
            res_arr[k, i, 3] = g
            res_arr[k, i, 4] = income
            res_arr[k, i, 5] = payment

    # save res in .npy
    print(f"Experiment is done!")
    np.save(os.path.join(result_dir, f'M_and_G.npy'), res_arr)
    # Save the results of theoretical calculations in .xls
    file_name = f"{par_name}_thm_for_plt.xls"  # file_name
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("THEORETICAL")
    sheet.write(0, 0, "best_p1")
    avg_best_p1 = np.average(thm_p1_arr)
    sheet.write(0, 1, float(avg_best_p1))
    workbook.save(os.path.join(result_dir, file_name))

    # CHANGE Array to DataFrame
    rep_time, parameter_num, approach_num = res_arr.shape
    reshaped_arr = res_arr.reshape(parameter_num * rep_time, approach_num)  # shape=(rep_time*a_arr, app)
    df = pd.DataFrame(reshaped_arr, columns=app_lst)
    df[par_name] = np.tile(par_arr, rep_time)  # Add x-axis
    # PLOT
    x = df[par_name].unique()
    y_mean = df.groupby(par_name).mean()
    y_std = df.groupby(par_name).std()
    # plot line
    color_lst = ['#3862A7', '#D9763E', '#5CAB6E', '#C55356']  # 'b', 'o', 'g', 'r'
    fig, ax1 = plt.subplots()
    # Plot a graph of workers' strategy choices
    for j, app in enumerate(app_lst[:3]):
        ax1.plot(x, y_mean[app], color_lst[j], label=app)
        ax1.fill_between(x, y_mean[app] - y_std[app], y_mean[app] + y_std[app], color=color_lst[j], alpha=0.2)
    # label_font
    ax1.set_xlabel(r'$p_1$')
    ax1.set_ylabel(r'M')
    # Draw the theoretical optimal value of p1
    plt.vlines(avg_best_p1, 0, g_info.m + 1, colors='r', label='best_alpha', ls='--')
    # legend_font
    legend = plt.legend(loc='best')
    frame = legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')  # Set the legend background to be transparent
    # grid
    plt.grid()
    # show
    plt.savefig(os.path.join(result_dir, f"p1-M.jpg"), bbox_inches='tight', pad_inches=0.02)
    plt.show()

    # Plot the platform's income and outcome graphs
    fig, ax2 = plt.subplots()
    for j, app in enumerate(app_lst[4:]):
        ax2.plot(x, y_mean[app], color_lst[j], label=app)
        ax2.fill_between(x, y_mean[app] - y_std[app], y_mean[app] + y_std[app], color=color_lst[j], alpha=0.2)
    # label_font
    ax1.set_xlabel(r'$p_1$')
    ax1.set_ylabel(r'Income & Payment')
    # Draw the theoretical optimal value of p1
    plt.vlines(avg_best_p1, 0, np.max(y_mean['income'])*1.01, colors='r', label='best_alpha', ls='--')
    # legend_font
    legend = plt.legend(loc='best')
    frame = legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')  # Set the legend background to be transparent
    # grid
    plt.grid()
    # show
    plt.savefig(os.path.join(result_dir, f"p1-In_Out.jpg"), bbox_inches='tight', pad_inches=0.02)
    plt.show()

    # SAVE in .xls
    file_name = f"{par_name}_for_plt.xls"  # file_name
    workbook = xlwt.Workbook()
    sheet_lst = ["mean", "low_bound", "up_bound"]
    sheet_res_dir = dict()
    sheet_res_dir["mean"] = y_mean.values  # Save the average value
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

    print(f'{par_name} is DONE! :)')
