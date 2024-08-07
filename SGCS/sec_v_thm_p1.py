from libs.game_info import GameInfo
from libs.game_monte_carlo import GameMonteCarlo
from libs.game_pdf_eq import GamePDF
import numpy as np
import pandas as pd
import os
import time
import xlwt
import matplotlib.pyplot as plt


# Verification of Theorem 1 of SGCS, fig(a), the correctness of the theoretical solution best_p1,
# under the condition that workers' costs follow different distributions
if __name__ == '__main__':
    # game_info
    g_info = GameInfo(os.path.join("./source_data/", "par_game.yaml"))
    # index_i
    par_name = 'thm_p1'
    par_arr = g_info.p1_arr
    result_dir = f"./results/{time.strftime('%m_%d_%H_%M')}_{par_name}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # index_j
    app_lst = ['NORM', 'EXP', 'UNI']
    # index_k
    rep_time = 500
    # res dic
    res_lst = ['g_arr', 'thm_best_p1', 'thm_best_g']  # thm is for theorem
    res_dir = dict()
    for res_name in res_lst:
        res_dir[res_name] = np.zeros((rep_time, par_arr.shape[0], len(app_lst)))
    # experiments
    for j, app_name in enumerate(app_lst):
        # update C distribution
        dis = app_name
        g_info.c0_dis = dis
        g_info.c1_dis = dis
        for k in range(rep_time):
            g_info.omg, g_info.c = g_info.refresh_workers()  # Regenerate worker groups
            game_monte = GameMonteCarlo(g_info)  # Monte Carlo method
            game_pdf = GamePDF(g_info)  # Formula method
            res_dir["g_arr"][k, :, j] = game_monte.g_arr
            res_dir["thm_best_p1"][k, :, j] = game_pdf.p1_arr[game_pdf.best_idx]
            res_dir["thm_best_g"][k, :, j] = game_pdf.g_arr[game_pdf.best_idx]
            if (k + 1) % 100 == 0:
                print(f"{par_name}_{app_name}_[{k + 1} / {rep_time}]")
    # save res in .npy
    for res_name in res_lst:
        print(f"SAVE res_{res_name} ...")
        res_arr = res_dir.get(res_name)
        np.save(os.path.join(result_dir, f'{res_name}.npy'), res_arr)
    # CHANGE Array to DataFrame
    res_arr = res_dir.get("g_arr")  # shape=[rep_time, p1_arr, app_lst]
    rep_time, parameter_num, approach_num = res_arr.shape
    reshaped_arr = res_arr.reshape(parameter_num * rep_time, approach_num)  # shape=(rep_time*p1_arr, app)
    df = pd.DataFrame(reshaped_arr, columns=app_lst)
    df[par_name] = np.tile(par_arr, rep_time)  # Add x-axis

    # PLOT
    x = df[par_name].unique()
    y_mean = df.groupby(par_name).mean()
    y_std = df.groupby(par_name).std()
    # plot line
    color_lst = ['#3862A7', '#D9763E', '#5CAB6E', '#C55356']  # 'b', 'o', 'g', 'r'
    figure, ax = plt.subplots()
    for j, app in enumerate(app_lst):
        ax.plot(x, y_mean[app], color_lst[j], label=app)
        ax.fill_between(x, y_mean[app] - y_std[app], y_mean[app] + y_std[app], color=color_lst[j], alpha=0.2)
    # get theoretical best_p1, best_g1
    thm_p1_lst = list()
    thm_g_lst = list()
    for j, app_name in enumerate(app_lst):
        res_arr = res_dir.get("thm_best_p1")
        thm_p1_lst.append(np.average(res_arr[:, :, j]))
        res_arr = res_dir.get("thm_best_g")
        thm_g_lst.append(np.average(res_arr[:, :, j]))
    plt.scatter(np.array(thm_p1_lst), np.array(thm_g_lst), c='red', label=r'$p_1^*$'+" in Theorem 1")
    # label_font
    ax.set_xlabel(r'$p_1$')
    ax.set_ylabel("Game profit of the platform (G)", labelpad=-2)
    # legend_font
    legend = plt.legend(loc='best')
    frame = legend.get_frame()
    frame.set_alpha(1)
    # frame.set_facecolor('none')
    # grid
    plt.grid()
    # show
    plt.savefig(os.path.join(result_dir, par_name+".png"), bbox_inches='tight', pad_inches=0.02)
    plt.show()

    # save result in .xls
    # Save line data
    file_name = f"{par_name}_for_plt.xls"  # file_name
    workbook = xlwt.Workbook()
    sheet_lst = ["mean", "low_bound", "up_bound"]
    sheet_res_dir = dict()
    sheet_res_dir["mean"] = y_mean.values  # Save the average value
    sheet_res_dir["low_bound"] = y_mean.values - y_std.values  # lower bound
    sheet_res_dir["up_bound"] = y_mean.values + y_std.values  # upper bound
    for sheet_name in sheet_lst:
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
    # Save the results of theoretical calculations
    sheet = workbook.add_sheet("THEORETICAL")
    for j, app_name in enumerate(app_lst):
        sheet.write(0, j + 1, app_name)
    sheet.write(1, 0, "best_p1")
    for j, best_p1 in enumerate(thm_p1_lst):
        sheet.write(1, j + 1, float(best_p1))
    sheet.write(2, 0, "best_g")
    for j, best_g in enumerate(thm_g_lst):
        sheet.write(2, j + 1, float(best_g))
    workbook.save(os.path.join(result_dir, file_name))
    print(f"{time.strftime('%m_%d, %H:%M')}_{par_name} is DONE! :)")
