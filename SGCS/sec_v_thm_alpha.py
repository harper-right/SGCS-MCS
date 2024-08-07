from libs.util import get_worker_response
from libs.util import get_g
from libs.game_info import GameInfo
from libs.game_pdf_eq import GamePDF
from uav_apps.r_gene import RepGene
import numpy as np
import os
import time
import xlwt
import yaml


# Verification of Theorem 3 of SGCS, fig(c), alpha-G-H
if __name__ == '__main__':
    # game_info
    g_info = GameInfo(os.path.join("./source_data/", "par_game.yaml"))
    # uav parameters
    uav_par_file = os.path.join("./source_data/", "par_uav.yaml")
    with open(uav_par_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # global
    eta_par = float(config['global'].get('eta'))
    # gene
    pop = int(config['gene'].get('pop_size'))
    sur = float(config['gene'].get('sur_rate'))
    mut = float(config['gene'].get('mut_rate'))
    stp_iters = int(config['gene'].get('max_stp_iter'))
    iters = int(config['gene'].get('num_iters'))

    # index_i
    par_name = 'thm_alpha'  # alpha-G, alpha-H
    par_arr = np.round(np.arange(0.06, 0.5, 0.06), 2)  # alpha pars
    # index_k
    rep_time = 5
    # res_lst
    res_lst = ['G', 'H']
    res_dir = dict()
    for res_name in res_lst:
        res_dir[res_name] = np.zeros((rep_time, par_arr.shape[0]))
    alpha_plus_res = np.zeros(rep_time)
    # save result path
    result_dir = f"./results/{time.strftime('%m_%d_%H_%M')}_{par_name}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for k in range(rep_time):
        # To get alpha-G(game profit), using monte-carlo
        game_rep_time = 100
        alpha_arr = par_arr.copy()
        g_arr = np.zeros((game_rep_time, alpha_arr.shape[0]))
        for rep in range(game_rep_time):
            g_info.omg, g_info.c = g_info.refresh_workers()  # Regenerate the worker group
            # get best p1
            game_eq = GamePDF(g_info)
            b_idx = game_eq.best_idx
            best_p1 = game_eq.p1_arr[b_idx]
            # initial p_arr
            p_arr = np.array([0.0, 0.0, best_p1])
            # MONTE CARLO
            for i, alpha in enumerate(alpha_arr):
                x_arr = get_worker_response(g_info, p_arr, alpha)
                g_arr[rep, i], _, _ = get_g(g_info, p_arr, alpha, x_arr)
        avg_g_arr = np.average(g_arr, axis=0)
        idx = np.argmax(avg_g_arr)
        alpha_plus = alpha_arr[idx]
        alpha_plus_res[k] = alpha_plus
        # Creating the dictionary
        alpha_g_dict = {alpha: g_value for alpha, g_value in zip(alpha_arr, avg_g_arr)}

        # To get alpha-H(verification cost)
        set_name = "t_drive"
        data_base = f"./source_data/{set_name}/"
        file_name = "n60.vrp"  # task_num is the default 60
        ltsp_path = os.path.join(data_base, file_name)
        for i, par in enumerate(par_arr):
            print(f'[{k + 1}/{rep_time}]_alpha_{par:.2f}___________')
            sgcs_app = RepGene(ltsp_path, par, pop, sur, mut, stp_iters, iters)
            res_dir["G"][k, i] = alpha_g_dict.get(par)
            res_dir['H'][k, i] = sgcs_app.path_len * eta_par
    # save res in .npy
    for res_name in res_lst:
        print(f"SAVE res_{res_name} ...")
        res_arr = res_dir.get(res_name)
        np.save(os.path.join(result_dir, f'{res_name}.npy'), res_arr)
    # get average results
    for res_name in res_lst:
        res_arr = res_dir.get(res_name)
        res_dir[res_name] = np.average(res_arr, axis=0)
        res_arr = res_dir.get(res_name)

    # save average results in .xls
    file_name = f"{par_name}_for_plt.xls"
    workbook = xlwt.Workbook()
    for res_name in res_lst:
        sheet_name = res_name
        sheet = workbook.add_sheet(sheet_name)
        res_arr = res_dir.get(res_name)
        sheet.write(0, 0, par_name)
        for i, par in enumerate(par_arr):
            sheet.write(i + 1, 0, float(par))
        sheet.write(0, 1, res_name)
        for i in range(res_arr.shape[0]):
            sheet.write(i + 1, 1, float(res_arr[i]))
    sheet_name = "alpha_plus"
    sheet = workbook.add_sheet(sheet_name)
    sheet.write(0, 0, "alpha_plus")
    sheet.write(1, 0, float(np.average(alpha_plus_res)))
    workbook.save(os.path.join(result_dir, file_name))

    print(f"{time.strftime('%m_%d, %H:%M')}_{par_name} is DONE! :)")
