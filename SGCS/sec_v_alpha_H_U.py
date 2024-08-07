from libs.util import get_worker_response
from libs.util import get_g
from libs.game_info import GameInfo
from libs.game_pdf_eq import GamePDF
from uav_apps.r_rand import RepRand
from uav_apps.r_loc import RepLoc
from uav_apps.r_flow import RepFlow
from uav_apps.r_gene import RepGene
import numpy as np
import os
import time
import xlwt
import yaml


# Validate the correctness of the SGCS framework under different baseline data collection algorithms
# Variation of H (Verification cost) and U (Utility) with respect to alpha
if __name__ == '__main__':
    # game_info
    g_info = GameInfo(os.path.join("./source_data/", "par_game.yaml"))
    # uav parameters
    uav_par_file = os.path.join("./source_data/", "par_uav.yaml")
    with open(uav_par_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # global
    eta_par = float(config['global'].get('eta'))
    # rand
    max_stp_iter = int(config['rand'].get('max_stp_iter'))
    # loc
    mut_rate = float(config['loc'].get('mut_rate'))
    # flow
    # mut_rate = float(config['flow'].get('mut_rate'))
    # max_stp_iter = int(config['flow'].get('max_stp_iter'))
    # gene
    pop = int(config['gene'].get('pop_size'))
    sur = float(config['gene'].get('sur_rate'))
    mut = float(config['gene'].get('mut_rate'))
    stp_iters = int(config['gene'].get('max_stp_iter'))
    iters = int(config['gene'].get('num_iters'))

    # index_i
    par_name = 'alpha_H_U'  # Verification rate
    par_arr = np.round(np.arange(0.03, 0.5, 0.03), 2)  # The value of the verification rate, horizontal axis
    # index_j
    app_lst = ['r_rand', 'r_loc', 'r_flow', 'r_gene']
    # index_k
    rep_time = 5
    # trajectory dataset name
    data_sets_lst = ["t_drive", "roma", "mobility"]
    task_num = 60  # default setting

    # index_set, each set has a separate folder to save the results
    for set_idx, set_name in enumerate(data_sets_lst):
        # station distribution dataset
        data_base = "./source_data/" + f"{set_name}/"
        # save result path
        result_dir = f"./results/{time.strftime('%m_%d_%H_%M')}_{par_name}_{set_name}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # To get alpha-G(game profit), using monte-carlo
        game_rep_time = 100
        alpha_arr = par_arr.copy()  # alpha_arr
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
        # Creating the dictionary
        alpha_g_dict = {alpha: g_value for alpha, g_value in zip(alpha_arr, avg_g_arr)}

        # To get alpha-H(verification cost)
        ltsp_path = os.path.join(data_base, f"n{task_num}.vrp")
        # res dic
        res_lst = ["path_len", "H", "U"]
        res_dir = dict()
        for res_name in res_lst:
            res_dir[res_name] = np.zeros((rep_time, par_arr.shape[0], len(app_lst)))
        # experiments
        for i, par in enumerate(par_arr):  # alpha
            for k in range(rep_time):
                print(f'{par_name}_{par}_[{k + 1}/{rep_time}]___________')
                # R_RAND
                r_rand = RepRand(ltsp_path, par, max_stp_iter)
                res_dir['path_len'][k, i, 0] = r_rand.path_len
                res_dir['H'][k, i, 0] = r_rand.path_len * eta_par
                res_dir['U'][k, i, 0] = alpha_g_dict.get(par) - r_rand.path_len * eta_par
                # R_LOC
                r_loc = RepLoc(ltsp_path, par, mut_rate)
                res_dir['path_len'][k, i, 1] = r_loc.path_len
                res_dir['H'][k, i, 1] = r_loc.path_len * eta_par
                res_dir['U'][k, i, 1] = alpha_g_dict.get(par) - r_loc.path_len * eta_par
                # R_FLOW
                r_flow = RepFlow(ltsp_path, par, max_stp_iter, mut_rate)
                res_dir['path_len'][k, i, 2] = r_flow.path_len
                res_dir['H'][k, i, 2] = r_flow.path_len * eta_par
                res_dir['U'][k, i, 2] = alpha_g_dict.get(par) - r_flow.path_len * eta_par
                # R_GENE
                r_gene = RepGene(ltsp_path, par, pop, sur, mut, stp_iters, iters)
                res_dir['path_len'][k, i, 3] = r_gene.path_len
                res_dir['H'][k, i, 3] = r_gene.path_len * eta_par
                res_dir['U'][k, i, 3] = alpha_g_dict.get(par) - r_gene.path_len * eta_par
            # save results in .xls
            file_name = f"{par_name}_{par}_for_plt.xls"
            workbook = xlwt.Workbook()
            for res_name in res_lst:
                sheet_name = res_name
                sheet = workbook.add_sheet(sheet_name)
                res_arr = res_dir.get(res_name)
                # Write header
                sheet.write(0, 0, "rep_time")
                for k in range(rep_time):
                    sheet.write(k + 1, 0, k + 1)
                for j, app_name in enumerate(app_lst):
                    sheet.write(0, j + 1, app_name)
                # Write data
                for k in range(rep_time):
                    for j in range(len(app_lst)):
                        sheet.write(k + 1, j + 1, float(res_arr[k][i][j]))
            workbook.save(os.path.join(result_dir, file_name))

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
        file_name = f"n{task_num}_{par_name}_for_plt.xls"
        workbook = xlwt.Workbook()
        for res_name in res_lst:
            sheet_name = res_name
            sheet = workbook.add_sheet(sheet_name)
            res_arr = res_dir.get(res_name)
            sheet.write(0, 0, par_name)
            for i, par in enumerate(par_arr):
                sheet.write(i + 1, 0, float(par))
            for j, app_name in enumerate(app_lst):
                sheet.write(0, j + 1, app_name)
            for i in range(res_arr.shape[0]):
                for j in range(res_arr.shape[1]):
                    sheet.write(i + 1, j + 1, float(res_arr[i][j]))
        sheet_name = "alpha_plus"
        sheet = workbook.add_sheet(sheet_name)
        sheet.write(0, 0, "alpha_plus")
        sheet.write(1, 0, float(alpha_plus))
        workbook.save(os.path.join(result_dir, file_name))

    print(f"{time.strftime('%m_%d, %H:%M')}_{par_name} is DONE! :)")
