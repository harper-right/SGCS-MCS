from libs.util import get_worker_response
from libs.util import get_g
from libs.game_info import GameInfo
from libs.game_pdf_eq import GamePDF
from uav_apps.r_gene import RepGene
from compared_schemes.itd import ITD
from compared_schemes.vir_ea import VirEa
from compared_schemes.uitde import UITDE
import numpy as np
import os
import time
import xlwt
import yaml


# cmp_apps is a comparison with strategies in other references; m means the horizontal axis is worker_num
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
    par_name = 'cmp_M'  # Compared + Worker number
    par_lst = [200, 400, 600, 800, 1000, 1200, 1400]  # worker number
    data_sets_lst = ["t_drive", "roma", "mobility"]
    # index_j
    app_lst = ['SGCS', 'ITD', "VIR_EA", "UITDE"]
    big_theta_proportion = 0.30  # for ITD
    phi = 0.10  # for VIR_EA
    max_path = 4E4  # for UITDE

    # index_k
    rep_time = 10

    # To get U
    # index_set, each set has a separate folder to save the results
    for set_idx, set_name in enumerate(data_sets_lst):
        # station distribution dataset
        data_base = "./source_data/" + f"{set_name}/"
        # save result path
        result_dir = f"./results/{time.strftime('%m_%d_%H_%M')}_{par_name}_{set_name}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # res dic
        res_lst = ['alpha', 'path_len', 'H', 'G', 'U']
        res_dir = dict()
        for res_name in res_lst:
            res_dir[res_name] = np.zeros((rep_time, len(par_lst), len(app_lst)))

        for i, par in enumerate(par_lst):  # change worker number
            g_info.m = par
            # experiments
            for k in range(rep_time):
                print(f'{set_name}_{par_name}_{par}_[{k + 1}/{rep_time}]___________')
                # To get alpha-G(game profit), using monte-carlo
                game_rep_time = 100
                alpha_arr = np.linspace(0.0, 1.0, num=101)  # alpha_arr
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
                    for alpha_idx, alpha in enumerate(alpha_arr):
                        x_arr = get_worker_response(g_info, p_arr, alpha)
                        g_arr[rep, alpha_idx], _, _ = get_g(g_info, p_arr, alpha, x_arr)
                avg_g_arr = np.average(g_arr, axis=0)
                idx = np.argmax(avg_g_arr)
                best_alpha = alpha_arr[idx]
                # Creating the dictionary
                alpha_g_dict = {round(alpha, 2): g_value for alpha, g_value in zip(alpha_arr, avg_g_arr)}
                # To get alpha of each apps
                task_num = 60
                file_name = f"n{task_num}.vrp"
                ltsp_path = os.path.join(data_base, file_name)
                # SGCS
                alpha = round(best_alpha, 2)
                sgcs_app = RepGene(ltsp_path, alpha, pop, sur, mut, stp_iters, iters)
                res_dir['alpha'][k, i, 0] = alpha
                res_dir['path_len'][k, i, 0] = sgcs_app.path_len
                res_dir['H'][k, i, 0] = sgcs_app.path_len * eta_par
                res_dir["G"][k, i, 0] = alpha_g_dict.get(alpha)
                res_dir["U"][k, i, 0] = alpha_g_dict.get(alpha) - sgcs_app.path_len * eta_par
                # ITD
                big_theta = int(big_theta_proportion * task_num)
                itd_app = ITD(ltsp_path, big_theta, pop, sur, mut, stp_iters, iters)
                alpha = round(itd_app.actual_verification_rate, 2)
                res_dir['alpha'][k, i, 1] = alpha
                res_dir['path_len'][k, i, 1] = itd_app.path_len
                res_dir['H'][k, i, 1] = itd_app.path_len * eta_par
                res_dir["G"][k, i, 1] = alpha_g_dict.get(alpha)
                res_dir["U"][k, i, 1] = alpha_g_dict.get(alpha) - itd_app.path_len * eta_par
                # VIR_EA
                vir_ea = VirEa(ltsp_path, alpha, stp_iters, phi)
                alpha = round(vir_ea.actual_verification_rate, 2)
                res_dir['alpha'][k, i, 2] = alpha
                res_dir['path_len'][k, i, 2] = vir_ea.path_len
                res_dir['H'][k, i, 2] = vir_ea.path_len * eta_par
                res_dir["G"][k, i, 2] = alpha_g_dict.get(alpha)
                res_dir["U"][k, i, 2] = alpha_g_dict.get(alpha) - vir_ea.path_len * eta_par
                # UITDE
                uitde_app = UITDE(ltsp_path, alpha, stp_iters, max_path)
                alpha = round(uitde_app.actual_verification_rate, 2)
                res_dir['alpha'][k, i, 3] = alpha
                res_dir['path_len'][k, i, 3] = uitde_app.path_len
                res_dir['H'][k, i, 3] = uitde_app.path_len * eta_par
                res_dir['G'][k, i, 3] = alpha_g_dict.get(alpha)
                res_dir['U'][k, i, 3] = alpha_g_dict.get(alpha) - uitde_app.path_len * eta_par
            # save results in .xls
            file_name = f"{set_name}_{par_name}_{par}_for_plt.xls"
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
        file_name = f"{set_name}_{par_name}_for_plt.xls"
        workbook = xlwt.Workbook()
        for res_name in res_lst:
            sheet_name = res_name
            sheet = workbook.add_sheet(sheet_name)
            res_arr = res_dir.get(res_name)
            sheet.write(0, 0, par_name)
            for i, par in enumerate(par_lst):
                sheet.write(i + 1, 0, float(par))
            for j, app_name in enumerate(app_lst):
                sheet.write(0, j + 1, app_name)
            for i in range(res_arr.shape[0]):
                for j in range(res_arr.shape[1]):
                    sheet.write(i + 1, j + 1, float(res_arr[i][j]))
        workbook.save(os.path.join(result_dir, file_name))

    print(f"{time.strftime('%m_%d, %H:%M')}_{par_name} is DONE! :)")
