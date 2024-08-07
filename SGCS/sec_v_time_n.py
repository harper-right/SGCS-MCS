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


# Comparison of runtime, with respect to the number of tasks
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
    par_name = 'time_n'  # running_time + Task number
    # par_lst = [30, 40, 50, 60, 70, 80, 90]  # task number
    par_lst = [40]  # FIXME only for test
    # data_sets_lst = ["t_drive", "roma", "mobility"]
    data_sets_lst = ["mobility"]
    # index_j
    app_lst = ['SGCS', 'ITD', "VIR_EA", "UITDE"]
    big_theta_proportion = 0.30  # for ITD
    phi = 0.10  # for VIR_EA
    max_path = 4E4  # for UITDE

    # index_k
    rep_time = 3

    # To get U
    # `index_set`, save results for each set in a separate folder
    for set_idx, set_name in enumerate(data_sets_lst):
        # station distribution dataset
        data_base = "./source_data/" + f"{set_name}/"
        # save result path
        result_dir = f"./results/{time.strftime('%m_%d_%H_%M')}_times_{set_name}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # To get alpha-G(game profit), using monte-carlo
        game_rep_time = 100
        alpha_arr = np.linspace(0.0, 1.0, num=101)  # alpha_arr
        g_arr = np.zeros((game_rep_time, alpha_arr.shape[0]))
        for rep in range(game_rep_time):
            g_info.omg, g_info.c = g_info.refresh_workers()
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
        best_alpha = alpha_arr[idx]
        # Creating the dictionary
        alpha_g_dict = {round(alpha, 2): g_value for alpha, g_value in zip(alpha_arr, avg_g_arr)}

        # res dic
        times_res = np.zeros((len(par_lst), len(app_lst)))  # In seconds
        # experiments
        for i, par in enumerate(par_lst):
            file_name = "n" + str(par) + ".vrp"
            ltsp_path = os.path.join(data_base, file_name)
            # SGCS
            alpha = round(best_alpha, 2)
            start_time = time.time()
            for k in range(rep_time):
                sgcs_app = RepGene(ltsp_path, alpha, pop, sur, mut, stp_iters, iters)
            end_time = time.time()
            print(f"Task number {par}, SGCS is Done")
            times_res[i, 0] = end_time - start_time
            # ITD
            big_theta = big_theta_proportion * par
            start_time = time.time()
            for k in range(rep_time):
                itd_app = ITD(ltsp_path, big_theta, pop, sur, mut, stp_iters, iters)
            end_time = time.time()
            print(f"Task number {par}, ITD is Done")
            times_res[i, 1] = end_time - start_time
            # VIR_EA
            start_time = time.time()
            for k in range(rep_time):
                vir_ea = VirEa(ltsp_path, alpha, stp_iters, phi)
            end_time = time.time()
            print(f"Task number {par}, VIR_EA is Done")
            times_res[i, 2] = end_time - start_time
            # UITDE
            start_time = time.time()
            for k in range(rep_time):
                uitde_app = UITDE(ltsp_path, alpha, stp_iters, max_path)
            end_time = time.time()
            print(f"Task number {par}, UITDE is Done")
            times_res[i, 3] = end_time - start_time
        # get average results
        avg_times_res = times_res / rep_time
        # save res in .npy
        print(f"SAVE res_times ...")
        np.save(os.path.join(result_dir, f'run_times.npy'), avg_times_res)
        # save average results in .xls
        file_name = f"{set_name}_{par_name}_for_plt.xls"
        workbook = xlwt.Workbook()
        sheet_name = "time_in_s"
        sheet = workbook.add_sheet(sheet_name)
        sheet.write(0, 0, par_name)
        for i, par in enumerate(par_lst):
            sheet.write(i + 1, 0, float(par))
        for j, app_name in enumerate(app_lst):
            sheet.write(0, j + 1, app_name)
        for i in range(avg_times_res.shape[0]):
            for j in range(avg_times_res.shape[1]):
                sheet.write(i + 1, j + 1, float(avg_times_res[i][j]))
        workbook.save(os.path.join(result_dir, file_name))

    print(f"{time.strftime('%m_%d, %H:%M')}_{par_name} is DONE! :)")
