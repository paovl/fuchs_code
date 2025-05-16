"""
@author: pvltarife
"""
import os
import subprocess
import random

def run_script(script, parameters):
    try:
        process = subprocess.run(['python', script] + parameters, check=True)
        print(f"-----------------RUNNING SCRIPT----------------------")
        # Check if the process ended correctly
        if process.returncode == 0:
            print(f"The running of {script} has ended successfully.")
        else:
            print(f"The running of {script} has ended because an error ocurred.")
    except subprocess.CalledProcessError as e:
        print("Error at running {script}: {e}")

if '___main___': 

    script = "new_trainCVEye_unet.py"
    dir = 'SEP'
    cfg_file = "conf.config_70iter"
    db = "global"
    # data_dir = 'ransac_TH_1.5_r_45'
    data_dir = 'original'

    # Experiments unet

    seed = 0
    dseed = 0

    # # Rings
    # patch_size = [5, 10, 20]
    # patch_volume = ['False', 'True']
    # patch_type = 'ring'
    # patch_num = [1, 3, 5]
    # patch_w_start = [40, 60, 80]
    # loss = ['NCC', 'SSIM']

    # for i in patch_size:
    #     for j in patch_volume:
    #         for k in patch_num:
    #             for x in patch_w_start:
    #                 if j == 'False':
    #                     weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_size" + str(i) + "_num" + str(k) + "_limit" + str(x)
    #                 else:
    #                     weights_dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_size" + str(i) + "_num" + str(k) + "_limit" + str(x)
    #             print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)
    
    # # Arcs

    # patch_size = [10, 20, 30]
    # patch_volume = ['False', 'True']
    # patch_type = 'arc'
    # patch_num = [1, 3, 5]
    # patch_h_start = [352]

    # for i in patch_size:
    #     for j in patch_volume:
    #         for k in patch_num:
    #             for x in patch_h_start:
    #                 if j == 'False':
    #                     weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_size" + str(i) + "_num" + str(k) + "_limit" + str(x)
    #                 else:
    #                     weights_dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_size" + str(i) + "_num" + str(k) + "_limit" + str(x)
    #             print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)

    # # Both

    # patch_size = [30, 40, 50, 60]
    # patch_volume = ['False', 'True']
    # patch_type = 'both'
    # patch_num = [1, 3, 5]
    # patch_h_start = [352]
    # patch_w_start = [60, 80]

    # for i in patch_size:
    #     for j in patch_volume:
    #         for k in patch_num:
    #             for x in patch_w_start:
    #                 for y in patch_h_start:

    #                     if j == 'False':
    #                         weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_size" + str(i) + "_num" + str(k) + "_limitw" + str(x) + "_h" + str(y)
    #                     else:
    #                         weights_dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_size" + str(i) + "_num" + str(k) + "_limitw" + str(x) + "_h" + str(y)
                    
    #                     print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #                     parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #                     run_script(script, parameters)


    # Map
    # patch_type = 'map'
    # patch_num = [1]
    # loss = ['MSE']

    # for i in patch_num:
    #     for l in loss:
    #         weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(original_norm_r2_fixed_2maps)"
    #         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #         parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #         run_script(script, parameters)
    
    # dilation = [1, 3, 5, 10, 15]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(original_norm_r2_mask_dilation" + str(d) + "_fixed)"
    #             print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)

    # patch_type = 'map'
    # patch_num = [1, 2]
    # loss = ['MSE']

    # for i in patch_num:
    #     for l in loss:
    #         weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss"
    #         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #         parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #         run_script(script, parameters)

    # patch_type = 'map'
    # patch_num = [1]
    # loss = ['LocalNCC']

    # for i in patch_num:
    #     for l in loss:
    #         weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(original_norm_r2_mask_fixed)"
    #         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #         parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #         run_script(script, parameters)

    # for i in patch_num:
    #     for l in loss:
    #         weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(more_closer_norm)"
    #         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #         parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #         run_script(script, parameters)
    # patch_type = 'map'
    # patch_num = [1]
    # loss = ['MSE']

    # for i in patch_num:
    #     for l in loss:
    #         weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(abs)"
    #         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #         parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #         run_script(script, parameters)

    # Map
    patch_type = 'map'
    patch_num = [1]
    loss = ['MSE']
    dilation = [3]
    prob = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for i in patch_num:
        for l in loss:
            for d in dilation:
                for p in prob:
                    weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")"
                    print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
                    parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed), "prob_aug", str(p)]
                    run_script(script, parameters)

    # patch_type = 'map'
    # patch_num = [1]
    # loss = ['MSESSIM']
    # dilation = [3]
    # alpha = [0.7, 0.8, 0.9]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for a in alpha:
    #                 weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_alpha" + str(a) + "(mask_dilation" + str(d) + ")"
    #                 print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #                 parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #                 run_script(script, parameters)