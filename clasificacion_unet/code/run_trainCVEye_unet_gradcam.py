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

    script = "trainCVEye_unet_gradcam.py"
    dir = 'SEP'
    cfg_file = "conf.config_70iter"
    seed = 0
    dseed = 0

    # cfg_file = ["conf.config_70iter_ELE_4", "conf.config_70iter_CORNEA-DENS_0", "conf.config_70iter_PAC_0"]

    patch_type = 'map'
    patch_num = [2]
    loss = ['MSE']

    # for i in patch_num:
    #     for l in loss:
    #         weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_final"
    #         print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #         parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #         run_script(script, parameters)
    
    # for i in patch_num:
    #     for l in loss:
    #         weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask)_final"
    #         print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #         parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #         run_script(script, parameters)

    # patch_type = 'map'
    # patch_num = [2]
    # loss = ['MSE']
    # dilation = [20]
    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)
    
    # patch_type = 'map'
    # patch_num = [2]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)

    # patch_type = 'both'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]
    # alpha_weighted = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for a in alpha_weighted:
    #                 weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "newWeightedLoss_alpha" + str(a) + "(mask_dilation" + str(d) + ")_final"
    #                 print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #                 parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #                 run_script(script, parameters)

    patch_type = 'map'
    patch_num = [2]
    loss = ['MSE']
    alpha_weighted = [3]
    error_type = ['ele_pac']

    for i in patch_num:
        for l in loss:
            for b in alpha_weighted:
                for e in error_type:
                    if e == '':
                        weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_beta" + str(b) + "_final"
                    else:
                        weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_beta" + str(b) + "(" + e + ")_final"
                    print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
                    parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
                    run_script(script, parameters)
    
    # patch_type = 'map'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]
    # alpha_weighted = [4, 5, 6]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
                
    #             for a in alpha_weighted:
    #                 weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss_alpha" + str(a) + "(mask_dilation" + str(d) + ")_final"
    #                 print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #                 parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #                 run_script(script, parameters)

    # patch_type = 'both'
    # patch_num = [1]
    # loss = ['SSIM', 'MAE', 'NCC', 'LocalNCC']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)
    
    # patch_type = 'ring'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)

    # patch_type = 'arc'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)

    # Volume

    # patch_type = 'both'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_volume_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)
    
    # patch_type = 'ring'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_volume_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)

    # patch_type = 'arc'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_volume_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)
    
    # Weighted Loss

    # Map
    # patch_type = 'map'
    # patch_num = [1, 2]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)
    
    # patch_type = 'both'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)
    
    # patch_type = 'ring'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)

    # patch_type = 'arc'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)

    # # Volume

    # patch_type = 'both'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_volume_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)
    
    # patch_type = 'ring'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_volume_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)

    # patch_type = 'arc'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_volume_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)


    # patch_type = 'map'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)
    
    # patch_type = 'map'
    # patch_num = [2]
    # loss = ['MSE']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)

    # patch_type = 'map'
    # patch_num = [1]
    # loss = ['MAE', 'NCC', 'LocalNCC', 'SSIM']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir + '     weights ssl -------> ' + weights_dir)
    #             parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #             run_script(script, parameters)

    # patch_type = 'map'
    # patch_num = [1]
    # loss = ['MSESSIM']
    # dilation = [3]
    # alpha = [0.1, 1, 0]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for a in alpha:
    #                 weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_alpha" + str(a) + "(mask_dilation" + str(d) + ")_final"
    #                 print(script + ' data -------> ' + dir+ '     weights ssl -------> ' + weights_dir)
    #                 parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed) ]
    #                 run_script(script, parameters)

