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

    script = "aggregation.py"

    # patch_type = 'map'
    # patch_num = [1, 2]
    # loss = ['MSE']
    # dseed = 0
    # seed = 0
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir)
    #             parameters = ["dir", dir]
    #             run_script(script, parameters)
    
    patch_type = 'both'
    patch_num = [1]
    loss = ['MSE']
    dseed = 0
    seed = 0
    dilation = [3]
    alpha_weight = [0.5, 1, 3, 5, 7, 9]

    for i in patch_num:
        for l in loss:
            for d in dilation:
                for a in alpha_weight:
                    dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss_alpha" + str(a) + "(mask_dilation" + str(d) + ")_final"
                    print(script + ' data -------> ' + dir)
                    parameters = ["dir", dir]
                    run_script(script, parameters)
    
    # patch_type = 'map'
    # patch_num = [1]
    # loss = ['MSE']
    # dseed = 0
    # seed = 0
    # dilation = [3]
    # alpha_weight = [0.5, 1, 3, 5, 7, 9]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for a in alpha_weight:
    #                 dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss_alpha" + str(a) + "(mask_dilation" + str(d) + ")_final"
    #                 print(script + ' data -------> ' + dir)
    #                 parameters = ["dir", dir]
    #                 run_script(script, parameters)
    
    patch_type = 'both'
    patch_num = [1]
    loss = ['MSE']
    dseed = 0
    seed = 0
    dilation = [3]
    alpha_weight = [0.5, 1, 2, 3, 5, 10]

    for i in patch_num:
        for l in loss:
            for d in dilation:
                for a in alpha_weight:
                    dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss_alpha" + str(a) + "(mask_dilation" + str(d) + ")_final"
                    print(script + ' data -------> ' + dir)
                    parameters = ["dir", dir]
                    run_script(script, parameters)
    
    # patch_type = 'ring'
    # patch_num = [1]
    # loss = ['MSE']
    # dseed = 0
    # seed = 0
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             dir = "unet_" + patch_type + "_volume_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir)
    #             parameters = ["dir", dir]
    #             run_script(script, parameters)

    # patch_type = 'arc'
    # patch_num = [1]
    # loss = ['MSE']
    # dseed = 0
    # seed = 0
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             dir = "unet_" + patch_type + "_volume_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' data -------> ' + dir)
    #             parameters = ["dir", dir]
    #             run_script(script, parameters)

    # loss = ['MSESSIM']
    # alpha = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for a in alpha:
    #                 dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_alpha" + str(a) + "(mask_dilation" + str(d) + ")_prueba"
    #                 print(script + ' data -------> ' + dir)
    #                 parameters = ["dir", dir]
    #                 run_script(script, parameters)
    
    # loss = ['SSIM']
    # erosion = [1, 3, 5, 7, 9]

    # for i in patch_num:
    #     for l in loss:
    #         for e in erosion:
    #             dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_erosion" + str(e) + ")_prueba"
    #             print(script + ' data -------> ' + dir)
    #             parameters = ["dir", dir]
    #             run_script(script, parameters)

    # seed = [0]
    # dseed = [0]
    # th = [0, 0.005, 0.01, 0.015, 0.02]
    # for i in seed:
    #     for d in dseed:
    #         for x in th:
    #                 dir = "no_deactivate" + "_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_binaryWeightedLoss(full DB)"
    #                 print(script + ' data -------> ' + dir)
    #                 parameters = ["dir_results", dir]
    #                 run_script(script, parameters)   

    
    
    