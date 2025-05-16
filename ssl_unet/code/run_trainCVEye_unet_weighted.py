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

    script = "trainCVEye_unet_weighted.py"
    seed = 0
    dseed = 0

    # patch_type = 'both'
    # patch_num = [1]
    # loss = ['MSE']
    # volume = ['False']
    # dilation = [3]
    # alpha_weighted = [3]
    # error_type = ['ele_pac']
    
    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for v in volume:
    #                 for a in alpha_weighted:
    #                     for e in error_type:
    #                         if v == 'False':
    #                             dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss(" + e + ")_alpha" + str(a) + "(mask_dilation" + str(d) + ")_final2"
    #                         else:
    #                             dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss(" + e + ")_alpha" + str(a) +"(mask_dilation" + str(d) + ")_final"
    #                         print(script + ' dir-------> ' + dir)
    #                         parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d), "volume", str(v), "alpha_weighted", str(a), 'error_type', str(e)] 
    #                         run_script(script, parameters)

    patch_type = 'map'
    patch_num = [2]
    loss = ['MSE']
    alpha_weighted = [1.5, 2, 2.5, 3.5, 4, 4.5]
    error_type = ['ele_pac']
    
    for i in patch_num:
        for l in loss:
            for b in alpha_weighted:
                for e in error_type:
                    if e == '':
                        dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_beta" + str(b) + "_final"
                    else:
                        dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_beta" + str(b) + "(" + e + ")_final"
                    print(script + ' dir-------> ' + dir)
                    parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "alpha_weighted", str(b), 'error_type', str(e)] 
                    run_script(script, parameters)

    # Weighted Losses
    patch_type = 'map'
    patch_num = [2]
    loss = ['MSESSIM']
    alpha_loss = [0.7]
    alpha_weighted = [1.5, 2, 2.5, 3.5, 4, 4.5]
    error_type = ['ele_pac']
    
    for i in patch_num:
        for l in loss:
            for a in alpha_loss:
                for b in alpha_weighted:
                    for e in error_type:
                        if e == '':
                            dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_alpha" + str(a) + "_beta" + str(b) + "_final"
                        else:
                            dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_alpha" + str(a) + "_beta" + str(b) + "(" + e + ")_final"
                        print(script + ' dir-------> ' + dir)
                        parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "alpha_loss", str(a), "alpha_weighted", str(b), 'error_type', str(e)] 
                        run_script(script, parameters)


    # patch_type = 'both'
    # patch_num = [1]
    # loss = ['MSE']
    # volume = ['False']
    # dilation = [1, 5, 7, 9]
    
    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for v in volume:
    #                 if v == 'False':
    #                     dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #                 else:
    #                     dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #                 print(script + ' dir-------> ' + dir)
    #                 parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d), "volume", str(v)] 
    #                 run_script(script, parameters)
    
    # erosion = [1, 5, 7, 9]

    # for i in patch_num:
    #     for l in loss:
    #         for e in erosion:
    #             for v in volume:
    #                 if v == 'False':
    #                     dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_erosion" + str(d) + ")_final"
    #                 else:
    #                     dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_erosion" + str(d) + ")_final"
    #                 print(script + ' dir-------> ' + dir)
    #                 parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "erosion", str(e), "volume", str(v)] 
    #                 run_script(script, parameters)
    
    # patch_type = 'map'
    # patch_num = [1]
    # loss = ['MSE']
    # volume = ['False']
    # dilation = [1, 5, 7, 9]
    
    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #             print(script + ' dir-------> ' + dir)
    #             parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d), "volume", str(v)] 
    #             run_script(script, parameters)
    
    # erosion = [1, 5, 7, 9]

    # for i in patch_num:
    #     for l in loss:
    #         for e in erosion:
    #             for v in volume:
    #                 dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_erosion" + str(d) + ")_final"
    #                 print(script + ' dir-------> ' + dir)
    #                 parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "erosion", str(e), "volume", str(v)] 
    #                 run_script(script, parameters)
    
    # patch_type = 'ring'
    # patch_num = [1]
    # loss = ['MSE']
    # volume = ['False']
    # dilation = [1, 5, 7, 9]
    
    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for v in volume:
    #                 if v == 'False':
    #                     dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #                 else:
    #                     dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #                 print(script + ' dir-------> ' + dir)
    #                 parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d), "volume", str(v)] 
    #                 run_script(script, parameters)
    
    # erosion = [1, 5, 7, 9]

    # for i in patch_num:
    #     for l in loss:
    #         for e in erosion:
    #             for v in volume:
    #                 if v == 'False':
    #                     dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_erosion" + str(d) + ")_final"
    #                 else:
    #                     dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_erosion" + str(d) + ")_final"
    #                 print(script + ' dir-------> ' + dir)
    #                 parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "erosion", str(e), "volume", str(v)] 
    #                 run_script(script, parameters)
    
    # patch_type = 'arc'
    # patch_num = [1]
    # loss = ['MSE']
    # volume = ['False']
    # dilation = [1, 5, 7, 9]
    
    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for v in volume:
    #                 if v == 'False':
    #                     dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #                 else:
    #                     dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #                 print(script + ' dir-------> ' + dir)
    #                 parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d), "volume", str(v)] 
    #                 run_script(script, parameters)
    
    # erosion = [1, 5, 7, 9]

    # for i in patch_num:
    #     for l in loss:
    #         for e in erosion:
    #             for v in volume:
    #                 if v == 'False':
    #                     dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_erosion" + str(d) + ")_final"
    #                 else:
    #                     dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_erosion" + str(d) + ")_final"
    #                 print(script + ' dir-------> ' + dir)
    #                 parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "erosion", str(e), "volume", str(v)] 
    #                 run_script(script, parameters)
    
    
    # patch_type = 'map'
    # patch_num = [1]
    # loss = ['MSE']
    # volume = ['False']
    # dilation = [3]
    # alpha_weighted = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7, 9]
    
    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for v in volume:
    #                 for a in alpha_weighted:
    #                     if v == 'False':
    #                         dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "newWeightedLoss_alpha" + str(a) + "(mask_dilation" + str(d) + ")_final"
    #                     else:
    #                         dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "newWeightedLoss_alpha" + str(a) +"(mask_dilation" + str(d) + ")_final"
    #                     print(script + ' dir-------> ' + dir)
    #                     parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d), "volume", str(v), "alpha_weighted", str(a)] 
    #                     run_script(script, parameters)
    

    # patch_type = 'ring'
    # patch_num = [1]
    # loss = ['MSE']
    # volume = ['False']
    # dilation = [3]
    # alpha_weighted = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7, 9]
    
    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for v in volume:
    #                 for a in alpha_weighted:
    #                     if v == 'False':
    #                         dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "newWeightedLoss_alpha" + str(a) + "(mask_dilation" + str(d) + ")_final"
    #                     else:
    #                         dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "newWeightedLoss_alpha" + str(a) +"(mask_dilation" + str(d) + ")_final"
    #                     print(script + ' dir-------> ' + dir)
    #                     parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d), "volume", str(v), "alpha_weighted", str(a)] 
    #                     run_script(script, parameters)
    

    # patch_type = 'both'
    # patch_num = [1]
    # loss = ['MSE']
    # volume = ['False']
    # dilation = [3]
    # alpha_weighted = [0, 4, 6]
    
    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for v in volume:
    #                 for a in alpha_weighted:
    #                     if v == 'False':
    #                         dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss_alpha" + str(a) + "(mask_dilation" + str(d) + ")_final"
    #                     else:
    #                         dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss_alpha" + str(a) +"(mask_dilation" + str(d) + ")_final"
    #                     print(script + ' dir-------> ' + dir)
    #                     parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d), "volume", str(v), "alpha_weighted", str(a)] 
    #                     run_script(script, parameters)
    
    # patch_type = 'map'
    # patch_num = [1]
    # loss = ['MSE']
    # volume = ['False']
    # dilation = [3]
    # alpha_weighted = [0, 4, 5, 6]
    
    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for v in volume:
    #                 for a in alpha_weighted:
    #                     if v == 'False':
    #                         dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss_alpha" + str(a) + "(mask_dilation" + str(d) + ")_final"
    #                     else:
    #                         dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss_alpha" + str(a) +"(mask_dilation" + str(d) + ")_final"
    #                     print(script + ' dir-------> ' + dir)
    #                     parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d), "volume", str(v), "alpha_weighted", str(a)] 
    #                     run_script(script, parameters)
    
    # # Ring

    # patch_type = 'ring'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]
    # volume = ['False', 'True']

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for v in volume:
    #                 if v == 'False':
    #                     dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss(mask_dilation" + str(d) + ")_final"
    #                 else:
    #                     dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #                 print(script + ' dir-------> ' + dir)
    #                 parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d), "volume", str(v)]
    #                 run_script(script, parameters)

    # # Arc
    # patch_type = 'arc'
    # patch_num = [1]
    # loss = ['MSE']
    # dilation = [3]
    # volume = ['False', 'True']

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for v in volume:
    #                 if v == 'False':
    #                     dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "WeightedLoss(mask_dilation" + str(d) + ")_final"
    #                 else:
    #                     dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_final"
    #                 print(script + ' dir-------> ' + dir)
    #                 parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d),  "volume", str(v)]
    #                 run_script(script, parameters)


    # patch_type = 'map'
    # patch_num = [1]
    # loss = ['MSE', 'MAE', 'NCC', 'LocalNCC', 'SSIM']
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_prueba"
    #             print(script + ' dir-------> ' + dir)
    #             parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d)]
    #             run_script(script, parameters)

    # patch_type = 'map'
    # patch_num = [1]
    # loss = ['MSESSIM']  
    # alpha = [1, 0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             for a in alpha:
    #                 dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_alpha" + str(a) + "(mask_dilation" + str(d) + ")_prueba"
    #                 print(script + ' dir-------> ' + dir)
    #                 parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d), "alpha_loss", str(a)]
    #                 run_script(script, parameters)

    # Both 

    patch_type = 'map'
    patch_num = [1]
    loss = ['MSE', 'MAE', 'NCC', 'LocalNCC']
    dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_prueba"
    #             print(script + ' dir-------> ' + dir)
    #             parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d)]
    #             run_script(script, parameters)

    patch_type = 'map'
    patch_num = [2]
    loss = ['MSE', 'SSIM']
    dilation = [3]

    # for i in patch_num:
    #     for l in loss:
    #         for d in dilation:
    #             dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss(mask_dilation" + str(d) + ")_prueba"
    #             print(script + ' dir-------> ' + dir)
    #             parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "dilation", str(d)]
    #             run_script(script, parameters)

    # patch_type = 'map'
                    
    # # Rings
    # patch_size = [5, 10, 20]
    # patch_volume = ['False', 'True']
    # patch_type = 'ring'
    # patch_num = [1, 3, 5]
    # patch_w_start = [40, 60, 80]
    # loss = ['MSE']
    
    # for i in patch_size:
    #     for j in patch_volume:
    #         for k in patch_num:
    #             for x in patch_w_start:
    #                 for l in loss:
    #                     if j == 'False':
    #                         dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_size" + str(i) + "_num" + str(k) + "_limit" + str(x) + "_" + l + "Loss"
    #                     else:
    #                         dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_size" + str(i) + "_num" + str(k) + "_limit" + str(x) + "_" + l + "Loss"
                        
    #                     print(script + ' dir-------> ' + dir)
    #                     parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "patch_size", str(i), "patch_volume", str(j), "patch_type", str(patch_type), "patch_num", str(k), "patch_w_start", str(x), "loss", l]
    #                     run_script(script, parameters)
        
    # # # Arcs
    # patch_size = [10, 20, 30]
    # patch_volume = ['False', 'True']
    # patch_type = 'arc'
    # patch_num = [1, 3, 5]
    # patch_h_start = [352]
    # loss = ['MSE']

    # for i in patch_size:
    #     for j in patch_volume:
    #         for k in patch_num:
    #             for x in patch_h_start:
    #                 for l in loss:
    #                     if j == 'False':
    #                         dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_size" + str(i) + "_num" + str(k) + "_limit" + str(x) + "_" + l + "Loss"
    #                     else:
    #                         dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_size" + str(i) + "_num" + str(k) + "_limit" + str(x) + "_" + l + "Loss"
                        
    #                     print(script + ' dir-------> ' + dir)
    #                     parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "patch_size", str(i), "patch_volume", str(j), "patch_type", str(patch_type), "patch_num", str(k), "patch_h_start", str(x), "loss", l]
    #                     run_script(script, parameters)

    # # # Both

    # patch_size = [30, 40, 50, 60]
    # patch_volume = ['False', 'True']
    # patch_type = 'both'
    # patch_num = [1, 3, 5]
    # patch_h_start = [352]
    # patch_w_start = [60, 80]
    # loss = ['MSE']

    # for i in patch_size:
    #     for j in patch_volume:
    #         for k in patch_num:
    #             for x in patch_w_start:
    #                 for y in patch_h_start:
    #                     for l in loss:
    #                         if j == 'False':
    #                             dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_size" + str(i) + "_num" + str(k) + "_limitw" + str(x) + "_h" + str(y) + "_" + l + "Loss"
    #                         else:
    #                             dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_size" + str(i) + "_num" + str(k) + "_limitw" + str(x) + "_h" + str(y) + "_" + l + "Loss"
                        
    #                         print(script + ' dir-------> ' + dir)
    #                         parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "patch_size", str(i), "patch_volume", str(j), "patch_type", str(patch_type), "patch_num", str(k), "patch_h_start", str(y), "patch_w_start", str(x), "loss", l]
    #                         run_script(script, parameters)



