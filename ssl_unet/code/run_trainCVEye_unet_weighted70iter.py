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

    script = "trainCVEye_unet_weighted70iter.py"
    seed = 0
    dseed = 0

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