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

    script = "trainCVEye_unet70iter.py"
    dir = 'SEP'
    seed = 0
    dseed = 0

    # cfg_file = ["conf.config_70iter_ELE_4", "conf.config_70iter_CORNEA-DENS_0", "conf.config_70iter_PAC_0"]


    # Map
    patch_type = 'map'
    patch_num = [1, 2]
    loss = ['MSE']

    for i in patch_num:
        for l in loss:
            weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_tfm"
            print('weights ssl -------> ' + weights_dir)
            parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed)]
            run_script(script, parameters)

    # Both
    patch_type = 'both'
    patch_num = [1]
    loss = ['MSE']
    volume = [False, True]

    for i in patch_num:
        for l in loss:
            for v in volume:
                if v:
                    weights_dir = "unet_" + patch_type + "_volume_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_tfm"
                else:
                    weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_tfm"
                print('weights ssl -------> ' + weights_dir)
                parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed)]
                run_script(script, parameters)
    
    # Ring
    patch_type = 'ring'
    patch_num = [1]
    loss = ['MSE']
    volume = [False, True]

    for i in patch_num:
        for l in loss:
            for v in volume:
                if v:
                    weights_dir = "unet_" + patch_type + "_volume_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_tfm"
                else:
                    weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_tfm"
                print('weights ssl -------> ' + weights_dir)
                parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed)]
                run_script(script, parameters)
    
    # Arc
    patch_type = 'arc'
    patch_num = [1]
    loss = ['MSE']
    volume = [False, True]

    for i in patch_num:
        for l in loss:
            for v in volume:
                if v:
                    weights_dir = "unet_" + patch_type + "_volume_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_tfm"
                else:
                    weights_dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_tfm"
                print('weights ssl -------> ' + weights_dir)
                parameters = ["dir", dir, "dir_weights_file", weights_dir, "weights_seed", str(seed), "weights_dseed", str(dseed)]
                run_script(script, parameters)