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
    seed = 0
    dseed = 0

    patch_type = 'map'
    patch_num = [1, 2]
    loss = ['MSE']
    
    for i in patch_num:
        for l in loss:
                dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_tfm"
                print(script + ' dir-------> ' + dir)
                parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l] 
                run_script(script, parameters)

    patch_type = 'both'
    patch_num = [1]
    loss = ['MSE']
    volume = ['False', 'True']
    
    for i in patch_num:
        for l in loss:
            for v in volume:
                if v == 'False':
                    dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_tfm"
                else:
                    dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_tfm"
                print(script + ' dir-------> ' + dir)
                parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "volume", str(v)] 
                run_script(script, parameters)

    patch_type = 'ring'
    patch_num = [1]
    loss = ['MSE']
    volume = ['False', 'True']
    
    for i in patch_num:
        for l in loss:
            for v in volume:
                if v == 'False':
                    dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_tfm"
                else:
                    dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_tfm"
                print(script + ' dir-------> ' + dir)
                parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "volume", str(v)] 
                run_script(script, parameters)

    patch_type = 'arc'
    patch_num = [1]
    loss = ['MSE']
    volume = ['False', 'True']
    
    for i in patch_num:
        for l in loss:
            for v in volume:
                if v == 'False':
                    dir = "unet_" + patch_type + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_tfm"
                else:
                    dir = "unet_" + patch_type + "_volume" + "_seed" + str(seed) + "_dseed" + str(dseed) + "_num" + str(i) + "_" + l + "Loss_tfm"
                print(script + ' dir-------> ' + dir)
                parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "type", str(patch_type), "num", str(i), "loss", l, "volume", str(v)] 
                run_script(script, parameters)
    