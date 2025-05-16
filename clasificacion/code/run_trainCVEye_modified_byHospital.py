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

    script = "trainCVEye_modified_byHospital.py"
    dir = 'SEP'
    cfg_file = "conf.config_70iter"
    db = "global"
    data_dir = 'ransac_TH_1.5_r_45'
    # data_dir = 'original'

    seed = [0]
    dseed = [0]
    th = [0.07]
    a = [1000]
    hospital = ['HRyC', 'Cruces', 'GM', 'Clinico', 'Paz']
    for h in hospital:
        for i in seed:
            for k in dseed:
                for y in a:
                    for x in th:
                        weights_dir = "no_deactivate"+"_seed" + str(i)+ "_dseed" + str(k) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(k), "dir_weights_file", weights_dir, "hospital", h, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                        run_script(script, parameters)