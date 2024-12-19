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

    script = "trainCVEye_modified_perHospital.py"
    dir = 'RESNET'
    cfg_file = "conf.config_70iter"
    db = "global"
    data_dir = 'ransac_TH_1.5_r_45'
    seed = "0"

    random_seed = 0

    # PRUEBA 1
    print(script + ' data -------> ' + data_dir + ' --- seed = ' + seed)
    parameters = ["cfg_file", cfg_file, "db", db, "bio", "0", "dir", dir, "data_dir", data_dir, "results_dir", "medical_rings3_angles3_fusion5_epochs50", "seed", seed, "type", "2", "fusion", "5", "delete_last_layer", "1", "rings", "1", "3", "5", "angles", "120", "240", "360", "max_radius", "7", "max_angle", "360"]
    run_script(script, parameters)


