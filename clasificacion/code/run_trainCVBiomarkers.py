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
    script = "trainCVBiomarkers_modified.py"
    dir = 'SEP'
    cfg_file = "conf.config_70iter"
    db = "global"
    bio = "0"
    data_dir = 'ransac_TH_1.5_r_45'
    results_dir = "biomarkers"
    seed = "0"

    # PRUEBA 1
    print(script + ' data -------> ' + data_dir + ' --- seed = ' + str(seed))
    parameters = ["cfg_file", cfg_file, "db", db, "dir", dir, "data_dir", data_dir,"results_dir", results_dir, "seed", seed] 
    run_script(script, parameters)

   