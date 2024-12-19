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

    script = "trainCVEye_5_folds.py"
    cropEyePolar = []
    deactivate = [70]
    seed = [0, 10, 20, 30 , 40]
    
    # #cropEyePolar = [20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    
    # for radius in deactivate: 
    #     dir = "deactivate" + str(radius) + ''
    #     print(script + ' dir-------> ' + dir)
    #     parameters = ["dir", dir, "deactivate", str(radius)]
    #     run_script(script, parameters)

    # for radius in cropEyePolar: 
    #     dir = "cropEyePolar" + str(radius) + ''
    #     print(script + ' dir-------> ' + dir)
    #     parameters = ["dir", dir, "cropEyePolar_px", str(radius)]
    #     run_script(script, parameters)
    

    # for i in seed: 
    #     dir = "no_deactivate" + '_seed'+ str(i)
    #     print(script + ' dir-------> ' + dir)
    #     parameters = ["dir", dir, "seed", str(i)]
    #     run_script(script, parameters)

    for j in deactivate:
        for i in seed: 
            dir = "deactivate" + str(j) + '_seed'+ str(i)
            print(script + ' dir-------> ' + dir)
            parameters = ["dir", dir, "deactivate", str(j), "seed", str(i)]
            run_script(script, parameters)
