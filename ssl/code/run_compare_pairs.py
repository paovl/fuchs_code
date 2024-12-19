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

    script = "compare_pairs.py"
    cropEyePolar = []
    deactivate = [70]
    seed = [0]
    
    for i in seed:
        for radius in deactivate: 
            dir = "deactivate" + str(radius) + '_seed' + str(i)
            print(script + ' dir-------> ' + dir)
            parameters = ["dir", dir]
            run_script(script, parameters)

    # dir = "no_deactivate"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir", dir]
    # run_script(script, parameters)

    for radius in cropEyePolar: 
        dir = "cropEyePolar" + str(radius) + ''
        print(script + ' dir-------> ' + dir)
        parameters = ["dir", dir]
        run_script(script, parameters)