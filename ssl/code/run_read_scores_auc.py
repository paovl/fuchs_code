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

    script = "read_scores_auc.py"
    deactivate = [70]
    seed = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 230, 240, 250, 260, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
    th = [0.07]
    a = [1000]
    
    dseed = [0]
    for i in seed:
        for d in dseed:
            for y in a:
                for x in th:
                    for j in deactivate:
                        dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
                        print(script + ' dir-------> ' + dir)
                        parameters = ["dir", dir]
                        run_script(script, parameters)