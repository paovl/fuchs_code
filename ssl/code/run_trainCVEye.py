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

    script = "trainCVEye.py"
    deactivate = [70]
    seed = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,120, 130, 140, 150, 160, 170, 180, 190, 200]
    th = [0.07]
    a = [1000]

    # for i in seed:
    #     for y in a:
    #         for x in th:
    #             for j in deactivate:
    #                 dir = "deactivate" + str(j) +"_seed" + str(i)+ "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
    #                 print(script + ' dir-------> ' + dir)
    #                 parameters = ["dir", dir, "deactivate", str(j), "seed", str(i),"th_weights", str(x), "a_weights", str(y)]
    #                 run_script(script, parameters)

    # for i in seed: 
    #     dir = "no_deactivate" + '_seed'+ str(i) + "_noCrop(full DB)"
    #     print(script + ' dir-------> ' + dir)
    #     parameters = ["dir", dir, "seed", str(i)]
    #     run_script(script, parameters)

    # deactivate = [70]
    # seed = [50, 60, 70, 80, 90]

    # for j in deactivate:
    #     for i in seed: 
    #         dir = "deactivate" + str(j) + '_seed'+ str(i) + "_loss(full DB)"
    #         print(script + ' dir-------> ' + dir)
    #         parameters = ["dir", dir, "deactivate", str(j), "seed", str(i)]
    #         run_script(script, parameters)
    
    dseed = [0]
    for i in seed:
            for y in a:
                for x in th:
                    for j in deactivate:
                        dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(i) + "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
                        print(script + ' dir-------> ' + dir)
                        parameters = ["dir", dir, "deactivate", str(j), "seed", str(i), "dataset_seed", str(i), "th_weights", str(x), "a_weights", str(y)]
                        run_script(script, parameters)
    