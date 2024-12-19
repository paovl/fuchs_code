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

    script = "readScores_bagging.py"
    cropEyePolar = []

    deactivate = [70]
    th = [0.07]
    a = [1000]

    # dseed = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
    # seed = [200]
    # for j in deactivate:
    #     for d in dseed:
    #         for i in seed:
    #             for y in a:
    #                 for x in th:
    #                     dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
    #                     print(script + ' data -------> ' + dir)
    #                     parameters = ["dir_results", dir]
    #                     run_script(script, parameters)

    # seed = [0, 10, 20, 30, 40, 50, 60, 70 , 80, 90, 130, 140, 150, 190, 200, 210, 220, 240, 270, 300, 330, 340, 350, 360, 370, 380, 400, 100, 110, 120, 160, 170, 180, 230, 250, 260, 280, 290, 310, 320, 390]
    # for j in deactivate:
    #     for i in seed:
    #         for y in a:
    #             for x in th:
    #                 dir = "deactivate" + str(j) +"_seed" + str(i)+ "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
    #                 print(script + ' data -------> ' + dir)
    #                 parameters = ["dir_results", dir]
    #                 run_script(script, parameters)

    seed = [370, 380, 390, 400]
    dseed = [0]
    for j in deactivate:
        for d in dseed:
            for i in seed:
                for y in a:
                    for x in th:
                        dir = "deactivate" + str(j) +"_seed" + str(i) + "_dseed" + str(d)+ "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
                        print(script + ' data -------> ' + dir)
                        parameters = ["dir_results", dir]
                        run_script(script, parameters)

    dir = "none"
    print(script + ' dir-------> ' + dir)
    parameters = ["dir_results", dir]
    run_script(script, parameters)
    
    