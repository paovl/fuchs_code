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

    deactivate = [40]
    seed = 0
    dseed = 0

    for j in deactivate:
            dir = "deactivate" + str(j) +"_seed" + str(seed)+ "_dseed" + str(dseed) + "_loss(full DB)"
            print(script + ' data -------> ' + dir)
            parameters = ["dir_results", dir]
            run_script(script, parameters)

    # for j in deactivate:
    #         dir = "deactivate" + str(j) +"_seed" + str(seed)+ "_dseed" + str(dseed) + "_weightedLoss(full DB)"
    #         print(script + ' data -------> ' + dir)
    #         parameters = ["dir_results", dir]
    #         run_script(script, parameters)
        
    deactivate = [70]
    th = [0.07]
    a = [1000]
    dseed = [0]
    #seed = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
    seed = [0, 10, 310, 320, 330]
    # for j in deactivate:
    #     for d in dseed:
    #         for i in seed:
    #             for y in a:
    #                 for x in th:
    #                     dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
    #                     print(script + ' data -------> ' + dir)
    #                     parameters = ["dir_results", dir]
    #                     run_script(script, parameters)
    

    #seed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000]
    seed = [210, 220, 560, 570, 810, 820]
    # for j in deactivate:
    #         for i in seed:
    #             for y in a:
    #                 for x in th:
    #                     dir = "deactivate" + str(j) +"_seed" + str(i) + "_dseed" + str(i)+ "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
    #                     print(script + ' data -------> ' + dir)
    #                     parameters = ["dir_results", dir]
    #                     run_script(script, parameters)

    # dir = "none"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)
    
    