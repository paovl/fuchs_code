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

    # dir = "no_deactivate_noCrop_seed0_dseed0_loss(full DB)"
    # parameters = ["dir", dir]
    # run_script(script, parameters)

    # dir = "no_deactivate_seed0_dseed0_weightedLoss(full DB)"
    # parameters = ["dir", dir]
    # run_script(script, parameters)

    # deactivate = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # seed = [0]
    # dseed = [0]

    # for i in seed:
    #     for d in dseed:
    #         for j in deactivate:
    #             dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_loss(full DB)"
    #             print(script + ' dir-------> ' + dir)
    #             parameters = ["dir", dir]
    #             run_script(script, parameters)

    seed = [0]
    dseed = [0]
    th = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    a = [100, 200, 500, 1000, 10000]

    for i in seed:
        for d in dseed:
            for x in th:
                for y in a:
                    dir = "no_deactivate" + "_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y)+ "_sigmoidWeightedLoss(full DB)"
                    print(script + ' dir-------> ' + dir)
                    parameters = ["dir", dir]
                    run_script(script, parameters)

    # deactivate = [50]
    # seed = [0]
    # dseed = [0]
    # th = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    # a = [100, 200, 500, 1000, 10000]

    # for i in seed:
    #     for d in dseed:
    #         for x in th:
    #             for y in a:
    #                 for j in deactivate:
    #                     dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y)+ "_sigmoidWeightedLoss(full DB)"
    #                     print(script + ' dir-------> ' + dir)
    #                     parameters = ["dir", dir]
    #                     run_script(script, parameters)
    
    # th = [0, 0.005, 0.01, 0.015]
    # for i in seed:
    #     for d in dseed:
    #         for x in th:
    #             for j in deactivate:
    #                 dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_binaryWeightedLoss(full DB)"
    #                 print(script + ' dir-------> ' + dir)
    #                 parameters = ["dir", dir]
    #                 run_script(script, parameters)
    deactivate = [50]
    # seed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990]
    # seed = [220]
    # dseed = [0]
    # th = [0.07]
    # a = [1000]
    # for i in seed:
    #     for d in dseed:
    #         for x in th:
    #             for y in a:
    #                 for j in deactivate:
    #                     dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y)+ "_sigmoidWeightedLoss(full DB)"
    #                     print(script + ' dir-------> ' + dir)
    #                     parameters = ["dir", dir]
    #                     run_script(script, parameters)

    