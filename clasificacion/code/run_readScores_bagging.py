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

    # dir = "pretrained_noCrop"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "pretrained"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "pretrained_noPooling"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "pretrained_polarPooling_3rings"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "pretrained_polarPooling_4rings"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "pretrained_polarPooling_7rings"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "pretrained_polarPooling_3arcs"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "pretrained_polarPooling_4arcs"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "pretrained_polarPooling_6arcs"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "pretrained_polarPooling_3rings_3arcs"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "pretrained_polarPooling_4rings_4arcs"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "pretrained_polarPooling_allRingsArcs"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)


    # dir = "modified_pretrained_noCrop"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "modified_pretrained_polarPooling_3rings_fusion1"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "modified_pretrained_polarPooling_4rings_fusion1"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "modified_pretrained_polarPooling_7rings_fusion1"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "modified_pretrained_polarPooling_3arcs_fusion1"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "modified_pretrained_polarPooling_4arcs_fusion1"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "modified_pretrained_polarPooling_6arcs_fusion1"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "modified_pretrained_polarPooling_3rings_3arcs_fusion1"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "modified_pretrained_polarPooling_4rings_4arcs_fusion1"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "modified_pretrained_polarPooling_allRingsArcs_fusion1"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "modified_pretrained_allRingsArcs_fusion3"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "modified_pretrained_allRingsArcs_fusion3"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "modified_pretrained_allRingsArcs_fusion4"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "none"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "no_deactivate_noCrop_seed0_dseed0_loss(full DB)"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "no_deactivate_seed0_dseed0_loss(full DB)"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # dir = "no_deactivate_seed0_dseed0_weightedLoss(full DB)"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir_results", dir]
    # run_script(script, parameters)

    # By hospitals 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    # th = [0.04]
    # a = [500]
    # dseed = [0]
    # seed = [0]
    # hospital = ['HRyC']
    # for h in hospital:
    #     for d in dseed:
    #         for i in seed:
    #             for y in a:
    #                 for x in th:
    #                     dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_"+ h
    #                     print(script + ' data -------> ' + dir)
    #                     parameters = ["dir_results", dir]
    #                     run_script(script, parameters)

    # th = [0.07]
    # a = [1000]
    # dseed = [0]
    # seed = [0]
    # for d in dseed:
    #     for i in seed:
    #         for y in a:
    #             for x in th:
    #                 dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_bio"
    #                 print(script + ' data -------> ' + dir)
    #                 parameters = ["dir_results", dir]
    #                 run_script(script, parameters)

    th = [0.07]
    a = [1000]
    dseed = [0]
    seed = [0]
    hospital = ['HRyC', 'Cruces', 'GM', 'Clinico', 'Paz']
    for d in dseed:
        for i in seed:
            for y in a:
                for x in th:
                    for h in hospital:
                        dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_"+ h + "_90train"
                        print(script + ' data -------> ' + dir)
                        parameters = ["dir_results", dir]
                        run_script(script, parameters)

    # seed = [0]
    # dseed = [0]
    # th = [0, 0.005, 0.01, 0.015, 0.02]
    # for i in seed:
    #     for d in dseed:
    #         for x in th:
    #                 dir = "no_deactivate" + "_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_binaryWeightedLoss(full DB)"
    #                 print(script + ' data -------> ' + dir)
    #                 parameters = ["dir_results", dir]
    #                 run_script(script, parameters)   

    
    
    