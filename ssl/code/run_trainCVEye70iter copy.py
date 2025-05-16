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

    script = "trainCVEye70iter.py"

    # No deactivate (original loss)
    # deactivate = -1
    # seed = 0
    # dseed = 0
    # dir = "no_deactivate_noCrop_seed" + str(seed)+ "_dseed" + str(dseed) + "_loss(full DB)"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir", dir, "seed", str(seed), "dataset_seed", str(dseed), "loss_type", "0", "fusion", "1", "loss_type", "0", "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    # run_script(script, parameters)

    # No deactivate (original Loss)
    # deactivate = -1
    # seed = 0
    # dseed = 0
    # dir = "no_deactivate_seed" + str(seed)+ "_dseed" + str(dseed) + "_loss(full DB)"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir", dir, "deactivate", str(deactivate), "seed", str(seed), "dataset_seed", str(dseed), "loss_type", "0", "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    # run_script(script, parameters)

    #No deactivate (weighted Loss)
    # deactivate = -1
    # seed = 0
    # dseed = 0
    # dir = "no_deactivate_seed" + str(seed)+ "_dseed" + str(dseed) + "_weightedLoss(full DB)"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir", dir, "deactivate", str(deactivate), "seed", str(seed), "dataset_seed", str(dseed), "loss_type", "1", "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    # run_script(script, parameters)
    
    # Deactivate crop (original loss)
    # deactivate = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # seed = [0]
    # dseed = [0]
    # for i in seed:
    #     for k in dseed:
    #         for j in deactivate:
    #             dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(k) + "_loss(full DB)"
    #             print(script + ' dir-------> ' + dir)
    #             parameters = ["dir", dir, "deactivate", str(j), "seed", str(i), "dataset_seed", str(k),"loss_type", "0", "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    #             run_script(script, parameters)
    
    # Deactivate crop (sample Weight Loss)
    # deactivate = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # seed = [0]
    # dseed = [0]
    # for i in seed:
    #     for k in dseed:
    #         for j in deactivate:
    #             dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(k) + "_weightedLoss(full DB)"
    #             print(script + ' dir-------> ' + dir)
    #             parameters = ["dir", dir, "deactivate", str(j), "seed", str(i), "dataset_seed", str(k),"loss_type", "1", "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    #             run_script(script, parameters)
    
    # No deactivate crop (original loss)
    # deactivate = -1
    # seed = 0
    # dseed = 0
    # dir = "no_deactivate_seed" + str(seed)+ "_dseed" + str(dseed) + "_loss(full DB)"
    # print(script + ' dir-------> ' + dir)
    # parameters = ["dir", dir, "deactivate", str(deactivate), "seed", str(seed), "dataset_seed", str(dseed), "loss_type", "0"]
    # run_script(script, parameters)

    # Deactivate crop (original loss)
    # deactivate = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # seed = [0]
    # dseed = [0]
    # for i in seed:
    #     for k in dseed:
    #         for j in deactivate:
    #             dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(k) + "_loss(full DB)"
    #             print(script + ' dir-------> ' + dir)
    #             parameters = ["dir", dir, "deactivate", str(j), "seed", str(i), "dataset_seed", str(k),"loss_type", "0"]
    #             run_script(script, parameters)
    
    # Deactivate crop (sample Weight Loss)
    # deactivate = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # seed = [0]
    # dseed = [0]
    # for i in seed:
    #     for k in dseed:
    #         for j in deactivate:
    #             dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(k) + "_weightedLoss(full DB)"
    #             print(script + ' dir-------> ' + dir)
    #             parameters = ["dir", dir, "deactivate", str(j), "seed", str(i), "dataset_seed", str(k),"loss_type", "1"]
    #             run_script(script, parameters)

    #Pruebas para validar deactivate con binary loss
    # deactivate = [-1]
    # seed = [0]
    # dseed = [0]
    # th = [0, 0.005, 0.01, 0.015, 0.02]
    # for i in seed:
    #     for k in dseed:
    #         for x in th:
    #             for j in deactivate:
    #                 dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(k) + "_th" + str(x) + "_binaryWeightedLoss(full DB)"
    #                 print(script + ' dir-------> ' + dir)
    #                 parameters = ["dir", dir, "deactivate", str(j), "seed", str(i), "dataset_seed", str(k), "th_weights", str(x), "loss_type", "2",  "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    #                 run_script(script, parameters)

    # #Pruebas para validar deactivate con sigmoid loss

    # create una lista de seed aleatoria de 20 elementos
    # [32, 86, 94, 6, 7, 17, 56, 5, 81, 16, 55, 60, 58, 72, 51, 64, 21, 96, 43, 10, 97, 83, 71, 54, 34, 28, 33, 38, 42, 35, 69, 89, 52, 79, 59, 49, 14, 44, 39, 62, 27, 9, 18, 2, 99, 73, 46, 63, 23, 53]
    # dseed = random.sample(range(0, 100), 50)
    # print("Random dseeds: ", dseed)
    th = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    a = [100, 200, 500, 1000]
    seed = [0]
    dseed = [0]
    for i in seed:
        for k in dseed:
            for y in a:
                for x in th:
                        dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(k) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_tfm"
                        print(script + ' dir-------> ' + dir)
                        parameters = ["dir", dir, "seed", str(i), "dataset_seed", str(k), "th_weights", str(x), "a_weights", str(y), "loss_type", "3", "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                        run_script(script, parameters)

    # deactivate = [-1]
    # th = [0.04]
    # a = [500]
    # # seed = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # seed = [0]
    # for i in seed:
    #     for k in dseed:
    #         for y in a:
    #             for x in th:
    #                 for j in deactivate:
    #                     dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(k) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_new"
    #                     print(script + ' dir-------> ' + dir)
    #                     parameters = ["dir", dir, "deactivate", str(j), "seed", str(i), "dataset_seed", str(k), "th_weights", str(x), "a_weights", str(y), "loss_type", "3", "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    #                     run_script(script, parameters)

    # deactivate = [50]
    # th = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    # a = [100, 200, 500, 10000]
    # seed = [0]
    # dseed = [0]
    # for i in seed:
    #     for k in dseed:
    #         for y in a:
    #             for x in th:
    #                 for j in deactivate:
    #                     dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(k) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                     print(script + ' dir-------> ' + dir)
    #                     parameters = ["dir", dir, "deactivate", str(j), "seed", str(i), "dataset_seed", str(k), "th_weights", str(x), "a_weights", str(y), "loss_type", "3", "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    #                     run_script(script, parameters)
    