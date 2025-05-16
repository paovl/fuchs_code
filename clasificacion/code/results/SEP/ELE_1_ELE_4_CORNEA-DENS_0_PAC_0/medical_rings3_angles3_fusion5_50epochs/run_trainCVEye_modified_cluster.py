"""
@author: pvltarife
"""
import os
import subprocess
import random
import sys

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

    script = "trainCVEye_modified.py"
    dir = 'SEP'
    cfg_file = "conf.config_70iter"
    db = "global"
    data_dir = 'ransac_TH_1.5_r_45'

    # Config file
    eConfig = {
        'cluster_id' : -1,
    }

    args = sys.argv[1::]
    for i in range(0,len(args),2):
        key = args[i]
        val = args[i+1]
        eConfig[key] = type(eConfig[key])(val)
        print (str(eConfig[key]))
          
    print('eConfig')
    print(eConfig)

    if eConfig['cluster_id'] == 0:
        deactivate = [10]
        seed = [0]
        dseed = [0]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_loss(full DB)"
                    print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                    parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                    run_script(script, parameters)
    elif eConfig['cluster_id'] == 1:
        deactivate = [20]
        seed = [0]
        dseed = [0]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_loss(full DB)"
                    print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                    parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                    run_script(script, parameters)
    elif eConfig['cluster_id'] == 2:
        deactivate = [30]
        seed = [0]
        dseed = [0]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_loss(full DB)"
                    print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                    parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                    run_script(script, parameters)
    elif eConfig['cluster_id'] == 3:
        deactivate = [40]
        seed = [0]
        dseed = [0]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_loss(full DB)"
                    print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                    parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                    run_script(script, parameters)
    elif eConfig['cluster_id'] == 4:
        deactivate = [50]
        seed = [0]
        dseed = [0]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_loss(full DB)"
                    print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                    parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                    run_script(script, parameters)

    elif eConfig['cluster_id'] == 5:
        deactivate = [60]
        seed = [0]
        dseed = [0]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_loss(full DB)"
                    print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                    parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                    run_script(script, parameters)

    elif eConfig['cluster_id'] == 6:
        deactivate = [70]
        seed = [0]
        dseed = [0]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_loss(full DB)"
                    print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                    parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                    run_script(script, parameters)
    
    elif eConfig['cluster_id'] == 7:
        deactivate = [80]
        seed = [0]
        dseed = [0]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_loss(full DB)"
                    print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                    parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                    run_script(script, parameters)

    elif eConfig['cluster_id'] == 8:
        deactivate = [90]
        seed = [0]
        dseed = [0]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_loss(full DB)"
                    print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                    parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                    run_script(script, parameters)


    elif eConfig['cluster_id'] == 9:
        deactivate = [100]
        seed = [0]
        dseed = [0]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_loss(full DB)"
                    print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                    parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                    run_script(script, parameters)

    elif eConfig['cluster_id'] == 10:
        deactivate = []
        seed = [0]
        dseed = [0]
        th = [0.01]
        a = [100, 200, 500, 1000, 10000]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    for y in a:
                        for x in th:
                            weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
                            print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                            parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                            run_script(script, parameters)
    
    elif eConfig['cluster_id'] == 11:
        deactivate = []
        seed = [0]
        dseed = [0]
        th = [0.02]
        a = [100, 200, 500, 1000, 10000]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    for y in a:
                        for x in th:
                            weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
                            print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                            parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                            run_script(script, parameters)
    
    elif eConfig['cluster_id'] == 12:
        deactivate = []
        seed = [0]
        dseed = [0]
        th = [0.03]
        a = [100, 200, 500, 1000, 10000]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    for y in a:
                        for x in th:
                            weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
                            print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                            parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                            run_script(script, parameters)
    elif eConfig['cluster_id'] == 13:
        deactivate = []
        seed = [0]
        dseed = [0]
        th = [0.04]
        a = [100, 200, 500, 1000, 10000]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    for y in a:
                        for x in th:
                            weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
                            print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                            parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                            run_script(script, parameters)
    elif eConfig['cluster_id'] == 14:
        deactivate = []
        seed = [0]
        dseed = [0]
        th = [0.05]
        a = [100, 200, 500, 1000, 10000]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    for y in a:
                        for x in th:
                            weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
                            print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                            parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                            run_script(script, parameters)
    
    elif eConfig['cluster_id'] == 15:
        deactivate = []
        seed = [0]
        dseed = [0]
        th = [0.06]
        a = [100, 200, 500, 1000, 10000]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    for y in a:
                        for x in th:
                            weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
                            print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                            parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                            run_script(script, parameters)
    
    elif eConfig['cluster_id'] == 16:
        deactivate = []
        seed = [0]
        dseed = [0]
        th = [0.07]
        a = [100, 200, 500, 1000, 10000]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    for y in a:
                        for x in th:
                            weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
                            print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                            parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                            run_script(script, parameters)
    
    elif eConfig['cluster_id'] == 17:
        deactivate = []
        seed = [0]
        dseed = [0]
        th = [0.08]
        a = [100, 200, 500, 1000, 10000]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    for y in a:
                        for x in th:
                            weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
                            print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                            parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                            run_script(script, parameters)
    
    elif eConfig['cluster_id'] == 18:
        deactivate = []
        seed = [0]
        dseed = [0]
        th = [0.09]
        a = [100, 200, 500, 1000, 10000]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    for y in a:
                        for x in th:
                            weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
                            print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                            parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                            run_script(script, parameters)
    
    elif eConfig['cluster_id'] == 19:
        deactivate = []
        seed = [0]
        dseed = [0]
        th = [0.1]
        a = [100, 200, 500, 1000, 10000]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    for y in a:
                        for x in th:
                            weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
                            print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                            parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                            run_script(script, parameters)
    
    # if eConfig['cluster_id'] == 0:
    #     deactivate = [50]
    #     th = [0.07]
    #     a = [1000]
    #     seed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #     dseed = [0]
    #     for j in deactivate:
    #         for d in dseed:
    #             for i in seed:
    #                 for y in a:
    #                     for x in th:
    #                         weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
    #                         parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #                         run_script(script, parameters)
    # elif eConfig['cluster_id'] == 1:
    #     deactivate = [50]
    #     th = [0.07]
    #     a = [1000]
    #     seed = [110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    #     dseed = [0]
    #     for j in deactivate:
    #         for d in dseed:
    #             for i in seed:
    #                 for y in a:
    #                     for x in th:
    #                         weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
    #                         parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #                         run_script(script, parameters)
    # elif eConfig['cluster_id'] == 2:
    #     deactivate = [50]
    #     th = [0.07]
    #     a = [1000]
    #     seed = [210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
    #     dseed = [0]
    #     for j in deactivate:
    #         for d in dseed:
    #             for i in seed:
    #                 for y in a:
    #                     for x in th:
    #                         weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
    #                         parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #                         run_script(script, parameters)
    # elif eConfig['cluster_id'] == 3:
    #     deactivate = [50]
    #     th = [0.07]
    #     a = [1000]
    #     seed = [310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
    #     dseed = [0]
    #     for j in deactivate:
    #         for d in dseed:
    #             for i in seed:
    #                 for y in a:
    #                     for x in th:
    #                         weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
    #                         parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #                         run_script(script, parameters)
    # elif eConfig['cluster_id'] == 4:
    #     deactivate = [50]
    #     th = [0.07]
    #     a = [1000]
    #     seed = [410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
    #     dseed = [0]
    #     for j in deactivate:
    #         for d in dseed:
    #             for i in seed:
    #                 for y in a:
    #                     for x in th:
    #                         weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
    #                         parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #                         run_script(script, parameters)
    
    # elif eConfig['cluster_id'] == 5:
    #     deactivate = [50]
    #     th = [0.07]
    #     a = [1000]
    #     seed = [510, 520, 530, 540, 550, 560, 570, 580, 590, 600]
    #     dseed = [0]
    #     for j in deactivate:
    #         for d in dseed:
    #             for i in seed:
    #                 for y in a:
    #                     for x in th:
    #                         weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
    #                         parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #                         run_script(script, parameters)
    # elif eConfig['cluster_id'] == 6:
    #     deactivate = [50]
    #     th = [0.07]
    #     a = [1000]
    #     seed = [610, 620, 630, 640, 650, 660, 670, 680, 690, 700]
    #     dseed = [0]
    #     for j in deactivate:
    #         for d in dseed:
    #             for i in seed:
    #                 for y in a:
    #                     for x in th:
    #                         weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
    #                         parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #                         run_script(script, parameters)

    # elif eConfig['cluster_id'] == 7:
    #     deactivate = [50]
    #     th = [0.07]
    #     a = [1000]
    #     seed = [710, 720, 730, 740, 750, 760, 770, 780, 790, 800]
    #     dseed = [0]
    #     for j in deactivate:
    #         for d in dseed:
    #             for i in seed:
    #                 for y in a:
    #                     for x in th:
    #                         weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
    #                         parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #                         run_script(script, parameters)
    # elif eConfig['cluster_id'] == 8:
    #     deactivate = [50]
    #     th = [0.07]
    #     a = [1000]
    #     seed = [810, 820, 830, 840, 850, 860, 870, 880, 890, 900]
    #     dseed = [0]
    #     for j in deactivate:
    #         for d in dseed:
    #             for i in seed:
    #                 for y in a:
    #                     for x in th:
    #                         weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
    #                         parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #                         run_script(script, parameters)
    # elif eConfig['cluster_id'] == 9:
    #     deactivate = [50]
    #     th = [0.07]
    #     a = [1000]
    #     seed = [910, 920, 930, 940, 950, 960, 970, 980, 990, 1000]
    #     dseed = [0]
    #     for j in deactivate:
    #         for d in dseed:
    #             for i in seed:
    #                 for y in a:
    #                     for x in th:
    #                         weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                         print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
    #                         parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #                         run_script(script, parameters)