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

    deactivate = [70]
    th = [0.07]
    a = [1000]

    if eConfig['cluster_id'] == 0:
        seed = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390]
        dseed = [0]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    for y in a:
                        for x in th:
                            weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
                            print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                            parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d)]
                            run_script(script, parameters)
    elif eConfig['cluster_id'] == 1:
        seed = [370]
        dseed = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
        for j in deactivate:
            for d in dseed:
                for i in seed:
                    for y in a:
                        for x in th:
                            weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
                            print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                            parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d)]
                            run_script(script, parameters)
    elif eConfig['cluster_id'] == 2:
        seed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 380, 390]
        for j in deactivate:
            for i in seed:
                for y in a:
                    for x in th:
                        weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(i) + "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(i)]
                        run_script(script, parameters)
    elif eConfig['cluster_id'] == 3:
        seed = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800]
        for j in deactivate:
            for i in seed:
                for y in a:
                    for x in th:
                        weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(i) + "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(i)]
                        run_script(script, parameters)
    elif eConfig['cluster_id'] == 4:
        seed = [810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990]
        for j in deactivate:
            for i in seed:
                for y in a:
                    for x in th:
                        weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(i) + "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(i)]
                        run_script(script, parameters)