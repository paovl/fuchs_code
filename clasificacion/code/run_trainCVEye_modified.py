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

    script = "trainCVEye_modified.py"
    dir = 'SEP'
    cfg_file = "conf.config_70iter"
    db = "global"
    data_dir = 'ransac_TH_1.5_r_45'

    deactivate = [70]
    th = [0.07]
    a = [1000]
    # seed = [220, 240, 270, 300, 330, 340, 350, 360, 370, 380, 400, 100, 110, 120, 160, 170, 180, 230, 250, 260, 280, 290, 310, 320, 390]
    # for j in deactivate:
    #     for i in seed:
    #         for y in a:
    #             for x in th:
    #                 weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
    #                 print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
    #                 parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #                 run_script(script, parameters)

    # dseed = [130, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
    # seed = [200]
    # for j in deactivate:
    #     for d in dseed:
    #         for i in seed:
    #             for y in a:
    #                 for x in th:
    #                     weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(d) + "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
    #                     print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
    #                     parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #                     run_script(script, parameters)
    
    weights_dir = "none"
    print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- weights seed = ' + str(-1))
    parameters = ["dir", dir, "weights_seed", str(-1), "weights_dseed", str(-1)]
    run_script(script, parameters)

    seed = [380, 390, 400, 130, 140, 150, 160]
    dseed = [0]
    for j in deactivate:
        for d in dseed:
            for i in seed:
                for y in a:
                    for x in th:
                        weights_dir = "deactivate" + str(j) +"_seed" + str(i) + "_dseed" + str(d)+ "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d)]
                        run_script(script, parameters)
    
    seed = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,120, 130, 140, 150, 160, 170, 180, 190, 200]
    dseed = [0]
    for j in deactivate:
        for i in seed:
            for y in a:
                for x in th:
                    weights_dir = "deactivate" + str(j) +"_seed" + str(i) + "_dseed" + str(i)+ "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
                    print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                    parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(i)]
                    run_script(script, parameters)
    
    # deactivate = [70]
    # seed = [80, 90]
    # th = [0.06]
    # a = [1000]

    # for j in deactivate:
    #     for i in seed:
    #         for y in a:
    #             for x in th:
    #                 weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_sigmoid_th" + str(x) + "_a" + str(y) + "_loss_sampleWeight(full DB)"
    #                 print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
    #                 parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #                 run_script(script, parameters)
    
