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
    dseeds = [32, 86, 94, 6, 7, 17, 56, 5, 81, 16, 55, 60, 58, 72, 51, 64, 21, 96, 43, 10, 97, 83, 71, 54, 34, 28, 33, 38, 42, 35, 69, 89, 52, 79, 59, 49, 14, 44, 39, 62, 27, 9, 18, 2, 99, 73, 46, 63, 23, 53]

    if eConfig['cluster_id'] == 0:
        th = [0.07]
        a = [1000]
        dseed = dseeds[0:10]
        seed = [0]
        for d in dseed:
            for i in seed:
                for x in th: 
                    for y in a:
                        weights_dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_new"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                        run_script(script, parameters)
    elif eConfig['cluster_id'] == 1: 
        th = [0.07]
        a = [1000]
        dseed = dseeds[10:20]
        seed = [0]
        for d in dseed:
            for i in seed:
                for x in th: 
                    for y in a:
                        weights_dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_new"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                        run_script(script, parameters)
    elif eConfig['cluster_id'] == 2: 
        th = [0.07]
        a = [1000]
        dseed = dseeds[20:30]
        seed = [0]
        for d in dseed:
            for i in seed:
                for x in th: 
                    for y in a:
                        weights_dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_new"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                        run_script(script, parameters)
    elif eConfig['cluster_id'] == 3: 
        th = [0.07]
        a = [1000]
        dseed = dseeds[30:40]
        seed = [0]
        for d in dseed:
            for i in seed:
                for x in th: 
                    for y in a:
                        weights_dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_new"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                        run_script(script, parameters)
    elif eConfig['cluster_id'] == 4: 
        th = [0.07]
        a = [1000]
        dseed = dseeds[40:50]
        seed = [0]
        for d in dseed:
            for i in seed:
                for x in th: 
                    for y in a:
                        weights_dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_new"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                        run_script(script, parameters)
    elif eConfig['cluster_id'] == 5:
        th = [0.04]
        a = [500]
        dseed = dseeds[0:10]
        seed = [0]
        for d in dseed:
            for i in seed:
                for x in th: 
                    for y in a:
                        weights_dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_new"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                        run_script(script, parameters)
    elif eConfig['cluster_id'] == 6: 
        th = [0.04]
        a = [500]
        dseed = dseeds[10:20]
        seed = [0]
        for d in dseed:
            for i in seed:
                for x in th: 
                    for y in a:
                        weights_dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_new"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                        run_script(script, parameters)
    elif eConfig['cluster_id'] == 7: 
        th = [0.04]
        a = [500]
        dseed = dseeds[20:30]
        seed = [0]
        for d in dseed:
            for i in seed:
                for x in th: 
                    for y in a:
                        weights_dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_new"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                        run_script(script, parameters)
    elif eConfig['cluster_id'] == 8: 
        th = [0.04]
        a = [500]
        dseed = dseeds[30:40]
        seed = [0]
        for d in dseed:
            for i in seed:
                for x in th: 
                    for y in a:
                        weights_dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_new"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                        run_script(script, parameters)
    elif eConfig['cluster_id'] == 9: 
        th = [0.04]
        a = [500]
        dseed = dseeds[40:50]
        seed = [0]
        for d in dseed:
            for i in seed:
                for x in th: 
                    for y in a:
                        weights_dir = "no_deactivate" +"_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_new"
                        print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
                        parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                        run_script(script, parameters)
    