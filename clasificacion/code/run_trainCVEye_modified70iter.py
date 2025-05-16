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

    # script = "trainCVEye_modified.py"
    dir = 'SEP'
    data_dir = 'ransac_TH_1.5_r_45'

    script = "trainCVEye_modified70iter.py"
    seed = [0]
    dseed = [0]
    th = [0]
    a = [100]
    # th = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    # a = [100, 200, 500, 1000]
    for i in seed:
        for k in dseed:
            for y in a:
                for x in th:
                    weights_dir = "no_deactivate"+"_seed" + str(i)+ "_dseed" + str(k) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)_tfm"
                    print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
                    parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(k), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
                    run_script(script, parameters)
    
    # script = "trainCVEye_modified65.py"
    # seed = [0]
    # dseed = [0]
    # th = [0.04]
    # a = [500]
    # for i in seed:
    #     for k in dseed:
    #         for y in a:
    #             for x in th:
    #                 weights_dir = "no_deactivate"+"_seed" + str(i)+ "_dseed" + str(k) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                 print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #                 parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(k), "dir_weights_file", weights_dir, "bio", "1", "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    #                 run_script(script, parameters)

    # script = "trainCVEye_modified70.py"
    # seed = [0]
    # dseed = [0]
    # th = [0.07]
    # a = [1000]
    # for i in seed:
    #     for k in dseed:
    #         for y in a:
    #             for x in th:
    #                 weights_dir = "no_deactivate"+"_seed" + str(i)+ "_dseed" + str(k) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                 print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #                 parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(k), "dir_weights_file", weights_dir, "bio", "1", "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    #                 run_script(script, parameters)

    # script = "trainCVEye_modified65.py"
    # seed = [0]
    # dseed = [0]
    # th = [0.07]
    # a = [1000]
    # for i in seed:
    #     for k in dseed:
    #         for y in a:
    #             for x in th:
    #                 weights_dir = "no_deactivate"+"_seed" + str(i)+ "_dseed" + str(k) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                 print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #                 parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(k), "dir_weights_file", weights_dir, "bio", "1", "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    #                 run_script(script, parameters)

    # Expriments for baseline architecture (average pool)


    # weights_dir = "pretrained_noCrop"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, 'type', '0', 'fusion', '0', 'delete_last_layer', '0']
    # run_script(script, parameters)

    # weights_dir = "pretrained"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, 'type', '0', 'fusion', '0', 'delete_last_layer', '0']
    # run_script(script, parameters)

    # weights_dir = "pretrained(original maps)"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, 'type', '0', 'fusion', '0', 'delete_last_layer', '0']
    # run_script(script, parameters)

    # Experiments for baseline architecture (no pool)
    # weights_dir = "pretrained_noPooling"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, 'type', '1', 'fusion', '0', 'delete_last_layer', '0']
    # run_script(script, parameters)

    # weights_dir = "pretrained_noPooling"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, 'type', '0', 'fusion', '0', 'delete_last_layer', '0']
    # run_script(script, parameters)

    # weights_dir = "pretrained_polarPooling_3rings_3arcs"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, 'type', '2', 'fusion', '0', 'delete_last_layer', '1']
    # run_script(script, parameters)

    # weights_dir = "pretrained_polarPooling_allRingsArcs"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, 'type', '2', 'fusion', '0', 'delete_last_layer', '1', "rings", "0", "arcs", "0"]
    # run_script(script, parameters)

    # # # 3 rings
    # weights_dir = "pretrained_polarPooling_3rings"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, 'type', '2', 'fusion', '0', 'delete_last_layer', '1', "rings", "1", "3", "5", "arcs", "360"]
    # run_script(script, parameters)

    # # # 4 rings
    # weights_dir = "pretrained_polarPooling_4rings"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, 'type', '2', 'fusion', '0', 'delete_last_layer', '1', "rings", "1", "3", "5", "6", "arcs", "360"]
    # run_script(script, parameters)

    # # # 7 rings
    # weights_dir = "pretrained_polarPooling_7rings"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, 'type', '2', 'fusion', '0', 'delete_last_layer', '1', "rings", "1", "2", "3", "4", "5", "6", "7", "arcs", "360"]
    # run_script(script, parameters)

    # # # 3 arcs
    # weights_dir = "pretrained_polarPooling_3arcs"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, 'type', '2', 'fusion', '0', 'delete_last_layer', '1', "rings", "7", "arcs", "120", "240", "360"]
    # run_script(script, parameters)

    # # # 4 rings
    # weights_dir = "pretrained_polarPooling_4arcs"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, 'type', '2', 'fusion', '0', 'delete_last_layer', '1', "rings", "7", "arcs", "90", "180", "270", "360"]
    # run_script(script, parameters)

    # # # 6 arcs
    # weights_dir = "pretrained_polarPooling_6arcs"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, 'type', '2', 'fusion', '0', 'delete_last_layer', '1', "rings", "7", "arcs", "60", "120", "180", "240", "300", "360"]
    # run_script(script, parameters)

    # Experiments for modified architecture ( two branch)

    # # 3 medical rings and 3 angular 3arcs
    # weights_dir = "modified_pretrained"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir]
    # run_script(script, parameters)

    # weights_dir = "modified_pretrained_noCrop"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir]
    # run_script(script, parameters)

    # # fusion 1

    # weights_dir = "modified_pretrained_polarPooling_3rings"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "arcs", "360"]
    # run_script(script, parameters)

    # weights_dir = "modified_pretrained_polarPooling_4rings_fusion1"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "360"]
    # run_script(script, parameters)

    # weights_dir = "modified_pretrained_polarPooling_7rings"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "2", "3", "4", "5", "6", "7", "arcs", "360"]
    # run_script(script, parameters)

    # weights_dir = "modified_pretrained_polarPooling_3arcs"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "1", "rings", "7", "arcs", "120", "240", "360"]

    # weights_dir = "modified_pretrained_polarPooling_4arcs"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "1", "rings", "7", "arcs", "90", "180", "270", "360"]

    # weights_dir = "modified_pretrained_polarPooling_6arcs"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "1", "rings", "7", "arcs", "60", "120", "180", "240", "300", "360"]

    # weights_dir = "modified_pretrained_polarPooling_3rings_3arcs_fusion1"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "1"]
    # run_script(script, parameters)

    # weights_dir = "modified_pretrained_polarPooling_4rings_4arcs_fusion1"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]

    # weights_dir = "modified_pretrained_polarPooling_allRingsArcs_fusion1"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "1", "rings", "0", "arcs", "0"]
    # run_script(script, parameters)

    # # fusion 2

    # weights_dir = "modified_pretrained_polarPooling_3rings_fusion2"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "2", "rings", "1", "3", "5", "arcs", "360"]

    # weights_dir = "modified_pretrained_polarPooling_4rings_fusion2"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "2", "rings", "1", "3", "5", "6", "arcs", "360"]

    # weights_dir = "modified_pretrained_polarPooling_7rings_fusion2"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "2", "rings", "1", "2", "3", "4", "5", "6", "7", "arcs", "360"]

    # weights_dir = "modified_pretrained_polarPooling_3arcs_fusion2"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "2", "rings", "7", "arcs", "120", "240", "360"]

    # weights_dir = "modified_pretrained_polarPooling_4arcs_fusion2"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "2", "rings", "7", "arcs", "90", "180", "270", "360"]

    # weights_dir = "modified_pretrained_polarPooling_6arcs_fusion2"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "2", "rings", "7", "arcs", "60", "120", "180", "240", "300", "360"]

    # weights_dir = "modified_pretrained_polarPooling_3rings_3arcs_fusion2"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "2"]

    # weights_dir = "modified_pretrained_polarPooling_4rings_4arcs_fusion2"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "2", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]

    # weights_dir = "modified_pretrained_polarPooling_allRingsArcs_fusion2"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "2", "rings", "0", "arcs", "0"]
    # run_script(script, parameters)

    # # fusion 3

    # weights_dir = "modified_pretrained_polarPooling_3rings_fusion3"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "3", "rings", "1", "3", "5", "arcs", "360"]

    # weights_dir = "modified_pretrained_polarPooling_4rings_fusion3"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "3", "rings", "1", "3", "5", "6", "arcs", "360"]

    # weights_dir = "modified_pretrained_polarPooling_7rings_fusion3"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "3", "rings", "1", "2", "3", "4", "5", "6", "7", "arcs", "360"]

    # weights_dir = "modified_pretrained_polarPooling_3arcs_fusion3"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "3", "rings", "7", "arcs", "120", "240", "360"]

    # weights_dir = "modified_pretrained_polarPooling_4arcs_fusion3"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "3", "rings", "7", "arcs", "90", "180", "270", "360"]

    # weights_dir = "modified_pretrained_polarPooling_6arcs_fusion3"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "3", "rings", "7", "arcs", "60", "120", "180", "240", "300", "360"]

    # weights_dir = "modified_pretrained_polarPooling_3rings_3arcs_fusion3"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "3"]

    # weights_dir = "modified_pretrained_polarPooling_4rings_4arcs_fusion3"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "3", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]

    # weights_dir = "modified_pretrained_polarPooling_allRingsArcs_fusion3"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "3", "rings", "0", "arcs", "0"]
    # run_script(script, parameters)

    # # fusion 4
    
    # weights_dir = "modified_pretrained_polarPooling_3rings_fusion4"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "4", "rings", "1", "3", "5", "arcs", "360"]

    # weights_dir = "modified_pretrained_polarPooling_4rings_fusion4"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "4", "rings", "1", "3", "5", "6", "arcs", "360"]

    # weights_dir = "modified_pretrained_polarPooling_7rings_fusion4"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "4", "rings", "1", "2", "3", "4", "5", "6", "7", "arcs", "360"]

    # weights_dir = "modified_pretrained_polarPooling_3arcs_fusion4"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "4", "rings", "7", "arcs", "120", "240", "360"]

    # weights_dir = "modified_pretrained_polarPooling_4arcs_fusion4"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "4", "rings", "7", "arcs", "90", "180", "270", "360"]

    # weights_dir = "modified_pretrained_polarPooling_6arcs_fusion4"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "4", "rings", "7", "arcs", "60", "120", "180", "240", "300", "360"]

    # weights_dir = "modified_pretrained_polarPooling_3rings_3arcs_fusion4"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "4"]

    # weights_dir = "modified_pretrained_polarPooling_4rings_4arcs_fusion4"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "4", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]

    # weights_dir = "modified_pretrained_polarPooling_allRingsArcs_fusion4"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "4", "rings", "0", "arcs", "0"]
    # run_script(script, parameters)

    # # fusion 5

    # weights_dir = "modified_pretrained_polarPooling_3rings_fusion5"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "5", "rings", "1", "3", "5", "arcs", "360"]

    # weights_dir = "modified_pretrained_polarPooling_4rings_fusion5"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "5", "rings", "1", "3", "5", "6", "arcs", "360"]

    # weights_dir = "modified_pretrained_polarPooling_7rings_fusion5"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "5", "rings", "1", "2", "3", "4", "5", "6", "7", "arcs", "360"]

    # weights_dir = "modified_pretrained_polarPooling_3arcs_fusion5"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "5", "rings", "7", "arcs", "120", "240", "360"]

    # weights_dir = "modified_pretrained_polarPooling_4arcs_fusion5"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "5", "rings", "7", "arcs", "90", "180", "270", "360"]

    # weights_dir = "modified_pretrained_polarPooling_6arcs_fusion5"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "5", "rings", "7", "arcs", "60", "120", "180", "240", "300", "360"]

    # weights_dir = "modified_pretrained_polarPooling_3rings_3arcs_fusion5"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "5"]

    # weights_dir = "modified_pretrained_polarPooling_4rings_4arcs_fusion5"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "5", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]

    # weights_dir = "modified_pretrained_polarPooling_allRingsArcs_fusion5"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "5", "rings", "0", "arcs", "0"]
    # run_script(script, parameters)

    # # All rings and arcs (all rings and arcs) Problem: For every image the 1 pixel region represent different in milimeters thanks to the crop Eye
    # weights_dir = "modified_pretrained_allRingsArcs"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "rings", "0", "arcs", "0"]
    # run_script(script, parameters)

    # Using self supervised learning

    # weights_dir = "no_deactivate_noCrop_seed0_dseed0_loss(full DB)"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    # run_script(script, parameters)

    # weights_dir = "no_deactivate_seed0_dseed0_loss(full DB)"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    # run_script(script, parameters)

    # weights_dir = "no_deactivate_seed0_dseed0_weightedLoss(full DB)"
    # print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    # parameters = ["dir", dir, "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    # run_script(script, parameters)


    # deactivate = [10, 20, 30, 40, 50, 60, 70]
    # for j in deactivate:
    #     weights_dir = "deactivate" + str(j) +"_seed0_dseed0_loss(full DB)"
    #     print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #     parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #     run_script(script, parameters)
    
    # deactivate = [10, 20, 30, 40, 50, 60, 70]
    # for j in deactivate:
    #     weights_dir = "deactivate" + str(j) +"_seed0_dseed0_weightedLoss(full DB)"
    #     print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #     parameters = ["dir", dir, "dir_weights_file", weights_dir]
    #     run_script(script, parameters) 

    # deactivate = [50]
    # seed = [0]
    # dseed = [0]
    # th = [0.07]
    # a = [1000, 100, 200, 500, 10000]
    # for i in seed:
    #     for k in dseed:
    #         for y in a:
    #             for x in th:
    #                 for j in deactivate:
    #                     weights_dir = "deactivate" + str(j) +"_seed" + str(i)+ "_dseed" + str(k) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                     print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #                     parameters = ["dir", dir, "dir_weights_file", weights_dir,  "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    #                     run_script(script, parameters)


    # Best but bio
    # seed = [0]
    # dseed = [0]
    # th = [0.07]
    # a = [1000]
    # for i in seed:
    #     for k in dseed:
    #         for y in a:
    #             for x in th:
    #                 weights_dir = "no_deactivate"+"_seed" + str(i)+ "_dseed" + str(k) + "_th" + str(x) + "_a" + str(y) + "_sigmoidWeightedLoss(full DB)"
    #                 print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir)
    #                 parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(k), "dir_weights_file", weights_dir, "bio", "1", "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    #                 run_script(script, parameters)


    # Binary weighted loss

    # seed = [0]
    # dseed = [0]
    # th = [0, 0.005, 0.01, 0.015, 0.02]
    # for i in seed:
    #     for d in dseed:
    #         for x in th:
    #                 weights_dir = "no_deactivate" + "_seed" + str(i)+ "_dseed" + str(d) + "_th" + str(x) + "_binaryWeightedLoss(full DB)"
    #                 print(script + ' data -------> ' + data_dir + '     weights ssl -------> ' + weights_dir + ' --- seed = ' + str(i))
    #                 parameters = ["dir", dir, "weights_seed", str(i), "weights_dseed", str(d), "dir_weights_file", weights_dir, "fusion", "1", "rings", "1", "3", "5", "6", "arcs", "90", "180", "270", "360"]
    #                 run_script(script, parameters)