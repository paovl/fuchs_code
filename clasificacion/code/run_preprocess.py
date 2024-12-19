"""
@author: pvltarife
"""
import subprocess
import numpy as np

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

    scriptHRYC = "preprocessDatasetHRYC.py"
    scriptMulti = "preprocessDatasetMulti.py"
    scriptFuse = "fuseDatasets.py"
    dir = 'RANSAC'


    th = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5] # thresholds to try
    r = [70, 60, 50, 45, 40, 30, 20, 10, 5]
    px = 100
    print("Data directory -------> " + dir)
    print("Threshold values: " + str(th))
    print("Disk removal radius: " + str(r))

    files = ['original', 'noRansac', 'noRansac_diskRemoval']

    for i in files:
        if i == 'original':
            var_ransac = -1
            var_disk = 0
        elif i == 'noRansac':
            var_ransac = 0
            var_disk = 0
        else: 
            var_ransac = 0
            var_disk = 20
        
        print(scriptHRYC + " for -------> " + i)
        parametersHRYC = ['dir', dir, 'th', '0', 'r', str(var_disk), 'ransac', str(var_ransac), 'pixels', str(px)]
        run_script(scriptHRYC, parametersHRYC)
        print(scriptMulti + " for -------> " + i )
        parametersMulti = ['dir', dir, 'th', '0', 'r', str(var_disk), 'ransac', str(var_ransac), 'pixels', str(px)]
        run_script(scriptMulti, parametersMulti)
        parameters_fuse = ['dir', dir, 'input_file', i]
        run_script(scriptFuse, parameters_fuse)

    for i in r:
        for j in th:
            print(scriptHRYC + " for -------> r = " + str(i) + ", th = " + str(j))
            parametersHRYC = ['dir', dir, 'th', str(j), 'r', str(i), 'ransac', '1', 'pixels', str(px)]
            run_script(scriptHRYC, parametersHRYC)
            print(scriptMulti + " for -------> r = " + str(i) + ", th = " + str(j))
            parametersMulti = ['dir', dir, 'th', str(j), 'r', str(i), 'ransac', '1', 'pixels', str(px)]
            run_script(scriptMulti, parametersMulti)
            parameters_fuse = ['dir', dir, 'input_file', 'ransac_TH_' + str(j) + '_r_'+ str(i), 'pixels', str(px)]
            run_script(scriptFuse, parameters_fuse)

    print("FINISHED EXECUTION")
 