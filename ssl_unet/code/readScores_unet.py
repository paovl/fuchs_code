import pickle
import pdb

if __name__ == '__main__':

    # read patients_val_batcgh

    path = "results"
    dir = "unet_map_seed0_dseed0_num1_MSELoss(mask_dilation3)_final/ELE_4_CORNEA-DENS_0_PAC_0"
    name_file = "val_batch_patients.pkl"

    # read file 

    file_path = path + "/" + dir + "/" + name_file

    with open(file_path, 'rb') as f:
        patients_val_batch = pickle.load(f)
    
    # print(patients_val_batch)

    r2_scores = patients_val_batch['r2_scores']

    ssim_errors = patients_val_batch['ssim_errors']
    mse_errors = patients_val_batch['mse_errors']
    mae_errors = patients_val_batch['mae_errors']
    ncc_errors = patients_val_batch['ncc_errors']
    local_ncc_errors = patients_val_batch['local_ncc_errors']

    mean_ssim = sum(ssim_errors) / len(ssim_errors)
    mean_mse = sum(mse_errors) / len(mse_errors)
    mean_mae = sum(mae_errors) / len(mae_errors)
    mean_ncc = sum(ncc_errors) / len(ncc_errors)
    mean_local_ncc = sum(local_ncc_errors) / len(local_ncc_errors)

    print("Mean SSIM: ", mean_ssim)
    print("Mean MSE: ", mean_mse)
    print("Mean MAE: ", mean_mae)
    print("Mean NCC: ", mean_ncc)
    print("Mean Local NCC: ", mean_local_ncc)


    import matplotlib.pyplot as plt
    import numpy as np

    # Create a histogram of the errors
    plt.hist(ssim_errors, bins=50, alpha=0.5, label='SSIM Errors')
    plt.hist(mse_errors, bins=50, alpha=0.5, label='MSE Errors')
    plt.xlabel('Error Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Errors')
    plt.legend()
    plt.show()
    

