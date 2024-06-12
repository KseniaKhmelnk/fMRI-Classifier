import nibabel as nib
import h5py
import os
import numpy as np

def min_max_norm(array):
    max = np.max(array)
    min = np.min(array)
    max_range = np.max([max, -min])
    return array / max_range


def normalization():

  print("Normalization start...")
  all_list_name = "testData_nrcki.txt"
  folder_path = "./data"
  # List all files in the given folder
  for filename in os.listdir(folder_path):
    # Construct full file path
    nii_data_path = os.path.join(folder_path, filename)
    nii = nib.load(os.path.join(nii_data_path, "melodic_IC.nii.gz"))
    nii2array = np.asarray(nii.dataobj)
    with open(os.path.join(nii_data_path, "melodic_mix"), "r") as f:
        ts = np.array([[float(i) for i in line.split()] for line in f.readlines()])
      
    hdf_data_path = f"./nrcki/{filename}"
    os.makedirs(hdf_data_path, exist_ok=True)
    length = nii2array.shape[-1]
    for i in range(length): # comp
        s_IC = nii2array[:,:,:,i]
        t_IC = ts[:,i]

        s_IC = min_max_norm(s_IC)
        t_IC = min_max_norm(t_IC)
        s_IC = np.expand_dims(s_IC, axis=(0, 1))
        t_IC = np.expand_dims(t_IC, axis=(0, 1))
          
        dic = {"data":s_IC, "tdata":t_IC}
        with h5py.File(os.path.join(hdf_data_path, "comp{:03}.hdf5".format(i+1)), 'w') as f:
            f.create_dataset("data", data=s_IC)
            f.create_dataset("tdata", data=t_IC)

          
        with open(os.path.join("./path2files", all_list_name), "a") as f:
            f.write(os.path.join(hdf_data_path, "comp{:03}.hdf5\n".format(i + 1)))
        
        print(f"Component {i + 1} of {filename} is extracted!")

    
if __name__ == "__main__":
    normalization()
    print("Normalization is finished! You can proceed to inference!")
