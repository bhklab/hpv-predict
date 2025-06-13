import pandas as pd 
import numpy as np
import nrrd
import h5py
from pathlib import Path
from dipy.align.reslice import reslice

#################### parameter ##################################
root_dir = Path('/home/lang/Data/TCIA/head_neck/HN1')
files_df = pd.read_csv(root_dir / 'medical_data/HN1_all_files.csv')

output_name = 'HN1_files'
folder_name = 'train_data/image_data/size_111'

output_spacing = (1., 1., 1.)
#################################################################

out_dir = (root_dir / folder_name)

if out_dir.exists():
    raise ValueError('output folder exists')

out_dir.mkdir(parents=True)
with open(out_dir / 'script.py', 'w') as ff:
    ff.write(Path(__file__).read_text())
(out_dir / 'script.py').chmod(0o444)

hdf5_file = out_dir / (output_name+'.h5')

print('{} patients are used'.format(len(files_df)))

def reslice_nrrd(file_path, new_spacing):
    image, header = nrrd.read(file_path)
    new_spacing = np.array(new_spacing)

    affine = np.eye(4)
    
    space_dir = header['space directions']
    spacing = abs(space_dir[np.where(space_dir)])

    resize_factor = np.round(image.shape * (spacing / new_spacing)) / image.shape
    img_out, new_affine = reslice(image, affine, spacing, new_spacing) 
    return img_out

with h5py.File(hdf5_file, 'w') as f:
    img_grp = f.create_group('ct_images')
    sgmt_grp = f.create_group('ct_sgmt')
    
    for ii, row in files_df.iterrows():
        print('{}/{}'.format(ii+1, len(files_df))) 
        img_grp.create_dataset(str(row['identifier']),
            data=reslice_nrrd(row['ct_path'], output_spacing))
        sgmt_grp.create_dataset(str(row['identifier']),
            data=reslice_nrrd(row['gtv_path'], output_spacing))

hdf5_file.chmod(0o444)
