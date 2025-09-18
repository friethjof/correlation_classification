import os
import shutil
import subprocess
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# This module should collect the data from the AFS directory (Uni-Hamburg) 
# and store it in a comprimised form.


# source directory

# path of hamilt:

path_src = Path('/afs/physnet.uni-hamburg.de/project/zoq_t/Driving_BJJ/fbb_mlx_N2/')

path_mlx_AppImage = Path('/home/friethjof/Documents/qdtk-0.5.7.AppImage')

path_makeOp =  path_src/'makeOperator.py'

path_pwd = Path(os.getcwd())

path_data = path_pwd/'data'

dof1 = 2
dof2 = 3



# procedure
# 1. iter through directory (divided in batches)
# 2. copy makeOperator.py in each directory
# 3. execute makeOperator script
# 4. call qdtk_analysis.x and generate dmat2
# 5. convert dmat2 into npz
# 6. create correlation matrix by loading gpop
# 7. copy npz to destination
# 8. delete makeOperator.py, hamilt and raw dmat2-file

dm2_name = f'dmat2_dof{dof1}_dof{dof2}_grid'
n = 400
time = [0]
grid = np.linspace(-5, 5, 400)



parser = argparse.ArgumentParser()
parser.add_argument("-n", help="number of batch")
args = parser.parse_args()


#-------------------------------------------------------------------------------
# helper functions
def get_1bd():
    """Read in text-file where gpop file is stored and convert to numpy arrays.
    Assume that the gpop-file has the name 'gpop' and is located in the 
    current working directory.
    Assume 3 species with densities separated by number of grid points n
    
    Args:
        None
    
    Returns:
        Three 1-dimensional array of shapes (n,), (n,), (n,)
    """
    """Get and return the density population of species A (B) from path_gpop
    E.g. gpopA.shape = (2002,300); len(gpopA) = 2002."""

    gpop_file = pd.read_csv('gpop', delim_whitespace=True, header=None,
        names=['a','b','c'])
    gpop_file = gpop_file.values

    tsteps = len(gpop_file)/(3*n+4)
    gpop_file = np.array(np.split(gpop_file, tsteps))
    gpopA = gpop_file[:, 2:2+n, 1]
    gpopA = gpopA.astype(dtype='float64')
    gpopB = gpop_file[:, 3+n:3+n*2, 1]
    gpopB = gpopB.astype(dtype='float64')
    gpopC = gpop_file[:, 4+n*2:4+n*3, 1]
    gpopC = gpopC.astype(dtype='float64')
    # timesteps = gpop_file[:, 0, 1]
    # timesteps = timesteps.astype(dtype='float64')
    return gpopA, gpopB, gpopC



def read_in_dm2(path_dm2f_raw):
    """Read in text-file where dmat2 file is stored and convert to numpy array.

    Args:
        path_dm2f_raw (Path): path of the dmat2 file
    
    Returns:
        Two-dim numpy array containing the two-body density and shape (n, n)
    """


    # read in dmat-file
    dmat2_f = pd.read_csv(path_dm2f_raw, delim_whitespace=True, header=None)
    dmat2_f = dmat2_f.values
    # dmat2_f.shape = (18090000, 4)
    dmat2_f = np.array(np.split(dmat2_f, len(time), axis=0))
    # dmat2_f.shape = (201, 90000, 4)
    dmat2 = np.zeros((len(time), n, n))
    for t in range(len(time)):
        for j in range(n):
            for k in range(n):
                dmat2[t, j, k] = dmat2_f[t, (j*n+k), 3]

    return dmat2[0]


def generate_and_copy_files(src_top_dir):
    """Iterate through sub-folders in batch folder and perform the procedure as
    described above.

    Args:
        src_top_dir (Path): path of the batch directroy
    
    Returns:
        None
    """

    batch_ID = src_top_dir.split('_')[1][-3:]

    i = 0
    for path_dir in (path_src/src_top_dir).iterdir():
        i += 1

        path_dest = path_data/f'batch_{batch_ID}'/f'run_{i:05d}'

        if not (path_dir/'psi').is_file():
            continue
 
        print(i)
        if path_dest.is_dir():
            continue

        # 2. copy makeOperator.py in each directory
        shutil.copy(path_makeOp, path_dir)

        # 3. execute makeOperator script
        os.chdir(path_dir)
        subprocess.run(['python', 'makeOperator.py'])

        # 4. call qdtk_analysis.x and generate dmat2
        subprocess.run([
            path_mlx_AppImage, 'qdtk_analysis.x', '-opr', 'hamilt',
            '-psi', 'psi', '-rst', 'psi',
            '-dmat2', '-dof', str(dof1), '-dofB', str(dof2)
        ])

        # 5. convert dmat2 into npz
        dm2_arr = read_in_dm2(path_dir/dm2_name)

        # 6. create correlation matrix by loading gpop
        _, gpopB, gpopC = get_1bd()
        assert dof1 == 2 and dof2 == 3
        corrBC_arr = dm2_arr - np.outer(gpopB, gpopC)
        corrBC_red= corrBC_arr[100:300, 100:300]
        
        # # Plot correlation function
        # x, y = np.meshgrid(range(n), range(n))
        # im = plt.pcolormesh(x, y, corrBC_arr)
        # plt.colorbar(im)
        # plt.show()

        # 7. store files to destination
        os.makedirs(path_dest, exist_ok=True)
        shutil.copy(path_dir/'parameters.py', path_dest)
        np.savez(path_dest/'correlation_fct_BC.npz', corrBC=corrBC_red)

        # 8. delete makeOperator.py, hamilt and raw dmat2-file
        os.remove(path_dir/dm2_name)
        os.remove(path_dir/'hamilt')

        os.chdir(path_pwd)






#-------------------------------------------------------------------------------
# choose between batches
if args.n is None:
    print('Please provide a value for the argument: -n')
    exit()
n_list = [eval(el) for el in args.n.split(',')]

if 1 in n_list:
    generate_and_copy_files('fbb_scan001_N2_gAB_-1_0')
if 2 in n_list:
    generate_and_copy_files('fbb_scan002_N2_gAB_-0_8')
if 3 in n_list:
    generate_and_copy_files('fbb_scan003_N2_gAB_-0_6')
if 4 in n_list:
    generate_and_copy_files('fbb_scan004_N2_gAB_-0_4')
if 5 in n_list:
    generate_and_copy_files('fbb_scan005_N2_gAB_-0_2')
if 6 in n_list:
    generate_and_copy_files('fbb_scan006_N2_gAB_0_0')
if 7 in n_list:
    generate_and_copy_files('fbb_scan007_N2_gAB_0_2')
if 8 in n_list:
    generate_and_copy_files('fbb_scan008_N2_gAB_0_4')
if 9 in n_list:
    generate_and_copy_files('fbb_scan009_N2_gAB_0_6')
if 10 in n_list:
    generate_and_copy_files('fbb_scan010_N2_gAB_0_8')
if 11 in n_list:
    generate_and_copy_files('fbb_scan011_N2_gAB_1_0')
if 12 in n_list:
    generate_and_copy_files('fbb_scan012_N2_gAB_-1_0')
if 13 in n_list:
    generate_and_copy_files('fbb_scan013_N2_gAB_-0_5')
if 14 in n_list:
    generate_and_copy_files('fbb_scan014_N2_gAB_0_0')
if 15 in n_list:
    generate_and_copy_files('fbb_scan015_N2_gAB_0_5')
if 16 in n_list:
    generate_and_copy_files('fbb_scan016_N2_gAB_1_0')
if 17 in n_list:
    generate_and_copy_files('fbb_scan017_N2_gCC0')
if 18 in n_list:
    generate_and_copy_files('fbb_scan018_N2_weak_gBC')
