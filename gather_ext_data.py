import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np

# This module should collect the data from the AFS directory (Uni-Hamburg) 
# and store it in a comprimised form.


# source directory

# path of hamilt:

path_src_dir = Path('/afs/physnet.uni-hamburg.de/project/zoq_t/Driving_BJJ/fbb_mlx_N2/')

path_mlx_AppImage = Path('/home/friethjof/Documents/qdtk-0.5.7.AppImage')

path_makeOp =  path_src_dir/'makeOperator.py'

path_pwd = os.getcwd()

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

#-------------------------------------------------------------------------------
# helper functions

def read_in_dm2(path_dm2f_raw):

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


def generate_and_copy_files(top_dir):
    """Main function: iterate through """


    for path_dir in (path_src_dir/top_dir).iterdir():
        

        # 2. copy makeOperator.py in each directory
        # shutil.copy(path_makeOp, path_dir)

        # 3. execute makeOperator script
        # os.chdir(path_dir)
        # subprocess.run(['python', 'makeOperator.py'])

        # 4. call qdtk_analysis.x and generate dmat2
        # subprocess.run([
        #     path_mlx_AppImage, 'qdtk_analysis.x', '-opr', 'hamilt',
        #     '-psi', 'psi', '-rst', 'psi',
        #     '-dmat2', '-dof', str(dof1), '-dofB', str(dof2)
        # ])

        # 5. convert dmat2 into npz
        # dm2_arr = read_in_dm2(path_dir/dm2_name)

        # 6. create correlation matrix by loading gpop
        # 7. copy npz to destination
        # 8. delete makeOperator.py, hamilt and raw dmat2-file



        os.chdir(path_pwd)

        exit()





#-------------------------------------------------------------------------------
# batch 1
generate_and_copy_files('fbb_scan001_N2_gAB_-1_0')
