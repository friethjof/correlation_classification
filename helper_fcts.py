import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import NearestNeighbors


def get_index(df_par, gBB, gCC, gAB, gAC, gBC):
    """Get index corresponding to a specific choice for the interaction parameters gXY
    
    Args:
        df_par (DataFrame): DataFrame of the parameter space
        gBB (float): interaction strength between B-B
        gCC (float): interaction strength between C-C
        gAB (float): interaction strength between A-B
        gAC (float): interaction strength between A-C
        gBC (float): interaction strength between B-C
    
    Returns:
        index (int) corresponding to input interaction parameters
    """
    
    mask = (
        (df_par.loc[:, 'gBB'] == gBB) &
        (df_par.loc[:, 'gCC'] == gCC) &
        (df_par.loc[:, 'gAB'] == gAB) &
        (df_par.loc[:, 'gAC'] == gAC) &
        (df_par.loc[:, 'gBC'] == gBC)
    )
    
    if all(~mask):
        print('No match found, return None')
    elif mask.sum() == 1:
        return np.where(mask)[0][0]
    else:
        print('More than one match found, return the first')
        return np.where(mask)[0][0]


def get_interaction_str(df_par, index):
    """
    Create interaction string from run defined by index ind

    Args:
        df_par (DataFrame): inpur data frame with interaction parameters
        index (int): index of run
    
    Returns:
        str with interaction values
    """


    # title_str = ''
    # title_str += r'$g_{BB}=' + str(df_par.loc[index, 'gBB']) + '$ '
    # title_str += r'$g_{CC}=' + str(df_par.loc[index, 'gCC']) + '$ '
    # title_str += r'$g_{AB}=' + str(df_par.loc[index, 'gAB']) + '$ '
    # title_str += r'$g_{AC}=' + str(df_par.loc[index, 'gAC']) + '$ '
    # title_str += r'$g_{BC}=' + str(df_par.loc[index, 'gBC']) + '$ '
    
    title_str = ''
    title_str += f"gBB={df_par.loc[index, 'gBB']}, "
    title_str += f"gCC={df_par.loc[index, 'gCC']}, "
    title_str += f"gAB={df_par.loc[index, 'gAB']}, "
    title_str += f"gAC={df_par.loc[index, 'gAC']}, "
    title_str += f"gBC={df_par.loc[index, 'gBC']}"
    
    return title_str

from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_5_random_cluster_examples(df_features, cluster_col, cluster_id, seed=0):
    """
    Plot 5 correlation plots which are part of the n-th cluster 
    
    Args:
        df_features (DataFrame): data frame which includes all information 
        cluster_col (str): name of cluster colum
        cluster_id (int): cluster index
        seed (int): seed to make the random-pick deterministic, default=0
    
    Returns:
        fig instance
    """
    
    index_list = df_features.index[df_features.loc[:, cluster_col] == cluster_id]
        
    np.random.seed(seed)
    # choose 5 picks from index_list
    random_picks = np.random.choice(index_list, size=5, replace=False)
    
    fig, axes = plt.subplots(1, 5, figsize=(12, 2), sharex=True, sharey=True, constrained_layout=True)
    plt.rc('text', usetex=True)
    
    x_grid = np.linspace(-100, 100, 200)
    x, y = np.meshgrid(x_grid, x_grid)
    
    for i, ax in enumerate(axes):

        corr_mat_ori = np.load(
            df_features.loc[random_picks[i], "path"]
            + '/correlation_fct_BC.npz'
        )['corrBC']
        
        ax.set_aspect(1)

        vlim = np.max(np.abs(corr_mat_ori))
        im = ax.pcolormesh(x, y, corr_mat_ori, shading='auto', cmap='bwr',
                           vmin=-vlim, vmax=vlim)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad=0.05)
        fig.colorbar(im, cax=cax)
        
        int_str = get_interaction_str(
            df_par=df_features,
            index=random_picks[i]
        )
        print(f'sample {i}', int_str, f'index: {random_picks[i]}')


def plot_kNN_distance(features, k, elbowpoint):
    """Plot for each feature the distance to the k-th neighbor.
    
    Args:
        features (DataFrame): feature data frame
        k (int): k-th neighbor
        elbowpoint (int): number of sample for which the eps should be determined
    
    Returns:
        kNN distance for elbowpoint (float)
    """
    

    # Fit nearest neighbors
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(features)
    distances, indices = neigh.kneighbors(features)

    # Take the k-th nearest neighbor distance for each point and sort it
    k_distances = np.sort(distances[:, k-1])
    
    # determine kNN-distance of elbowpoint
    eps = k_distances[elbowpoint]
    
    
    # Plot
    fig, axes = plt.subplots(figsize=(8,4))
    axes.plot(k_distances)
   

    axes.axvline(elbowpoint, ls='--', color='gray')
    axes.axhline(eps, ls='--', color='gray')


    axes.set_xlabel('Sample')
    axes.set_ylabel(f'{k}th Nearest Neighbor Distance')
    axes.set_title(f'K-distance plot with K={k}, eps={eps:.3f}')
    
    plt.show()
    
    return eps



def flat_triu_to_full_matrix(flat_triu, N):
    """Convert a flatted triangular matrix to a full matrix of size N x N
    
    Assume that the matrix is symmetric, then fill the lower triangular part
    with the values of the upper part.

    Args:
        flat_triu (np.array): triangular matrix flatted to an array
        N (int): number of columns/rows of the full matrix
    
    Returns:
        Symmetric (N x N)-np.array
    """

    # convert flatted array to matrix:
    mat_half = np.zeros((N, N))
    mat_half[np.triu_indices(N)] = flat_triu

    # add reflection and subtract diagonal
    mat = np.rot90(np.fliplr(mat_half)) + mat_half - np.diag(np.diag(mat_half))

    return mat


def plot_comparison_betw_2_mats(
        mat1,
        mat2,
        x_grid,
        ax1_title='', 
        ax2_title='', 
        ax3_title='',
        main_title='',
        separate_cbars=False,
        fig_name=None
    ):
    """Plot two matrices next to each other as well as their absolute difference.
    
    Args:
        mat1 (np.array): matrix 1 
        mat2 (np.array): matrix 2
        ax1_title (str): title of axis 1
        ax2_title (str): title of axis 2
        ax3_title (str): title of axis 3
        main_title (str): main title above ax-titles
        separate_cbars (bool): whether to allow a cbar for ax1 and ax2
        fig_name (Path): path of figure to store, if None then show only
        
    Returns:
        None    
    """

    #===========================================================================
    # Define grid structure
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8, 3), dpi=300)

    fig.subplots_adjust(left=0.15, right=0.89, top=0.92, bottom=0.12)



    if separate_cbars:
        canv = gridspec.GridSpec(1, 2, width_ratios=[2.1, 1], wspace=0.25)
        canv_left = gridspec.GridSpecFromSubplotSpec(
            1, 2, canv[0, 0], width_ratios=[1, 1], wspace=0.25)
    else:
        canv = gridspec.GridSpec(1, 2, width_ratios=[2.1, 1], wspace=0.4)
        canv_left = gridspec.GridSpecFromSubplotSpec(
            1, 2, canv[0, 0], width_ratios=[1, 1], wspace=0.1)

    ax1 = plt.subplot(canv_left[0, 0], aspect=1)
    ax2 = plt.subplot(canv_left[0, 1], aspect=1)    
    ax3 = plt.subplot(canv[0, 1], aspect=1)

    
    #===========================================================================
    # make plots
    
    
    x, y = np.meshgrid(x_grid, x_grid)
    ax1.set_title(ax1_title)
    ax2.set_title(ax2_title)
    ax3.set_title(ax3_title)
    
    vmin, vmax = mat1.min(), mat1.max()
    im = ax1.pcolormesh(x, y, mat1, shading='auto')
    im.set_rasterized(True)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="9%", pad=0.05)
    if separate_cbars:
        vmin, vmax = mat2.min(), mat2.max()
        fig.colorbar(im, cax=cax, extend='both')
    else:
        cax.set_axis_off()
    
    im = ax2.pcolormesh(x, y, mat2, shading='auto', vmin=vmin, vmax=vmax)
    im.set_rasterized(True)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="9%", pad=0.05)
    fig.colorbar(im, cax=cax, extend='both')
    
    im = ax3.pcolormesh(x, y, np.abs(mat1 - mat2), shading='auto', cmap='bwr')
    im.set_rasterized(True)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="9%", pad=0.05)
    fig.colorbar(im, cax=cax)
    
    ax1.set_xlabel(r'$x_1$')
    ax2.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    ax3.set_xlabel(r'$x_1$')
    ax2.set_yticklabels([])
    
    # make main title    
    ax2.annotate(
        main_title,
        xy=(0.6, 1.25),
        xycoords='axes fraction',
        ha='center',
    )
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([-100, 0, 100])
        ax.set_yticks([-100, 0, 100])
        

    if fig_name is None:
        plt.show()
 
    else:
        
        plt.savefig(fig_name)
        plt.close()

        if fig_name[-3:] == 'png':
            subprocess.check_output(["convert", fig_name, "-trim", fig_name])
        elif fig_name[-3:] == 'pdf':
            subprocess.check_output(["pdfcrop", fig_name, fig_name])
    

    return None


def plot_backscaled_comparison(df_par, corr_backscaled, index, pca_n, bool_save=False):
    """Reconstruct correlation matrix from the PCA and compare to the original one.
        
    Args:
        df_par (DataFrame): DataFrame containing the paths to of original correlation matrices
        corr_backscaled (numpy.array): Get matrix where rows are flattened correlation
                                       matrices with only upper trinagualr entries being stored.
        index (int): index of the individual correlation matrix
        pca_n (int): number of PCA components
        bool_save (boolean): determines whether to save the figure, default=False
        
    Return:
        None
    
    """

    # load original matrix
    corr_mat_ori = np.load(df_par.loc[index, "path"] + '/correlation_fct_BC.npz')['corrBC']
    n_grid = corr_mat_ori.shape[0]

    # reconstructed matrix, which has been reduced by PCA
    corr_flat_pca = corr_backscaled[index, :]
    corr_mat_pca = flat_triu_to_full_matrix(flat_triu=corr_flat_pca, N=n_grid)


    # define input paramters for plotting function
    title_str = ''
    title_str += r'$g_{BB}=' + str(df_par.loc[index, 'gBB']) + '$, '
    title_str += r'$g_{CC}=' + str(df_par.loc[index, 'gCC']) + '$, '
    title_str += r'$g_{AB}=' + str(df_par.loc[index, 'gAB']) + '$, '
    title_str += r'$g_{AC}=' + str(df_par.loc[index, 'gAC']) + '$, '
    title_str += r'$g_{BC}=' + str(df_par.loc[index, 'gBC']) + '$'

    if bool_save:
        fig_name = f'figures/corr_ind{index}_pca_n_{pca_n}.png'
    else:
        fig_name = None

    # plot
    plot_comparison_betw_2_mats(
        mat1=corr_mat_ori,
        mat2=corr_mat_pca,
        x_grid = np.linspace(-100, 100, n_grid),
        ax1_title=r'$\mathcal{C}_{\mathrm{ori}}$',
        ax2_title=r'$\mathcal{C}_{\mathrm{pca}}$'+ f', n={pca_n}',
        ax3_title=r'$|\mathcal{C}_{\mathrm{pca}} - \mathcal{C}_{\mathrm{ori}}|$',
        main_title=title_str,
        fig_name=fig_name
    )

    return None
