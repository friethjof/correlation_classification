# Cassification of Correlation Patterns


## 1. In Short

The aim is to group a large data set (~18000 samples) of matrices into different clusters. Each matrix shows the spatially resolved to-body correlation matrix and is of the size 400x400. Depending on the input parameters to generate the correlation matrix, the respective pattern of correlation matrix can change. We want to group the matrices in clusters with respect to the underlying correlation pattern. Furthermore, we want to find the connection between clusters and the input parameters.


## 2. File-Content

- gather_ext_data.py
    - call external analysis scripts
    - read output-data (text files)
    - calculate correlation matrix
    - copy parameter file and store correlation matrices as npz in a separate folder structure

- create_parameter_DataFrame.ipynb
    - read in parameter-file and create corresponding DataFrame (df_parameter.p)

- create_DataFrame_corr_mat.ipynb
    - reduce dimensionality of t

- find_clusters_red_sample_v1_gAB_gAC.py

- find_clusters_red_sample_v2_gAB_gAC_gBC.py

- find_clusters_red_sample_v3_gAB_gAC_gBC.py


## 3. Install

Package requirements are stored in *environment.yml* , which can be installed by

```
conda env create -f environment.yml
```


## 4. Scenario

In this project, we consider three interacting species of atoms, i.e., species $A$, $B$, $C$.

- All species are represented by two particles, $N_A=N_B=N_C=2$.
- All atoms have the same mass and are traped in a one-dimensional harmonic oscillator potential.
- The atoms interact with each other via a contact interaction potential.
    - The tunable parameters are denoted by the interaction strengths: $g_{BB},g_{CC},g_{AB},g_{AC},g_{BC}$.
    - For example, $g_{BB}$ denotes the interaction between the two particles of the $B$-species, while $g_{BC}$ is the interaction strength between particles of species $B$ and $C$.
    - If the interaction strength is negative (e.g., $g_{BC}<0$) the particles attract each other, if it is positive (e.g., $g_{BC}>0$) the particles repell each other. The particles are non-interacting if the strength vanishes, e.g., when $g_{BC}=0$.

For each interaction strength combination we obtain the numerical exact ground state of the 6-particle system.


## 5. Observable (Data Samples)

We calculate the *two-body correlation matrix* as

$$ \mathcal{C}_{BC}^{(2)}(x_B, x_C) = \rho_{BC}^{(2)}(x_B, x_C) - \rho_{B}^{(1)}(x_B)\rho_{C}^{(1)}(x_C), $$

where $\rho_{BC}^{(2)}(x_B, x_C)$ is the spatially resolved two-body density and gives the probability of finding/measuring at the same time one particle of species $B$ and one particle of species $C$ at the positions $x_B$ and $x_C$. This *conditional* probability is the compared to the *unconditional* one constructed by the product of the one-body densities $\rho_{B}^{(1)}(x_B)$ and $\rho_{C}^{(1)}(x_C)$, where, e.g., $\rho_{B}^{(1)}(x_B)$ gives the probability of finding one particle of species $B$ at position $x_B$ regardless of the locations of the other particles.

In general we define:

$$ \mathcal{C}_{BC}^{(2)}(x_B, x_C) = 
\begin{cases}
    < 0  & \text{particles $B$ and $C$ are anti-correlated at positions } x_B \text{ and } x_C. \\
    > 0  & \text{particles $B$ and $C$ are correlated at positions } x_B \text{ and } x_C. \\
    = 0  & \text{particles $B$ and $C$ are not correlated at positions } x_B \text{ and } x_C. \\
\end{cases} $$

We refer to the particles as **correlated** if the pattern of the two-body correlation matrix exhibits positive values along its diagonal and negative values at its off-diagonal. Meaning, we speak of correlated particles if it is more likely to find these particles at the same position ($\hat{=}$ diagonal area of the correlation matrix).
On the other hand, we refer to the particles as **anti-correlated** when they avoid each other, i.e., when they avoid to occupy the same spatial location. This is expressed in the correlation matrix as negative values at the diagonal and positive values at the off-diagonals.

These are the most general cases which highly depend on the applied interaction parameters. Besides these two basic types of correlation patterns, there might be others which appear at different interaction parameter configurations.


## 6. Expectation of Cluster Formation

- For a decoupled species $A$, i.e., if $g_{AB}=g_{AC}=0$, we find a *correlated* pattern of $\mathcal{C}_{BC}^{(2)}(x_B, x_C)$ when $g_{BC}<0$, meaning the particles of species $B$ and $C$ attract each other, and an *anti-correlated* pattern, when the particles repel each other, i.e., $g_{BC}>0$.

- If $g_{BC}=0$, i.e., particles belonging to species $B$ and $C$ are non-interacting, one might expect that the correlation function $\mathcal{C}_{BC}^{(2)}(x_B, x_C)$ might be zero everywhere. However, this is not the case since the third species $A$ can mediate an *induced interaction* between species $A$ and $B$, but only if both species $B$ and $C$ are interacting with species $A$ ($g_{AB},g_{AC}\neq 0$). Specifically, we expect an induced attraction between $B$ and $C$, when $g_{AB} \cdot g_{AC} > 0$, and an induced repulsion for $g_{AB} \cdot g_{AC} < 0$. The leads to the expected correlation patterns:
    - $g_{AB} \cdot g_{AC} > 0$: correlated pattern
    - $g_{AB} \cdot g_{AC} < 0$: anti-correlated pattern
    
- Now, if we consider non-zero interaction parameters $g_{AB},g_{AC},g_{BC}\neq0$, the *induced* interaction related to $g_{AB} \cdot g_{AC}$ has to be compared to the *actual* interspecies interaction strentgth $g_{BC}$. Depending which type of interaction is dominant we will either observe a correlated or anti-correlated pattern emerging in the correlation matrix $\mathcal{C}_{BC}^{(2)}(x_B, x_C)$:

    - $\alpha \cdot g_{AB} \cdot g_{AC} + \beta \cdot g_{BC} < 0$: correlated pattern
    - $\alpha \cdot g_{AB} \cdot g_{AC} + \beta \cdot g_{BC} > 0$: anti-correlated pattern

    where $\alpha$ and $\beta$ are unkown.


## 7. Aim

- We are interested in training a model which groups the correlation patterns into clusters and finds clusters which align with the expected behavior described above.
- Since we vary more than the described parameters, we aim at identifying new correlation patterns.


## 8. Data Gathering

In advance of this analysis, we have generated the data using the ML-MCTDHX method (see Literature) and have calculated each scenario as a different job on a computing cluster.

Each parameter strength, $g_{AB}, g_{AC}, g_{BC}, g_{BB}, g_{CC}$ is can obtain one of the following values $[-1, -0.8,  -0.6, -0.5, -0.4, -0.2, -0.05, 0, 0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 1]$. However, we do not consider every possible combination. The total number of data samples is $N_{\rm{samples}}=18014$.

Each data sample corresponds to a correlation matrix. The correlation functions are resolved on a spatial grid with $400$ grid points. Thus, the size of each correlation function is $(400 \times 400)$.

Each element of the correlation matrix represents a feature so that the DataFrame of the raw data has the size of $(18014 \times 160000)$. This is way too large - at least for my laptop!

However, we can further reduce the dimensionality by exploiting some symmetry propoerties and and applying a principle component analysis (PCA), see below!

Eventually we end up with a number of features of the order of $\sim 50$ so that the size of the evaluated DataFrame is something like $(18014 \times 50)$.


### 8.1 Starting point

For each parameter configuration we have access to a psi-file which represents the many-body wave function and contains all information of the many-body problem. All psi-files are stored externally.


### 8.2 Create and store correlation functions

Script: **gather_ext_data.py**

- Calculate from the psi-file the two-body correlation between species B and C. By employing an external analysis-tool from the ML-MCTDHX method.

- The correlation functions describe the correlation between two particles resolved on a spatial grid with 400 grid points. Thus, the size of each correlation function is $(400 \times 400)$.

- To reduce the required storage: correlation functions/matrices are trimmed. The values at the edges are close to zero so we can cut them. **New size of correlation matrix:** $(200 \times 200)$

- Store each correlation functions as a npz-file in a data-folder together with the parameters.py-file. Store as batches, i.e., folders with up to 1000 subfolders.


### 8.3 Create a parameter-DataFrame of all samples

Script: **create_DataFrame_parameter.ipynb**

- Iterate through all folders.

- Read in parameters.py file and extract values for interaction parameters $g_{BB}, g_{CC}, g_{AB}, g_{AC}, g_{BC}$.

- Collect the interaction parameters as well as the paths to the external folders in a DataFrame.

- Store DataFrame as pickle in **df_parameter.p**.


### 8.4 Reduce Dimensionality of Data

Script: **create_DataFrame_corr_mat.ipynb**

- Iterate through data using paths from **df_parameter.p**.

- Load correlation functions and create a matrix, where each row corresponds to a flattened correlation-matrix. The matrices are **symmetric** along the diagonal due to a symmetric setup. Therefore, we focus only on the upper triangular of the matrices, which we flatten like

<table>
  <tr>
    <td>
      <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; text-align: center;">
        <tr><td>a</td><td>b</td><td>c</td></tr>
        <tr><td>b</td><td>d</td><td>e</td></tr>
        <tr><td>c</td><td>e</td><td>f</td></tr>
      </table>
    </td>
    <td style="font-size: 24px; padding-left: 20px; vertical-align: middle;">→</td>
    <td>
      <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; text-align: center;">
        <tr><td>a</td><td>b</td><td>c</td></tr>
        <tr><td></td><td>d</td><td>e</td></tr>
        <tr><td></td><td></td><td>f</td></tr>
      </table>
    </td>
    <td style="font-size: 24px; padding-left: 20px; vertical-align: middle;">→</td>
    <td style="font-family: monospace; padding-left: 20px; vertical-align: middle;">[a, b, c, d, e, f]</td>
  </tr>
</table>

- Store matrices with up to 1000 flattened correlation functions as DataFrames in order to reduce memory.

- Reload and merge to a large DataFrame: size $(18014 \times 20100 )$.

- Make principle components analysis. Check how many components are needed: Result:

<table border="1" cellpadding="5" cellspacing="0">
  <tr>
    <th>n_comp</th>
    <th>explained variance</th>
  </tr>
  <tr>
    <td>2</td>
    <td>0.8722</td>
  </tr>
  <tr>
    <td>10</td>
    <td>0.9985</td>
  </tr>
  <tr>
    <td>50</td>
    <td>0.9999991</td>
  </tr>
</table>

- By comparing the original correlation matrix with the reduced ones, we find that $n_{\rm{comp}}=50$ yields adequate results.


- Combine df_parameter.p and the principle components and create the final DataFrame **df_main_std_pca_50.p** with size **18014x57**.


## 9. Find Clusters in a Reduced Sample Set (1st attempt)

Here, we try to divide a minimal sample into two clusters and for this task compare k-means with DBSCAN.

**K-Means**

Define the number of cluster centers $N$. Place the centers randomly in parameter space. Then assign each data point to the closest cluster. Determine for each cluster the new center and repeat this procedure. It stops when the within cluster variance is minimized.

**How does DBSCAN work?**

DBSCAN groups together points that lie in dense regions by expanding clusters from points that have at least a minimum number of neighbors within a given distance (eps). Points that don’t meet the density criterion are marked as noise or outliers and get the value -1.


### 9.1 Find Clusters in Minimal Sample Set

Script: **find_clusters_red_sample_v1_gAB_gAC.py**

We consider $g_{BB}=g_{CC}=g_{BC}=0$ and require $g_{AB},g_{AC}\neq 0$ so that we expect only two correlation patterns. Namely, a correlated correlation pattern when $g_{AB} \cdot g_{AC} > 0$ and an anti-correlated correlation pattern if $g_{AB} \cdot g_{AC} < 0$ (see discussion above).


**K-Means**

The clustering algorithm works partially well. However, two falsely classified samples can be identified in the overview plot. These are two samples assigned to cluster 1 (orange dots) lying on the off-diagoanl of the overview plot with $g_{AB}$ ($g_{AC}$) on the x-axis (y-axis).


**DBSCAN**

No hyper-parameter setting was able to uniquly classify all data points. The best result is obtained with min_samples=10 and eps=2.1. However, in this case there were many outliers.


### 9.2 Extension of Minimal Sample Set´

Script: **rfind_clusters_ed_sample_v2_gAB_gAC_gBC.py**

Here, we set $g_{BB}=g_{CC}=0$ and vary the parameters $g_{AB}, g_{AC}, g_{BC}$.

Remind, that we expect an interplay between interspecies interaction $g_{BC}$ and induced/mediated interaction which is related to $g_{AB}\cdot g_{AC}$.

However, using k-means and DBSCAN no hyper-parameter setting was able to pass the sanity check, i.e., to correctly classify the patterns appearing in the $g_{AB}-g_{AC}$ plane for $g_{BC}=g_{BB}=g_{CC}=0$.

This might be due the fact that the clustering algorithms cluster the correlation matrices according to absolute values and not according to the pattern of the matrices.


## 10. Find Clusters in a Reduced Sample Set (2nd attempt)


### 10.1 Data Preparation

Script: **create_DataFrame_corr_mat_v2.ipynb**

In previous attempts the clustering algorithms where applied on DataFrames consisting of correlation matrices
which are scaled with **StandardScaler** and compressed with **PCA**.

Here, we **Normalize** each correlation matrix so that we only sort the correlation matrices according to their patterns. Then we apply **StandardScaler** and **PCA**.


### 10.2 Find Clusters

Script: **find_clusters_red_sample_v3_gAB_gAC_gBC.ipynb**

**K-Means**

We get the most *reasonable* results (the results which we would expect) for K-Means when considering 4 or 5 clusters. Now we describe the clusters in the case of 5 clusters:

- Appearance of a correlation patttern when $g_{BC}<-0.05$ as well as when $g_{BC}<0.05$ and $g_{AB}\cdot g_{AC}>0$. So there is a small margin where the induced attraction overcomes a repulsive $g_{BC}$.
- Two vanishing patterns when $g_{BC}$, and either $g_{AB}=0$ and/or $g_{AC}=0$.
- Anti-correlation when $g_{BC}>0.05$ as well as when $g_{BC}>-0.05$ and $g_{AB}\cdot g_{AC}<0$. Also here is a small margin where the induced repulsion overcomes a attractive $g_{BC}$.
- For 5 clusters: A separate cluster appears when $g_{BC}>0.5$ and $g_{AB}+g_{AC}>-0.5$.


**DBSCAN**

DBSCAN yields similar results as k-means. However, there are many outliers when $g_{BC}$ is large and the others $g_{AB}, g_{AC}$ are smaller.


**Conclusion**

All in all, k-means yields *visually* better results and seems more reliable.



## 11. Find Clusters in the Full Sample Set

Script: **find_clusters_full_sample.ipynb**




## Literature

**Induced Interactions in ultracold 1D systems**

[1] J. Chen, J. M. Schurer and P. Schmelcher, *"Entanglement Induced Interactions in Binary Mixtures"*, Phys. Rev. Lett. **121**, 043401 (2018), DOI: https://doi.org/10.1103/PhysRevLett.121.043401 .

[2] F. Theel, S. I. Mistakidis and P. Schmelcher, *"Crossover from attractive to repulsive induced interactions and bound states of two distinguishable Bose polarons"*, SciPost Phys. **16**, 023 (2024), DOI: https://doi.org/10.21468/SciPostPhys.16.1.023 .


**Numerical Method (ML-MCTDHX)**

[3] L. Cao, V. Bolsinger, S. I. Mistakidis, G. M. Koutentakis, S. Krönke, J. M. Schurer, and P. Schmelcher, *"A unified ab initio approach to the correlated quantum dynamics of ultracold fermionic and bosonic mixtures"*, J. Chem. Phys. **147**, 044106 (2017), DOI: https://doi.org/10.1063/1.4993512 .


[4] L. Cao, S. Krönke, O. Vendrell, and P. Schmelcher, *The multi-layer multi-configuration time-dependent Hartree method for bosons: Theory, implementation, and applications*, J. Chem. Phys. **139**, 134103 (2013), DOI: https://doi.org/10.1063/1.4821350 .

[5] S. Krönke, L. Cao, O. Vendrell, and P. Schmelcher, *"Non-equilibrium quantum dynamics of ultra-cold atomic mixtures: The multi-layer multi-configuration time-dependent Hartree method for bosons"*, New J. Phys. **15**, 063018 (2013), DOI: https://doi.org/10.1088/1367-2630/15/6/063018 .

... and citations therein.
