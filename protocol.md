# Cassification of Correlation Patterns


###Description
Program to evaluate the correlation patterns of a three-component ultracold system consisting of two bosonic and one fermionic species (A, B, C) interacting with gAB, gAC, gBC, gBB, gCC.
In advance of this analysis, we have generated the data using the ML-MCTDHX method and calculated each scenario as a different job on a cluster.

The correlation function which should be analyzed correspond to the two-body correlation between two particles resolved on a grid.

- Size of the images: 400 x 400  (we have employed 400 grid points)
- Various interspecies interaction strengths are varied: gAB, gAC, gBC, gBB, gCC.

### Aim:

- We are interested in training a model that can reproduce the expected correlation pattern when passing interaction parameters which have not been probed.
- Find features of correlation patterns and check whether they form clusters.


### Procedure (chronologically):

- Generate data (this has been done already), i.e., for each parameter configuration we have access to a psi-file which represents the many-body wave function and contains all information of the many-body problem.

- Calculate from the psi-file the observables needed for the correlation function.

- To reduce required storage: reduce image sizes and trim the edges, where the correlation fucntion is close to zero. **New image size: 200x200.**

- Store the correlation functions as a npz file in a data-folder.



