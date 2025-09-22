# Cassification of Correlation Patterns

###In short

Group a large set (~18000 samples) of correlation matrices into different clusters. Each correlation matrix of size 400x400 correspond to a set of parameters. Find connection between clusters of correlation matrices and their parameters.


###Content

**protocol.md**

Gives more details on project and advancements.

gather_ext_data.py

- call external analysis scripts
- read output-data (text files)
- calculate correlation matrix
- copy parameter file and store correlation matrices as npz in a separate folder structure

create_parameter_DataFrame.ipynb

    - read in parameter-file and create corresponding DataFrame
