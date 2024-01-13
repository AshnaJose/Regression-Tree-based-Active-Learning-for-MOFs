The jupyter notebook **RT-AL_MOFs_example.ipynb** constains a comprehensive example of using Regression Tree-based active learning (RT-AL) on the QMOF dataset. 

The descriptor used in the example is Stoichiometric-120, that was computed using Matminer and is provided for the database in the file stoich120_fingerprints.csv. 

The target property is PBE band gap, provided in the fle labels.csv. 

The structures to compute the descriptor and the PBE band gap values were obtained from the work of Rosen et. al. (DOI: 10.1016/j.matt.2021.02.015)

Informative training sets are selected from the databases using RT-AL (regression_tree.py). The MAE values and predictions computed on the held-out test set, along with the true labels of the test set, and the MOFs selected in the training set for this example are provided in the Results folder.
