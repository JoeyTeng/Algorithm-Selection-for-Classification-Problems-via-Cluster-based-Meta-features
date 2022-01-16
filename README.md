# Topology and Meta-Learning

## Brief Intro

### About Naming

The initial object is to apply topology into meta-learning as a tool for characterisation of datasets. Yet, topology is a bit hard for me at that moment (and now), and thus I switched to an easier, conventional way of doing meta-learning.

### Abstract (Copied from [published paper](https://github.com/JoeyTeng/Algorithm-Selection-for-Classification-Problems-via-Cluster-based-Meta-features/blob/ac06510052d486237c418dc568b7f31fa10bffcd/docs/IEEE%20Camera%20Ready.pdf))

Meta-features describe the characteristics of the datasets to facilitate algorithm selection. This paper proposes a new set of meta-features based on clustering the instances within datasets. We propose the use of a greedy clustering algorithm, and evaluate the meta-features generated based on the learning curves produced by the Random Forest algorithm. We also compared the utility of the proposed meta-features against pre-existing meta-features described in the literature, and evaluated the applicability of these meta-features over a sample of UCI datasets. Our results show that these meta-features do indeed improve the performance when applied to the algorithm selection task.

## Structure

All scripts are coded for Python3.6/3.7

- ./requirement.txt
  - Dependencies to run scripts in this repository
  - Outdated, please refer to the import list at the top of each script
- ./convex_hull_cluster.py
  - Greedy algorithm with high time complexity
  - Use convex hulls to cluster the instances while ensuring **homogeneity**
  - Dependency
    - Internal: ./meta_features.py
    - External: numpy, scipy
  - Command example: python3 convex_hull_cluster.py path/to/input/[file]
    - Output files
      - [file].clusters.json: saved clustering results. Can be used to prevent re-calculation
      - [file].output.json: saved results with all meta-features' values calculated
      - [file].log
- ./spherical_cluster.py
  - Greedy algorithm with moderate time complexity
  - Similar to convex hull algorithm but uses spherical clusters instead
  - Dependency
    - Internal: ./meta_features.py
    - External: numpy
  - Command: (-h option for help)
    - E.g.: python3 spherical_cluster.py -r path/to/dir/with/inputs --log path/to/log
    - E.g.: python3 spherical_cluster.py -i path/to/input/file
  - Output files:
    - [input].cluaters.json: saved clustering results
      - Cannot be used to restore clustering results
      - (Too lazy to implement)
    - [input].output.json: saved results with all meta-features' values calculated
- ./sperical_brute_force.py
  - Brute force algorithm for optimal clustering result with the constraint of spherical clusters. Extremely high time complexity
    - Enumerate and check every possible combination of splitting the dataset into subsets
  - [Same as "./spherical_cluster.py" above]
- ./meta_features.py
  - Separate module to calculate meta-features with clustering results
  - Do not run it directly
- ./learning_rate.py
- utilities/
  - utilities/extract_meta_features.py:
    - Extract meta-features from .clusters.json (raw clustering results file), not .output.json (suppose to contain meta-features, but depreciated)

[WIP]
