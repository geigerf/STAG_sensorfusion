# STAG_sensorfusion
## Introduction

This is a minimal implementation of the adapted neural network (based on _Subramanian Sundaram, Petr Kellnhofer, Yunzhu Li, Jun-Yan Zhu, Antonio Torralba and Wojciech Matusik. "Learning the signatures of the human grasp using a scalable tactile glove". Nature, 2019._) that was found to have the best size-accuracy trade-off.
It is minimal in the sense that it tries to reduce the number of tunable variables to a minimum.
The goal of this neural network is to classify 17 different objects from a data fusion between tactile data and synchronized IMU data.


## System requirements

This lists the installed python packages and versions.
However, it is not absolutely necessary to install it with exactly the same versions.
Consult this list if there are some broken dependencies in your environment.

Required packages:
- Python            3.8.1
- numpy             1.18.1
- pytorch           1.4.0 CUDA version
- imbalanced-learn  0.6.2
- scikit-learn      0.22.1
- scipy             1.4.1

Imported standard packages:
- argparse
- collections
- datetime
- os
- random
- re
- shutil
- sys
- time


## Additional information

Running *python STAG_sensorfusion --help* in a python environment with the required packages will give a list of possible arguments.


## Contact

Please email any questions or comments to [geigerf@student.ethz.ch](geigerf@student.ethz.ch).
