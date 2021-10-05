# Online Training and Testing

Online training and testing requires two components: this repository for handling tensorflow modules, and the PCCProject/PCC-Uspace.git repository for
sending data over real networks and making observations.

This repository contains a training environment in /src/gym/online/shim\_env.py that connects to a pccclient program from the PCC-Uspace repository. The shim
environment acts as an intermediary that provides a gym-based API while receiving observations and taking actions remotely on the pccclient.

You can also test models trained in simulation (or online) as a tf.saved\_model.

For both training and testing, you'll need the PCCProject/PCC-Uspace.git repo with the `deep-learning` branch checked out.

The [deep-learning README](https://github.com/PCCproject/PCC-Uspace/blob/deep-learning/Deep_Learning_Readme.md) shows how to run the
pccclient program with either online training or real world testing.
