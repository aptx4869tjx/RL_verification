## Project Structure
Our project is organized as below:  
+ Trainify
    + abstract
    + abstract_env
    + verify 

- **abstract**: This folder contains the code implementation of abstract training. Each subdirectory corresponds to one task.
- **abstract_env**: This folder contains the environment code of reinforcement learning.
- **verify**: This folder contains the CEGAR Process of six tasks.
  
------------
## Installation

1. unzip
2. `docker load -i trainify.tar`
3. `docker run -it --name trainify-container trainify /bin/bash`
4. Finally if you exit the container and would like to resume using it, you can do as follows: `docker start -i trainify-container`.

------------

## Resource Requirements

All of our experiments are conducted on a workstation running Ubuntu 18.04 with a 32-core AMD Ryzen Threadripper CPU @ 3.7GHz and 128GB RAM.
But the verification program works in a single thread, so it does not require too much CPU resources. We estimate that running the following experiments on a normal laptop will take 2-3 times longer than the given statistics.

------------

## Reproduce `By Trainify` column in Table 2
1. (In container) Enter /home/RL_verification/  
run `cd /home/RL_verification/`
2. run either: 
```
python abstract/b1/b1_Tanh_2_20_p5.py
python abstract/b1/b1_Tanh_2_20_p6.py
python abstract/b1/b1_Tanh_2_100_p5.py
python abstract/b1/b1_Tanh_2_100_p6.py

python abstract/b2/b2_Tanh_2_20_p7.py
python abstract/b2/b2_Tanh_2_20_p8.py
python abstract/b2/b2_Tanh_2_100_p7.py
python abstract/b2/b2_Tanh_2_100_p8.py

python verify/cartpole/CP_Relu_3_64_p4.py

python verify/pendulum/pendulum_Relu_3_128_p3.py

python verify/cartpole/MC_sig_2_16_p1.py
python verify/cartpole/MC_sig_2_16_p2.py
python verify/cartpole/MC_sig_2_200_p1.py
python verify/cartpole/MC_sig_2_200_p2.py

python verify/tora/tora_Tanh_3_100_p9.py
python verify/tora/tora_Tanh_3_200_p9.py
```
For each task, the total time comsumption, the sum of train time, verification time and refine time in all iteration rounds, is shown below. Table 2 only shows the train time and verification time in an iteration.  In addition, the number of iterations also has an impact on the running time. Since the time spent on verification is highly related to the trained network, the total running time may fluctuate greatly, especially in cartpole task.
- b1: 6 minutes
- b2: 1 minutes
- cartpole: 8 hours
- pendulum: 5 minutes
- mountaincar: 90 minutes
- tora: 30 minutes

When these scripts are running, some related information will be printed out:
+ train: denotes the time of training in current iteration.
+ refine: denotes the time of refinement in current iteration.
+ construct kripke structure: denotes the time of constructing kripke structure in current iteration.
+ model checking: denotes the time of model checking in current iteration.

+ number of counterexamples: denotes the number of counterexamples returned from model checking.
+ number of all states in rtree: denotes the number of abstract states in rtree.
+ number of states after refinement: denotes the number of abstract states after refinement.


Notice that, in our implementation, we wrote a fixed network structure and the property to be verified.
In addition, if exception is thrown during running and "no corresponding abstract state" is printed
out, the solution is deleting the .dat and .idx files by executing following commands and rerunning.
```
rm *.dat
rm *.idx
```

------------

## Reproduce `Trainify` column in Table 3
1. Enter /home/RL_verification/  
run `cd /home/RL_verification/`  
2. run either:
```
python abstract/b1/b1_abs.py
python abstract/b2/b2_abs.py
python abstract/cartpole/cartpole_abs.py
python abstract/pendulum/pendulum_abs.py
python abstract/mc/mountaincar_abs.py
python abstract/tora/tora_abs.py
```
For each task, the running time is shown below:
- b1: 50 seconds
- b2: 15 seconds
- cartpole: 300 seconds
- pendulum: 150 seconds
- mountaincar: 300 seconds
- tora: 100 seconds

