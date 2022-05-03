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
2. `cd Trainify`

###  Build the docker image

3. `docker build -t trainify .`

### Create the docker container

4. `docker run -it --name trainify-container trainify /bin/bash`
5. Finally if you exit the container and would like to resume using it, you can do as follows: `docker start -i trainify`.

------------

## Resource Requirements

All of our experiments are conducted on a workstation running Ubuntu 18.04 with a 32-core AMD Ryzen Threadripper CPU @ 3.7GHz and 128GB RAM.
But the verification program works in a single thread, so it does not require too much CPU resources. We estimate that running the following experiments on a normal laptop will take 2-3 times longer than the given statistics.

------------

## Reproduce `By Trainify` column in Table 2
1. Enter /home/RL_verification/  
run `cd /home/RL_verification/`
2. run either: 
```
python verify/b1/b1_example.py
python verify/b2/b2_example.py
python verify/cartpole/cart_example.py
python verify/pendulum/pendulum_example.py
python verify/mountaincar/mountaincar_example.py
python verify/tora/tora_example.py
```
For each task, the total time comsumption, the sum of train time, verification time and refine time in all iteration rounds, is shown below. Table 2 only shows the train time and verification time in an iteration.  In addition, the number of iterations also has an impact on the running time. Since the time spent on verification is highly related to the trained network, the total running time may fluctuate greatly, especially in cartpole task.
- b1: 6 minutes
- b2: 3 minutes
- cartpole: 8 hours
- pendulum: 5 minutes
- mountaincar: 50 minutes
- tora: 30 minutes


Notice that, in our implementation, we wrote a fixed network structure and the property to be verified.   

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


