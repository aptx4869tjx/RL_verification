# RL_verification

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

## Table of Contents

- [RL_verification](#rl_verification)
    - [Introduction](#introduction)
    - [Install](#install)
    - [Usage](#usage)
        - [Train On Abstract States](#train-on-abstract-states)
        - [CEGAR Process](#cegar-process)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

----

## Introduction

This is the code repository for the paper
"Trainify: A CEGAR-Driven Training and Verification Framework for Verifiable Deep Reinforcement Learning"

## Install

To run this project, execute:

```
conda create -n RL_verification python=3.7

conda activate RL_verification

pip install -r requirements.txt
```

## Usage

### Train On Abstract States

run either:

```
abstract/b1/b1_abs.py
abstract/b2/b2_abs.py
abstract/cartpole/cartpole_abs.py
abstract/pendulum/pendulum_abs.py
abstract/mc/mountaincar_abs.py
abstract/tora/tora_abs.py
```

### CEGAR Process

run either:

```
verify/b1/b1_example.py
verify/b2/b2_example.py
verify/cartpole/cart_example.py
verify/pendulum/pendulum_example.py
verify/mountaincar/mountaincar_example.py
verify/tora/tora_example.py
```