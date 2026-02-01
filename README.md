# Evolutionary Algorithms

This repository contains implementations of various evolutionary algorithms, particularly neuroevolutionary algorithms. Ray is used to provide the distributed runtime. 


## Algorithms
- Implementation of OpenAI's Evolution Strategies as a Scalable Alternative to Reinforcement Learning(https://arxiv.org/abs/1703.03864). 
  The biggest differences from this implementation vs the original one is the following:
  - Using Ray to manage and communicate with workers instead of using the adhoc aws + redis.
  - Using Pytorch instead of Tensorflow
  - Python 3.13
  - Migrate from OpenAI Gym + closed source version of mujoco -> Gymnasium + OSS mujoco(thanks to OpenAI).   
  - Structlog instead of the manually written tabular logger


## TODO
- [ ] Post baseline performance metrics on mujoco environments


## Setup

- Install the pre-commit framework
- Run `uv sync`
