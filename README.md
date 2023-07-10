# Simulation and Retarget of Physically Simulated Characters
This repo is the pre-release of the codebase for SIGGRAPH 2023 paper [Simulation and Retargeting of Complex Multi-Character
Interactions](https://arxiv.org/pdf/2305.20041.pdf). We plan to cleanup and release the codebase later. 

...

## Installation

### Virtual Environment
```
module load anaconda3/5.0.1
pip install virtualenv --user
virtualenv scadive --python=python3
. /scadive/bin/activate
```

### fairmotion
``` 
https://github.com/fairinternal/fairmotion
```

### others
```
pip install pybullet==2.7.3 ray[rllib]==0.8.6 pandas requests
```

### This Project
```
git clone https://github.com/fairinternal/ScaDive.git
cd ScaDive
```

### AMASS (optional)
```
Python version should be 3.7
pip3 install git+https://github.com/nghorbani/configer
pip3 install torch torchvision
pip3 install torchgeometry
pip3 install pyrender
pip3 install git+https://github.com/nghorbani/human_body_prior
pip3 install git+https://github.com/nghorbani/amass
```

### Test
```
python env_humanoid_tracking.py
python rllib_driver.py --mode load --spec data/spec/test_humanoid_imitation.yaml
python rllib_driver.py --mode load --spec data/spec/test_fencing.yaml
```

### Training

This will create 4 workers, where 4 environments will be created because env_per_worker=1 is set. The directory where the project is currently saved should be set to *project_dir* and the directory where checkpoints will be saved should be set to *local_dir*. In the training, it will not create any window to render.

```
python rllib_driver.py --mode train --spec data/spec/test_humanoid_imitation.yaml --num_workers 4 --project_dir ./ --local_dir ./data/learning/test/
```

### Evaluation

For evaluation, we do not need to use more than 1 worker. The evaluation requires to create a window showing simulated agents.

```
python rllib_driver.py --mode load --spec data/spec/test_humanoid_imitation.yaml --num_workers 1 --project_dir ./ --checkpoint SAVED_CHECKPOINT
```

