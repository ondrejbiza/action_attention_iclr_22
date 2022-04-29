# Binding Actions to Objects in World Models

Source code for "Binding Actions to Objects in World Models. Ondrej Biza, Robert Platt, Jan-Willem van de Meent, Lawson L. S. Wong and Thomas Kipf. ICLR 2022 workshop on Objects, Structure and Causality."

Paper link: https://arxiv.org/abs/2204.13022

## Setup

We use a virtual environment with Python 3.9:
```
python3.9 -m venv env
source env/bin/activate
```

Install required packages. The PyTorch version we use requires CUDA 10.2.
```
pip install -r requirements.txt
```

## Usage

Currently, this repository contains the code for training, evaluating and visualizing our models on the toy object-based tasks with 2D Shapes and 3D Cubes.
These environments look identical to the ones used in C-SWM (https://arxiv.org/abs/1911.12247), but differ in their action representation.
The actions in the C-SWM environments are associated with objects by their indices, whereas our actions choose objects by their positions.
The outcome of this change is that it becomes more difficult to learn to distinguish individual objects; hence, the need for action attention.

Collect data for 2D Shapes:
```
python collect.py with "env_id=ShapesTrain-v0" "save_path=data/shapes_train" "num_episodes=1000" "seed=1"
python collect.py with "env_id=ShapesEval-v0" "save_path=data/shapes_eval" "num_episodes=10000" "seed=2"
```

Train and evaluate a baseline C-SWM:
```
python run_cswm.py with "seed=1" "model_config.copy_action=True" "model_save_path=data/cswm_shapes.pt"
```

Train and evaluate C-SWM with Hard Attention:
```
python run_cswm.py with "seed=1" "use_hard_attention=True" "model_save_path=data/cswm_ha_shapes.pt"
```

Visualize the learned attention weights:
```
python run_cswm.py with "seed=1" "use_hard_attention=True" "model_load_path=data/cswm_ha_shapes.pt" "viz_names=[Eval]"
```

The same steps with the 3D Cubes dataset:
```
# collect data
python collect.py with "env_id=CubesTrain-v0" "save_path=data/cubes_train" "num_episodes=1000" "seed=1"
python collect.py with "env_id=CubesEval-v0" "save_path=data/cubes_eval" "num_episodes=10000" "seed=2"
# train and evaluate a baseline
python run_cswm.py with "dataset_path=data/cubes_train" "eval_dataset_path=data/cubes_eval" "seed=1" "model_config.copy_action=True" "model_config.encoder=large" "model_save_path=data/cswm_cubes.pt"
# train and evaluate C-SWM with Hard Attention:
python run_cswm.py with "dataset_path=data/cubes_train" "eval_dataset_path=data/cubes_eval" "seed=1" "use_hard_attention=True" "model_config.encoder=large" "epochs=200" "learning_rate=1e-4" "model_save_path=data/cswm_ha_cubes.pt"
# visualize the learned attention weights
python run_cswm.py with "dataset_path=data/cubes_train" "eval_dataset_path=data/cubes_eval" "seed=1" "use_hard_attention=True" "model_config.encoder=large" "model_load_path=data/cswm_ha_cubes.pt"  "viz_names=[Eval]"
```

The data reported in our tables was collected by training all models with random seeds from 0 to 19.


Our paper also includes an experiment with Atari games and simulated robotic manipulation. For the Atari experiment, we use the dataset from: https://github.com/ondrejba/negative-sampling-icml-21. The robotic manipulation experiment is connect to our paper https://arxiv.org/abs/2202.05333. We are working on open-sourcing the code for both of these.

# Citation

```
@article{biza22binding,
  title={Binding Actions to Objects in World Models}, 
  author={Ondrej Biza, Robert Platt, Jan-Willem van de Meent, Lawson L. S. Wong and Thomas Kipf}, 
  journal={ICLR 2022 workshop on Objects, Structure and Causality}, 
  year={2022} 
}
```
