# Stable stone packing in 2D

This is a simplified implementation of [the StablePacking-2D algorithm](https://github.com/eesd-epfl/StablePacking-2D). The methodology is published in the paper entitled [An image convolution-based method for the irregular stone packing problem in masonry wall construction](https://www.sciencedirect.com/science/article/pii/S0377221724000730?via%3Dihub).

## Installation

- create virtual environment by `python -m venv env` and `source env/bin/activate`

- run `pip install -e .` in this directory for an editable install

## Usage

### Input/output Format

The inputs are binary stone images where stone pixels are 1, background pixels are 0.
The outputs are:

- Binary wall image where occupied pixels are 1 and non occupied pixels are 0.
- Segmented wall image where pixels of different stones are of different value.
- Transformation of each stone. 💡`test_example5.py` gives a way to correctly output the transformation (x translation, y translation, rotation angle) into a txt file with the corresponding stone id (extracted from stone file name).
- Rigid body model of the wall for kinematics analysis.

### Code structure

The main functionality of the stacking algorithm is implemented in the script `place_stone_2d.py`. The other scripts contain auxiliary functions such generating rigid body model for stability verification (`evaluate_kine.py`), solving kinematic analysis (`kine_2d.py`) and rotating stones (`rotate_stone.py`).

There are two main function to be called to solve a stone stacking problem. One is `generate_one_wall_best_pose_given_sequence`, where stones are given as input to build a wall from zero, another one is `generate_one_wall_best_pose_given_sequence_given_wall` where stones are given to continue building an existing wall.

The example in `test_example5.py` gives an example to explore a stone set iteratively until all stones are placed, by calling the two main functions.
