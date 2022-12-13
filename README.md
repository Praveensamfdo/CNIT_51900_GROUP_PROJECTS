# CNIT 51900 (Purdue University) Fall 2022

## Reproducibility steps
- Download the repository.
- It is recommended to use a virtual environment to run this project.
    - Create a virtual environment in the base directory using the following command: `conda env create  -f environment.yml --prefix ./envs`.
    - Activate the virtual environment using the following command: `conda activate ./envs` (use `conda deactivate` to exit from the virtual environment).
    - Create a folder called `processed_data` to store the preoprocessed datasets.

# Group project 01: Pun detection and location

- A transformer-based approach has been used for pun detection and location.
- Homographic puns have been considered for this task.

## Transformer-based pun detection
- Training: `python3 pun_detect.py`.
- Inference: `python3 pun_detect_inf.py`

## Transformer-based pun location
- Training: `python3 pun_loc.py`.
- Inference: `python3 pun_loc_inf.py`

## References
- https://arxiv.org/pdf/1909.00175.pdf

# Group project 02: Heterographic Pun Detection and Interpretation

- heterographic puns (where the pun and its latent target are phonologically similar) have been considered for this task.
- The code for this project is located in `het_pun_det_and_int.py`.


