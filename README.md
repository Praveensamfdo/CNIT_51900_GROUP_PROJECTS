## Reproducibility steps
- Download the repository.
- It is recommended to use a virtual environment to run this project.
    - Create a virtual environment in the base directory using the following command: `conda env create  -f environment.yml --prefix ./envs`.
    - Activate the virtual environment using the following command: `conda activate ./envs` (use `conda deactivate` to exit from the virtual environment).

- Create `model` and `embeddings` folders in the base directory.

## Transformer-based pun detection
- Training: `python3 pun_detect.py`.
- Inference: `python3 pun_detect_inf.py`

## Transformer-based pun location
- Training: `python3 pun_loc.py`.
- Inference: `python3 pun_loc_inf.py`

## References
- Original code: https://github.com/zoezou2015/PunLocation
- Paper: https://arxiv.org/pdf/1909.00175.pdf
