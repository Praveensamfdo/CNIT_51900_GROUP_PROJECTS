## Reproducibility steps
- Download the repository.
- It is recommended to use a virtual environment to run this project.
    - Create a virtual environment in the base directory using the following command: `conda env create  -f environment.yml --prefix ./envs`.
    - Activate the virtual environment using the following command: `conda activate ./envs` (use `conda deactivate` to exit from the virtual environment).

- Create `model` and `embeddings` folders in the base directory.
- Download the pretrained word embeddings [glove.6B.100d.txt](https://nlp.stanford.edu/data/glove.6B.zip "glove.6B.100d.txt"). Put the file in `embeddings` folder.
- Run `python3 final.py` for training and inference.

## Transformer-based pun detection and location
- A transformer based implementation can be found on `temp.py` (for training), and `tok.py` (for inference).

## References
- Original code: https://github.com/zoezou2015/PunLocation
- Paper: https://arxiv.org/pdf/1909.00175.pdf
