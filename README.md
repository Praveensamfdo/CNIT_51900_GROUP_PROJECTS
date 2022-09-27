- Download the repository.
- It is recommended to use a virtual environment to run this project.
- Create a virtual environment in the base directory using the following command: `conda env create  -f environment.yml --prefix ./envs`.
- Activate the virtual environment using the following command: `conda activate ./envs` (use `conda deactivate` to exit from the virtual environment).

- Create `model` and `embeddings` folders in the base directory.
- Download the pretrained word embeddings glove.6B.100d.txt [https://nlp.stanford.edu/data/glove.6B.zip]. Put the file in `embeddings` folder.
- Run `python3 final.py` for training and inference.
