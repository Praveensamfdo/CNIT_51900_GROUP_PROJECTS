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

# Group project 02

- heterographic puns (where the pun and its latent target are phonologically similar) have been considered for this task.
- Use `temp.py` to play with the heterographic puns in SemEval 2017 dataset.

```json
{'sentence': 'Do hotel managers get board with their jobs ?', 'label': tensor(1), 'pun_word': 'board', 'pun_location': 4}
{'sentence': 'I started to work at a seafood buffet , but then I pulled a mussel .', 'label': tensor(1), 'pun_word': 'mussel', 'pun_location': 14}
{'sentence': 'The discovery of how to make steel was ironic .', 'label': tensor(1), 'pun_word': 'ironic', 'pun_location': 8}
{'sentence': "I knew that the spirit couldn ' t float around very long . What ghost up must come down .", 'label': tensor(1), 'pun_word': 'ghost', 'pun_location': 14}
{'sentence': "There ' s nothing to stop me putting things in tins , said Tom cannily .", 'label': tensor(1), 'pun_word': 'cannily', 'pun_location': 14}
{'sentence': 'My name is Jim I train boxers', 'label': tensor(1), 'pun_word': 'Jim', 'pun_location': 3}
{'sentence': 'Do you know if they ever got detective Sam spayed .', 'label': tensor(1), 'pun_word': 'spayed', 'pun_location': 9}
{'sentence': 'There are no answers , only cross - references .', 'label': tensor(0), 'pun_word': None, 'pun_location': None}
{'sentence': 'Getting soap in your eyes is no lathering matter .', 'label': tensor(1), 'pun_word': 'lathering', 'pun_location': 7}
{'sentence': "You can ' t trust a tiger . You never know when he might be lion .", 'label': tensor(1), 'pun_word': 'lion', 'pun_location': 15}
{'sentence': 'Cheaters never prosper .', 'label': tensor(0), 'pun_word': None, 'pun_location': None}
```
