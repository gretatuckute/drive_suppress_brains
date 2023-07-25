## Driving and suppressing the human language network using large language models

This repository contains code and data accompanying: 
G. Tuckute, A. Sathe, S. Srikant, M. Taliaferro, M. Wang, M. Schrimpf, K. Kay, E. Fedorenko (2023): _Driving and suppressing the human language network using large language models_.

## Environment
The environment is a Python 3.8.11 environment that makes heavy use of [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/stable/), [HuggingFace Transformers](https://huggingface.co/docs/transformers/index), and [matplotlib](https://matplotlib.org/). To use the exact Python environment used in the paper, install it as:

```
conda env create -f env_drive-suppress-brains.yml
```

<!---

## XXXXX
We used to common model-brain evaluation metrics, namely regression and representational similarity analysis (RSA), as demonstrated in the figure below.

<img src="./illustrations/fig1.png" width="600"/>


### Regression
To perform regression from DNN activations (regressors) to brain/component responses, run [/aud_dnn/AUD_main.py](https://github.com/gretatuckute/auditory_brain_dnn/blob/main/aud_dnn/AUD_main.py). This script 1. Loads a DNN unit activations from a given model (*source_model*) and layer (*source_layer*), 2. Loads the target (*target*) of interest (either neural data: *NH2015* (Norman-Haignere et al., 2015; 7,694 voxels across 8 participants) or *B2021* (Boebinger et al., 2021; 26,792 voxels across 20 participants), or component data *NH2015comp* (Norman-Haignere et al., 2015; 6 components), 3. Runs a ridge-regression across 10 splits of the data (165 sounds; 83 sounds in train and 82 sounds in test) and stores the outputs in /results/ in subfolders with an identifier corresponding to the DNN name.

#### Note on how DNN unit activations are organized

## Generating plots
The figures in the paper can be reproduced via the notebooks in the [analyze](https://github.com/gretatuckute/auditory_brain_dnn/tree/main/aud_dnn/analyze) directory, e.g., [generate_Figure2.ipynb](https://github.com/gretatuckute/auditory_brain_dnn/blob/main/aud_dnn/analyze/generate_Figure2.ipynb) and so forth.


## Citation
```
@article{Tuckute2023.04.16.537080,
	abstract = {Transformer language models are today{\textquoteright}s most accurate models of language processing in the brain. Here, using fMRI-measured brain responses to 1,000 diverse sentences, we develop a GPT-based encoding model and use this model to identify new sentences that are predicted to drive or suppress responses in the human language network. We demonstrate that these model-selected {\textquoteleft}out-of-distribution{\textquoteright} sentences indeed drive and suppress activity of human language areas in new individuals (86\% increase and 98\% decrease relative to the average response to diverse naturalistic sentences). A systematic analysis of the model-selected sentences reveals that surprisal and well-formedness of linguistic input are key determinants of response strength in the language network. These results establish the ability of brain-aligned models to noninvasively control neural activity in higher-level cortical areas, like the language network.Competing Interest StatementThe authors have declared no competing interest.},
	author = {Greta Tuckute and Aalok Sathe and Shashank Srikant and Maya Taliaferro and Mingye Wang and Martin Schrimpf and Kendrick Kay and Evelina Fedorenko},
	doi = {10.1101/2023.04.16.537080},
	elocation-id = {2023.04.16.537080},
	eprint = {https://www.biorxiv.org/content/early/2023/05/06/2023.04.16.537080.full.pdf},
	journal = {bioRxiv},
	publisher = {Cold Spring Harbor Laboratory},
	title = {Driving and suppressing the human language network using large language models},
	url = {https://www.biorxiv.org/content/early/2023/05/06/2023.04.16.537080},
	year = {2023},
	bdsk-url-1 = {https://www.biorxiv.org/content/early/2023/05/06/2023.04.16.537080},
	bdsk-url-2 = {https://doi.org/10.1101/2023.04.16.537080}}
```

--->
