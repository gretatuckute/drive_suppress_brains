## Driving and suppressing the human language network using large language models

This repository contains code and data accompanying: 
G. Tuckute, A. Sathe, S. Srikant, M. Taliaferro, M. Wang, M. Schrimpf, K. Kay, E. Fedorenko (2023): _Driving and suppressing the human language network using large language models_.

## Environment
The environment is a Python 3.8.11 environment that makes heavy use of [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/stable/), [HuggingFace Transformers](https://huggingface.co/docs/transformers/index), and [matplotlib](https://matplotlib.org/). To use the exact Python environment used in the paper, install it as:

```
conda env create -f env_drive-suppress-brains.yml
```

## Repository organization
 * [data](./data)
 * [data_SI](./data_SI)
 * [env](./env)
 * [model-actv](./model-actv)
   * [gpt2-xl](./model-actv/gpt2-xl)
     * [last-tok](./model-actv/gpt2-xl/last-tok)
 * [regr-weights](./regr-weights)
   * [fit_mapping](./regr-weights/fit_mapping)
     * [last-tok](./regr-weights/fit_mapping/last-tok)
 * [results](./results)
 * [src](./src)
   * [plot_data](./src/plot_data)
   * [run_analyses](./src/run_analyses)
   * [statistics](./src/statistics)
 * [README.md](./README.md)

To populate the `data`, `data_SI`, `model-actv`, and `regr-weights` folders, please see the "Downloading data" section below.

The `data` folder contains a csv file with the event-related data (_brain-lang-data_participant_20230728.csv_; main experiment), a csv file for the blocked experiment (_brain-lang-blocked-data_participant_20230728.csv_), a csv file with the noise ceilings computed based on the event-related data (_NC-allroi-data.csv_), and finally a file with the associated column name descriptions (_column_name_descriptions.csv_). These files are used to run the [Figure_2.ipynb](https://github.com/gretatuckute/drive_suppress_brains/blob/main/src/plot_data/Figure2.ipynb), [Figure_3.ipynb](https://github.com/gretatuckute/drive_suppress_brains/blob/main/src/plot_data/Figure3.ipynb), [Figure_4.ipynb](https://github.com/gretatuckute/drive_suppress_brains/blob/main/src/plot_data/Figure4.ipynb), and [Figure_5.ipynb](https://github.com/gretatuckute/drive_suppress_brains/blob/main/src/plot_data/Figure5.ipynb) notebooks.

The `data_SI` folder contains csv files used to run the [SI_Figures.ipynb](https://github.com/gretatuckute/drive_suppress_brains/blob/main/src/plot_data/SI_Figures.ipynb).

The `env` folder contains the conda yml file _env_drive-suppress-brains.yml_.

The `model-actv` folder contains pre-computed model activations for GPT2-XL (last-token representation). The file _beta-control-neural-T_actv.pkl_ contains the activations for the _baseline_ set in a Pandas dataframe. The rows correspond to sentences, and the columns are multi-indexed according to layer and unit. The first level is layer (49 layers in GPT2-XL) and the second level is unit (1600 units in each representation vector in GPT2-XL). The file _beta-control-neural-T_stim.pkl_ contains the corresponding stimuli metadata in a Pandas dataframe. The two files are row-indexed using the same identifiers. The files _beta-control-neural-D_actv.pkl_ and _beta-control-neural-D_stim.pkl_ contain the activations for the _baseline_ set along with the _drive_/_suppress_ activations (derived via the main search approach).

The `regr-weights` folder contains the encoding model regression weights in the `fit_mapping` subfolder with an additional subfolder according to the parameters that were used to fit the encoding model.

The `results` folder is the default folder for storing outputs from `src/run_analyses`.

The `src` folder contains all code in the following subfolders: 
- `plot_data` contains a notebook that reproduces each of the main figures, as well as a notebook for the SI figures.
-  `run_analyses` contains code to run all main analyses in the paper.
-  `statistics` contains linear mixed effect (LME) statistics (in R).

## Downloading data
To download data used in the paper, run the [download_files.py](https://github.com/gretatuckute/drive_suppress_brains/blob/main/setup_utils/download_files.py) script. By default, it will download the files for the `data` folder. 

The `data` folder contains a csv file with the event-related data (_brain-lang-data_participant_20230728.csv_; main experiment). This file contains brain responses for the left hemisphere (LH) language regions for n=10 participants (n=5 _train_ participants, n=5 _evaluation_ participants) along with various metadata and behavioral data for each sentence (n=10 linguistic properties). The `data` folder also contains a csv file with brain responses for the blocked experiment (_brain-lang-blocked-data_participant_20230728.csv_, n=4 _evaluation_ participants). The folder also contains the noise ceilings computed based on the event-related data on n=5 _train_ participants (_NC-allroi-data.csv_). Finally, the file _column_name_descriptions.csv_ contains descriptions of the content of the columns in these csv files.

Using the additional flags, you can specify whether you want to download the `data_SI` files, the `model-actv` files, and the `regr-weights` files. 

## Analyzing data and generating plots
All code is in `src`. 

- The `src/plot_data` folder contains Jupyter Notebooks that analyze and generate plots for the main results in the paper. 
- The `src/run_analyses` folder contains Python scripts for running analyses. The two main scripts are: 
	1. [/src/run_analyses/fit_mapping.py](https://github.com/gretatuckute/drive_suppress_brains/blob/main/src/run_analyses/fit_mapping.py) fits an encoding model from features from a source model (in this case, GPT2-XL, cached in `model-actv/gpt2-xl`) to the participant-averaged brain data. The script will store outputs in `results` and the fitted regression weights in `regr-weights`.
	2. [/src/run_analyses/use_mapping_external.py](https://github.com/gretatuckute/drive_suppress_brains/blob/main/src/run_analyses/use_mapping_external.py) loads the regression weights from the encoding model and predicts each sentence in the supplied stimulus set.
- The `src/statistics` folder contains R code to run LME models.

<!---
## Citation
If you use this repository or data, please cite:

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

<!---

## XXXXX
We used to common model-brain evaluation metrics, namely regression and representational similarity analysis (RSA), as demonstrated in the figure below.

<img src="./illustrations/fig1.png" width="600"/>

### Regression


#### Note on how DNN unit activations are organized

## Generating plots
The figures in the paper can be reproduced via the notebooks in the [analyze](https://github.com/gretatuckute/auditory_brain_dnn/tree/main/aud_dnn/analyze) directory, e.g., [generate_Figure2.ipynb](https://github.com/gretatuckute/auditory_brain_dnn/blob/main/aud_dnn/analyze/generate_Figure2.ipynb) and so forth.


`.
├── data
├── env
├── model-actv
│    └── gpt2-xl
│          └── last-tok
├── regr-weights
│    └── fit_mapping
│          └── last-tok
├── results
├── src
│   ├── plot_data
│   ├── run_analyses
│   └── statistics
└── README.md
`


--->
