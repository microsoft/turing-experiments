# Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies

_TLDR:_ Which human behaviors can your large language model simulate? Turing _Experiments_ are better than the Turing Test.

This repo has the code and most of the data for the paper:

```
@inproceedings{turingExp22,
  title={Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies},
  author={Gati Aher and Rosa I. Arriaga and Adam Tauman Kalai},
  booktitle={Proceedings of the 40th International Conference on Machine Learning (ICML)},
year={2023},
url={https://arxiv.org/abs/2208.10264}
}
```

Submitted to [arXiv](https://arxiv.org/abs/2208.10264) on August 18, 2022.

_Abstract:_ We introduce a new type of test, called a Turing Experiment (TE), for evaluating how well a language model, such as GPT-3, can simulate different aspects of human behavior. Unlike the Turing Test, which involves simulating a single arbitrary individual, a TE requires simulating a representative sample of participants in human subject research. We give TEs that attempt to replicate well-established findings in prior studies. We design a methodology for simulating TEs and illustrate its use to compare how well different language models are able to reproduce classic economic, psycholinguistic, and social psychology experiments: Ultimatum Game, Garden Path Sentences, Milgram Shock Experiment, and Wisdom of Crowds. In the first three TEs, the existing findings were replicated using recent models, while the last TE reveals a "hyper-accuracy distortion"' present in some language models.

_Keywords:_ Turing Test, Large Language Models, Evaluation Metrics

---

## Requirements

1. Install necessary dependencies with conda:

```
conda env create -n humansim -f environment.yaml
conda activate humansim
```

2. To use OpenAI's language model engines to generate responses add your api key and organization as plaintext to `openai_api_key.txt` and `openai_organization.txt` (these are ignored by `.gitignore`).

## Usage

For the _Ultimatum Game_ TE, _Garden Path_ TE,  and the _Wisdom of Crowds_ TE, we provide all prompt templates and simulation result data to aid both re-running simulations or re-analyzing results.

For the _Milgram Shock_ TE we provide all prompt templates and, due to space concerns, a selection of representative result data files for both the original and alternative experiment scenarios.

- `scripts/` folder - contains jupyter notebooks and Python scripts for running and analyzing the experiments.
- `src/` folder - contains reusable modules and helper functions
- `data/` folder - contains data files
- `results/` folder - contains final analysis products, like figures

## 1. Ultimatum Game TE

[/scripts/Simulate_Ultimatum_Game_Experiment.ipynb](/scripts/Simulate_Ultimatum_Game_Experiment.ipynb) contains a notebook to run and analyze the Ultimatum Game TE. The prompt templates are given in

- [/data/prompt-templates/ultimatum_game/no-complete-accept.txt](/data/prompt-templates/ultimatum_game/no-complete-accept.txt)
- [/data/prompt-templates/ultimatum_game/no-complete-reject.txt](/data/prompt-templates/ultimatum_game/no-complete-reject.txt)

To query the OpenAI language models and generate new simulation results, uncomment "Section 4. Run Experiment".

Else, download consolidated results data files to the `data/simulation_results_consolidated/ultimatum_game/` folder.

```
.
└── data
    ...
    └── simulation_results_consolidated
        └── ultimatum_game
            ├── README.md
            ├── UG_surnames_total_money_10_text-ada-001_no-complete-accept.json.gz
            ├── UG_surnames_total_money_10_text-ada-001_no-complete-reject.json.gz
            ├── UG_surnames_total_money_10_text-babbage-001_no-complete-accept.json.gz
            ├── UG_surnames_total_money_10_text-babbage-001_no-complete-reject.json.gz
            ├── UG_surnames_total_money_10_text-curie-001_no-complete-accept.json.gz
            ├── UG_surnames_total_money_10_text-curie-001_no-complete-reject.json.gz
            ├── UG_surnames_total_money_10_text-davinci-001_no-complete-accept.json.gz
            ├── UG_surnames_total_money_10_text-davinci-001_no-complete-reject.json.gz
            ├── UG_surnames_total_money_10_text-davinci-002_no-complete-accept.json.gz
            └── UG_surnames_total_money_10_text-davinci-002_no-complete-reject.json.gz
```

Then run the jupyter notebook to generate the following analysis figures:

Ada vs. Davinci-02 vs. Humans:
![Ada vs. Davinci-02 vs. Humans](/results/ultimatum_game/UG_surnames_total_money_10_fig_ug_ada_dav.png)
Davinci-02 Correlation:
![Davinci-02 Correlation](/results/ultimatum_game/UG_surnames_total_money_10_fig_ug_corr_annot.png)
Davinci-02 Chivalry Effect Average:
![Davinci-02 Chivalry Effect Average](/results/ultimatum_game/UG_surnames_total_money_10_fig_ug_gender1.png)
Davinci-02 Chivalry Effect Histogram:
![Davinci-02 Chivalry Effect Histogram](/results/ultimatum_game/UG_surnames_total_money_10_fig_ug_gender2.png)

## 2. Garden Path TE

[/scripts/Simulate_Garden_Path_Experiment.ipynb](/scripts/Simulate_Garden_Path_Experiment.ipynb) contains a notebook to run and analyze the Garden Path TE. The prompt templates are given in

- [/data/prompt-templates/garden_path/no-complete-grammatical.txt](/data/prompt-templates/garden_path/no-complete-grammatical.txt)
- [/data/prompt-templates/garden_path/no-complete-ungrammatical.txt](/data/prompt-templates/garden_path/no-complete-ungrammatical.txt)

The sentence stimuli (garden path sentences and controls) are given in

- [/data/external/garden_path/Christianson_2001.tsv](/data/external/garden_path/Christianson_2001.tsv)
- [/data/external/garden_path/Alternates_2022.tsv](/data/external/garden_path/Alternates_2022.tsv)

Download consolidated results data files to the `data/simulation_results_consolidated/garden_path/` folder.

```
.
└── data
    ...
    └── simulation_results_consolidated
        └── garden_path
            ├── README.md
            ├── GP_surnames_Alternates_2022.json.gz
            └── GP_surnames_Christianson_2001.json.gz
```

Then run the jupyter notebook at `scripts/Simulate_Garden_Path_Experiment.ipynb`. The notebook generates the following figures:

Original Sentences Simulated Ratings:
![Original Sentences Simulated Ratings](/results/garden_path/GP_surnames_Christianson_2001_fig_gp.png)
Original Sentences Simulated Ratings - Control vs. Garden Path:
![Original Sentences Simulated Ratings - Control vs. Garden Path](/results/garden_path/GP_surnames_Christianson_2001_fig_gp_c_vs_gp.png)

By uncommenting lines in Section 1.5. "Experimental Conditions Settings", choose to simulate and analyze results from using the original stimuli (Christianson et al, 2001) or novel alternate stimuli sentences written by the authors.

```python
# Original Sentences
experiment_descriptor_sentences = "Christianson_2001"

# Novel Sentences
# experiment_descriptor_sentences = "Alternates_2022"
```

## 3. Milgram Shock TE

Due to the length of the prompt and corresponding cost of running this experiment, the authors do not recommend re-running the experiment or its alternate. We do provide all prompt-templates and code logic used so that others may examine the procedure and replicate if they choose.

Prompt templates are available in the following folders:

- [/data/prompt-templates/milgram/milgram_resources/](/data/prompt-templates/milgram/milgram_resources/)
- [/data/prompt-templates/milgram/milgram_alt_resources/](/data/prompt-templates/milgram/milgram_alt_resources/)

Logic for running the experiment (using language models to simulate both subject responses and experimenter judgments) are given in following Python scripts:

- [/scripts/simulate_milgram/simulate_milgram_experiment.py](/scripts/simulate_milgram/simulate_milgram_experiment.py)
- [/scripts/simulate_milgram/simulate_milgram_experiment_alternate.py](/scripts/simulate_milgram/simulate_milgram_experiment_alternate.py)

Analysis of the results are given in the following jupyter notebooks:

- [/scripts/simulate_milgram/analyze_milgram.ipynb](/scripts/simulate_milgram/analyze_milgram.ipynb)
- [/scripts/simulate_milgram/analyze_milgram_alternate.ipynb](/scripts/simulate_milgram/analyze_milgram_alternate.ipynb)

Looking at the analysis notebook outputs may be informative in terms of understanding diversity and error modes of language model completions.

Davinci-02 Orignal Experiment vs. Milgram's Original Experiment Findings:
![Davinci-02 Orignal Experiment vs. Milgram's Original Experiment Findings](/results/milgram/milgram_results/fig_subjects_remaining.png)
Davinci-02 Alternate Experiment vs. Milgram's Original Experiment Findings:
![Davinci-02 Alternate Experiment vs. Milgram's Original Experiment Findings](/results/milgram/milgram_results/fig_alt_subjects_remaining.png)

## 4. Wisdom of Crowds TE

- Everything is contained in the following jupyter notebook: `scripts\wisdom_of_crowds.ipynb`
- This generates the two figures for the paper in `results\wisdom` folder
- The data are stored in `data\wisdom_results` folder. If you erase the data, the notebook will regenerate them.
