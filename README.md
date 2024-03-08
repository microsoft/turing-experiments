# Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies

_TLDR:_ Which human behaviors can your large language model simulate? Turing _Experiments_ are better than the Turing Test.

This repo has the code and most of the data for the paper:

```
@inproceedings{turingExp22,
  title={Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies},
  author={Aher, Gati V and Arriaga, Rosa I and Kalai, Adam Tauman},
  booktitle={Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year={2023},
  url={https://arxiv.org/abs/2208.10264},
  organization={PMLR}
}
```

Submitted to [arXiv](https://arxiv.org/abs/2208.10264) on August 18, 2022.

_Abstract:_ We introduce a new type of test, called a Turing Experiment (TE), for evaluating to what extent a given language model, such as GPT models, can simulate different aspects of human behavior. A TE can also reveal consistent distortions in a language model's simulation of a specific human behavior. Unlike the Turing Test, which involves simulating a single arbitrary individual, a TE requires simulating a representative sample of participants in human subject research. We carry out TEs that attempt to replicate well-established findings from prior studies. We design a methodology for simulating TEs and illustrate its use to compare how well different language models are able to reproduce classic economic, psycholinguistic, and social psychology experiments: Ultimatum Game, Garden Path Sentences, Milgram Shock Experiment, and Wisdom of Crowds. In the first three TEs, the existing findings were replicated using recent models, while the last TE reveals a "hyper-accuracy distortion" present in some language models (including ChatGPT and GPT-4), which could affect downstream applications in education and the arts.

_Keywords:_ Turing Test, Large Language Models, Evaluation Metrics

---

## Requirements

1. Install necessary dependencies with conda:

```
conda env create -n turing-experiments -f environment.yml
conda activate turing-experiments
```

2. To use OpenAI's language model engines to generate responses add your api key and organization as plaintext to `openai_api_key.txt` and `openai_organization.txt` in the root directory (these are ignored by `.gitignore`).

3. To download the authors' data files, install and enable [git LFS](https://git-lfs.com/). Then download the data files using:

```
git lfs pull
```

The data files are large, so it might take several minutes (~5-10 min) for git LFS to download them from the remote server. Alternatively, the data files can be downloaded from GitHub using the "download raw files" button.

## Usage

For the _Ultimatum Game_ TE, _Garden Path_ TE, and the _Wisdom of Crowds_ TE, we provide all prompt templates and simulation result data to aid both re-running simulations or re-analyzing results.

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

Enable git LFS and run `git lfs pull` to see the authors' consolidated results data files in the `data/simulation_results_consolidated/ultimatum_game/` folder.

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

Enable git LFS and run `git lfs pull` to see the authors' consolidated results data files in the `data/simulation_results_consolidated/garden_path/` folder.

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

The author's consolidated results are avalible in:

- [orginal TE](/data/simulation_results_consolidated/milgram/Milgram_results_experiment_overview.json.gz)
- [alternative TE](/data/simulation_results_consolidated/milgram/Milgram_results_experiment_overview_alt_milg.json.gz)

Decompress these files into .json format by using `gzip -d <filename>`. The experiment overview includes full prompts / transcripts for 100 simulated participants.

Prompt templates are available in the following folders:

- [/data/prompt-templates/milgram/milgram_resources/](/data/prompt-templates/milgram/milgram_resources/)
- [/data/prompt-templates/milgram/milgram_alt_resources/](/data/prompt-templates/milgram/milgram_alt_resources/)

Logic for running the experiment (using language models to simulate both subject responses and experimenter judgments) are given in following Python scripts:

- [/scripts/simulate_milgram/simulate_milgram_experiment.py](/scripts/simulate_milgram/simulate_milgram_experiment.py)
- [/scripts/simulate_milgram/simulate_milgram_experiment_alternate.py](/scripts/simulate_milgram/simulate_milgram_experiment_alternate.py)

Note: davinci-text-002 has been deprecated so calling the OpenAI API for it no longer works. To see how the logic works, we have enabled a global flag that causes the script to return mocked values from the functions that call the OpenAI API.

```python
# Set GLOBAL_VAR_MOCKED = False to call the LLM API
GLOBAL_VAR_MOCKED = True
```

Analysis of the results are given in the following jupyter notebooks:

- [/scripts/simulate_milgram/analyze_milgram.ipynb](/scripts/simulate_milgram/analyze_milgram.ipynb)
- [/scripts/simulate_milgram/analyze_milgram_alternate.ipynb](/scripts/simulate_milgram/analyze_milgram_alternate.ipynb)

Looking at the analysis notebook outputs may be informative in terms of understanding diversity and error modes of language model completions.

Davinci-02 TE vs. Milgram's Original Experiment Findings:
![Davinci-02 TE vs. Milgram's Original Experiment Findings](/results/milgram/milgram_results/fig_subjects_remaining.png)
Davinci-02 Alternate TE vs. Milgram's Original Experiment Findings:
![Davinci-02 Alternate TE vs. Milgram's Original Experiment Findings](/results/milgram/milgram_results/fig_alt_subjects_remaining.png)

## 4. Wisdom of Crowds TE

- Everything is contained in the following jupyter notebook: `scripts\wisdom_of_crowds.ipynb`
- This generates the two figures for the paper in `results\wisdom` folder
- The data are stored in `data\wisdom_results` folder. If you erase the data, the notebook will regenerate them.

# Legal Notices

Microsoft and any contributors grant you a license to any code in the repository under the [MIT License](https://opensource.org/licenses/MIT), see the
[LICENSE](LICENSE) file, and grant you a license to the Microsoft documentation and other data
in this repository under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode),
see the [DATA_LICENSE](data/DATA_LICENSE) file.

Microsoft, Windows, Microsoft Azure and/or other Microsoft products and services referenced in the documentation
may be either trademarks or registered trademarks of Microsoft in the United States and/or other countries.
The licenses for this project do not grant you rights to use any Microsoft names, logos, or trademarks.
Microsoft's general trademark guidelines can be found at http://go.microsoft.com/fwlink/?LinkID=254653.

Privacy information can be found at https://privacy.microsoft.com/en-us/

Microsoft and any contributors reserve all other rights, whether under their respective copyrights, patents,
or trademarks, whether by implication, estoppel or otherwise.
