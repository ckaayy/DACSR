# Domain Adaptation for Cold‑Start Sequential Recommendation (DACSR)

## Summary

Sequential recommendation tracks users’ preferences over time based on historical activities to predict their next most probable action. This repository contains the DACSR model used to address cold‑start sequential recommendation by treating regular and cold‑start users as source and target domains, respectively, and applying domain adaptation techniques to narrow performance gaps caused by domain shifts.

Key features:
- **Dual‑Transformer Framework**: Separate transformer models for long (source) and short (target) sequences, collaboratively trained with shared item embeddings.
- **Emulated Target Domain**: Sampled short sequences from the source domain to simulate cold‑start conditions and bridge domain gaps via contrastive learning.
- **Robust Empirical Validation**: Experiments on five public datasets demonstrating consistent improvements over strong baselines under both length and item distribution shifts.

## Environment Setup

Follow these steps to create and activate the conda environment defined in `environment.yml`:

### 1. Ensure you have Miniconda or Anaconda installed
Download Miniconda: https://docs.conda.io/en/latest/miniconda.html

### 2. Clone this repository and navigate into it
```bash
git clone <REPO_URL>
cd <REPO_DIRECTORY>
```

### 3. Create the environment from the provided YAML file
```bash
conda env create -f environment.yml
```

### 4. Activate the new environment
Replace <env_name> with the name specified in environment.yml under 'name:'
```bash
conda activate <env_name>
```
## Running Experiments

You can run experiments on different datasets by executing `main_<dataset_name>.py`. For example:

```python
python main_beauty.py --user_split by_timestamp
```

All experiment settings can be adjusted via command-line arguments. Below are some basic options you can adjust:

```python
parser.add_argument(
    '--user_split',
    type=str,
    default='by_random',
    choices=['by_timestamp','by_random'],
    help='split the users into target and source'
)
```
This allows you to switch the user split setting between the **by_random** and **by_timestamp** scenarios.

```python
parser.add_argument(
    '--itemshift',
    type=int,
    default=0,
    help='keep items in target not in source'
)  # 0 indicates target items must all be in source; items not in source are removed
```
This allows you to switch between the **by_timestamp case 1** (filter out target-only items) and **case 2** (retain all target items) scenarios.

## Hyperparameters
We have provided the best sets of hyperparameters in `bestparam.txt` for both split scenarios



