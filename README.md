# Domain Adaptation for Cold‑Start Sequential Recommendation (DACSR)

## Summary

Sequential recommendation tracks users’ preferences over time based on historical activities to predict their next most probable action. This repository addresses cold‑start sequential recommendation by treating regular and cold‑start users as source and target domains, respectively, and applying domain adaptation techniques to narrow performance gaps caused by domain shifts.

Key features:
- **Dual‑Transformer Framework**: Separate transformer models for long (source) and short (target) sequences, collaboratively trained with shared item embeddings.
- **Emulated Target Domain**: Sampled short sequences from the source domain to simulate cold‑start conditions and bridge domain gaps via contrastive learning.
- **Advanced Adaptation Variants**: Three model variants (DACSR, DACSR+, DACSR++) progressively reduce item popularity bias and incorporate user similarity into the contrastive loss for enhanced robustness under compound shifts.
- **Robust Empirical Validation**: Experiments on five public datasets demonstrating consistent improvements over strong baselines under both length and item distribution shifts.

## Environment Setup

Follow these steps to create and activate the conda environment defined in `environment.yml`:

```bash
# 1. Ensure you have Miniconda or Anaconda installed
#    Download Miniconda: https://docs.conda.io/en/latest/miniconda.html

# 2. Clone this repository and navigate into it
git clone <REPO_URL>
cd <REPO_DIRECTORY>

# 3. Create the environment from the provided YAML file
conda env create -f environment.yml

# 4. Activate the new environment
#    Replace <env_name> with the name specified in environment.yml under 'name:'
conda activate <env_name>

# 5. Verify installation (optional)
python -c "import torch; print(f'Torch version: {torch.__version__}')"
```

### Updating the Environment

If dependencies change, update your environment with:

```bash
conda env update -f environment.yml --prune
```

### Deactivating the Environment

To exit the environment when finished:

```bash
conda deactivate
```
