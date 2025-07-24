import os
import json
import torch
import numpy as np
from types import SimpleNamespace
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from dataloaders import dataloader_factory
from models import model_factory

# -------------------------------------------------------------------
# CONFIGURATION (adjust here)
# -------------------------------------------------------------------
# Path to the experiment directory (contains config.json and models/ folder)
EXPERIMENT_DIR = (
    "/home/user/Documents/ColdStart/benchmarking22_0426/benchmarking22/WMTrue_itemshift0_usersplitby_timestamp_maxlen4_datasettoy_sampleone_splitrandom_in_target_temp0.1/test_mask0.6_betas1_betat0.03_betast0.01_betasst0.04_alpha0.2_negseed98765_datasettoy_modelSASRecT_learning_rate0.001_max_len20_num_blocks2_emb_dim64_weight_decay0_drop_rate0.3_drop_rate_emb0.5_num_heads1"
)

# Paths
CONFIG_PATH = os.path.join(EXPERIMENT_DIR, 'config.json')
MODELS_DIR = os.path.join(EXPERIMENT_DIR, 'models')

# Choose device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Automatically pick the checkpoint .pth file (prefer best_acc_model.pth)
pth_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
if not pth_files:
    raise FileNotFoundError(f"No .pth files found in {MODELS_DIR}")
checkpoint_name = 'best_acc_model.pth' if 'best_acc_model.pth' in pth_files else pth_files[0]
CHECKPOINT_PATH = os.path.join(MODELS_DIR, checkpoint_name)


def collect_source_embeddings(model, loader, device, weighted_mean=False):
    """
    Collect embeddings from source loader using weighted or simple mean pooling.
    Expects loader to yield (seqs_s, labels_s, popular_s, ...).
    """
    model.eval()
    embs_list = []
    with torch.no_grad():
        for batch in loader:
            seqs_s, _, popular_s, *rest = batch
            seqs_s = seqs_s.to(device)               # [B, L]
            popular_s = popular_s.to(device).unsqueeze(-1)  # [B, L, 1]
            feats_s, _ = model.model_s(seqs_s)       # [B, L, D]
            if weighted_mean:
                weighted_sum = (feats_s * popular_s).sum(dim=1)  # [B, D]
                norm = popular_s.sum(dim=1).clamp(min=1e-8)      # [B, 1]
                emb = (weighted_sum / norm).cpu().numpy()       # [B, D]
            else:
                emb = feats_s[:, -1, :].cpu().numpy()          # last hidden state
            embs_list.append(emb)
    return np.vstack(embs_list)


def collect_target_embeddings(model, loader, device, weighted_mean=False):
    """
    Collect embeddings from target loader using weighted or simple mean pooling.
    Expects loader to yield (seqs_t, labels_t[, popular_t, ...]).
    """
    model.eval()
    embs_list = []
    with torch.no_grad():
        for batch in loader:
            seqs_t = batch[0].to(device)             # [B, L]
            # attempt to get popularity if present
            if weighted_mean and len(batch) >= 3:
                popular_t = batch[2].to(device).unsqueeze(-1)  # [B, L, 1]
            else:
                popular_t = None
            feats_t, _ = model.model_t(seqs_t)       # [B, L, D]
            if weighted_mean and popular_t is not None:
                weighted_sum = (feats_t * popular_t).sum(dim=1)
                norm = popular_t.sum(dim=1).clamp(min=1e-8)
                emb = (weighted_sum / norm).cpu().numpy()
            else:
                emb = feats_t[:, -1, :].cpu().numpy()
            embs_list.append(emb)
    return np.vstack(embs_list)


def main():
    # 1) Load experiment args
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(f"config.json not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    saved_args = SimpleNamespace(**cfg)

    # 2) Build model and load checkpoint
    model = model_factory(saved_args).to(DEVICE)
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 3) Create data loaders
    train_src, train_tgt, train_combine, _, _ = dataloader_factory(saved_args)

    # 4) Collect embeddings using source/target loaders
    weighted = getattr(saved_args, 'weighted_mean', False)
    print("Collecting source embeddings...")
    source_embs = collect_source_embeddings(model, train_src, DEVICE, weighted_mean=weighted)
    print("Collecting target embeddings...")
    target_embs = collect_target_embeddings(model, train_tgt, DEVICE, weighted_mean=weighted)

    # 5) Combine and label
    embs = np.vstack([source_embs, target_embs])
    labels = np.array([0] * len(source_embs) + [1] * len(target_embs))

    # 6) Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    proj = tsne.fit_transform(embs)

    # 7) Plot & save
    plt.figure(figsize=(7,7), dpi=200)
    plt.scatter(proj[labels==0,0], proj[labels==0,1], s=5, label='source')
    plt.scatter(proj[labels==1,0], proj[labels==1,1], s=5, label='target')
    plt.legend()
    plt.title('t-SNE of Source vs Target Embeddings')
    plt.tight_layout()
    out_path = os.path.join(MODELS_DIR, 'tsne_plot.png')
    plt.show()
    plt.savefig(out_path, dpi=300)
    print(f"t-SNE plot saved to {out_path}")


if __name__ == '__main__':
    main()
