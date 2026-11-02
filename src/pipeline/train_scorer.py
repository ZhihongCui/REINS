# train_rule_scorer_only.py
import os
import sys
import json
import torch
import numpy as np

# Add project root to path for direct execution
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from src.utils.sampler import Sampler
from src.rule_system.scorer import build_rule_scorer, RuleScoringDataset
from src.utils.common import load_yaml
from src.pipeline.train import collate_fn, clean_rule_text, _flatten_state, evaluate_rule_scorer

def train_rule_scorer_only(
    samples_path="data/samples/Roundabout/samples_roundabout-v0_ep30.json",
    rules_path="data/extracted_rules.json",
    vocab_path="config/vocab.json",
    num_epochs=5,
    out_prefix="result/Roundabout"
):
    os.makedirs(out_prefix, exist_ok=True)
    ckpt_dir   = os.path.join(out_prefix, "checkpoints")
    fig_dir    = os.path.join(out_prefix, "figures")
    log_dir    = os.path.join(out_prefix, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1) Load samples
    print(f"[RuleScorer] Loading samples from {samples_path}")
    with open(samples_path, "r") as f:
        samples = json.load(f)

    # 2) Load rules (already calibrated + clean)
    print(f"[RuleScorer] Loading rules from {rules_path}")
    with open(rules_path, "r") as f:
        enriched_rules = json.load(f)

    # 3) Load vocab
    print(f"[RuleScorer] Loading vocab from {vocab_path}")
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    # 4) State dimension
    sample_timestep = samples[0]["trajectory"][0]
    state_dim = len(_flatten_state(sample_timestep))
    print(f"[INFO] State vector dimension = {state_dim}")

    rule_scorer = build_rule_scorer(
        method="transformer",
        state_dim=state_dim,
        vocab_path=vocab_path
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device = {device}")
    rule_scorer = rule_scorer.to(device)

    # Only point raw_text to clean text
    train_ready_rules = []
    for r in enriched_rules:
        rr = dict(r)
        rr["raw_text"] = r.get("rule_text_clean", r.get("rule_text", r.get("raw_text", "")))
        train_ready_rules.append(rr)

    # 5) Build dataset
    full_dataset = RuleScoringDataset(samples, train_ready_rules, vocab)
    print(f"[DEBUG] full_dataset size = {len(full_dataset)}")

    if len(full_dataset) == 0:
        raise RuntimeError("full_dataset is empty, RuleScorer cannot be trained")

    val_size = int(0.2 * len(full_dataset)) if len(full_dataset) > 0 else 0
    train_size = max(0, len(full_dataset) - val_size)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size]) if val_size > 0 else (full_dataset, [])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn) if val_size > 0 else None

    labels = [label.item() for _, _, label in full_dataset]
    pos_ratio = np.mean(labels) if labels else 0.0
    print(f"[INFO] Total training samples: {len(full_dataset)}")
    print(f"[INFO] Positive ratio: {pos_ratio:.4f}")
    print(f"[INFO] Train size = {train_size}, Val size = {val_size}")

    # 6) Train RuleScorer
    optimizer = torch.optim.Adam(rule_scorer.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val = float("inf")
    patience, bad_epochs = 5, 0
    epoch_losses = []
    epoch_step_losses = {}

    for epoch in range(num_epochs):
        rule_scorer.train()
        total_loss = 0.0
        step_losses = []

        for step, (state_vec, rule_texts, labels_t) in enumerate(train_loader, 1):
            labels_smooth = 0.05 + 0.9 * labels_t

            state_vec    = state_vec.to(device)
            labels_t     = labels_t.to(device)
            rule_logit, _ = rule_scorer(state_vec, rule_texts)
            loss = criterion(rule_logit, labels_smooth)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rule_scorer.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            step_losses.append(loss.item())

            if step % 20 == 0 or step == len(train_loader):
                print(f"[Epoch {epoch+1}/{num_epochs} | Step {step}/{len(train_loader)}] loss={loss.item():.4f}")

        avg_train_loss = total_loss / max(1, len(train_loader))
        epoch_step_losses[epoch + 1] = step_losses

        # Validation
        val_loss, auc = 0.0, 0.0
        if val_loader and len(val_loader) > 0:
            rule_scorer.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for state_vec, rule_texts, labels_v in val_loader:
                    state_vec = state_vec.to(device)
                    labels_v = labels_v.to(device)
                    rule_logit, _ = rule_scorer(state_vec, rule_texts)
                    val_loss += criterion(rule_logit, labels_v).item()
                    y_true.extend(labels_v.cpu().numpy())
                    y_pred.extend(torch.sigmoid(rule_logit).cpu().numpy())
            val_loss /= max(1, len(val_loader))
            try:
                auc = roc_auc_score(y_true, y_pred)
            except Exception:
                auc = float("nan")

        epoch_losses.append(val_loss)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train={avg_train_loss:.4f}  Val={val_loss:.4f}  AUC={auc:.4f}")

        if val_loader and val_loss < best_val - 1e-4:
            best_val = val_loss
            bad_epochs = 0
            torch.save(rule_scorer.state_dict(), os.path.join(ckpt_dir, "rule_scorer.pt"))
            print(f"[OK] Saved checkpoint -> {os.path.join(ckpt_dir, 'rule_scorer.pt')}")
        elif val_loader:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[Early Stop] Best Val Loss = {best_val:.4f}")
                break

        scheduler.step()

    # 6.5) Evaluation
    print("=" * 60)
    print("[Step 6.5] Evaluating RuleScorer performance (only)")
    rule_scorer.eval()
    eval_ds = val_ds if (val_ds and len(val_ds) > 0) else full_dataset
    evaluate_rule_scorer(rule_scorer, eval_ds, device, tag="val", out_dir=fig_dir)

    # 7) Save loss curves
    np.savetxt(os.path.join(log_dir, "loss_history_epoch.txt"),
               np.array(epoch_losses), fmt="%.6f")

    # Per-step training loss plot for the first few epochs
    first_n = min(3, len(epoch_step_losses))
    steps_per_epoch = len(train_loader)
    plt.figure(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, first_n))
    for idx, ep in enumerate(range(first_n)):
        raw_losses = epoch_step_losses[ep + 1]
        if len(raw_losses) > 0:
            smooth_losses = gaussian_filter1d(raw_losses, sigma=8)
            plt.plot(range(1, len(smooth_losses) + 1), smooth_losses,
                     color=colors[idx], linewidth=2, label=f"Epoch {ep+1}")
    plt.xlabel("Step"); plt.ylabel("Smoothed Training Loss")
    plt.title("Step-wise Smoothed Training Loss (First Epochs)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "step_loss_smoothed_firstN.png"), dpi=300)
    plt.close()
    print(f"[OK] Saved RuleScorer-only results into {out_prefix}")

if __name__ == "__main__":
    train_rule_scorer_only()
