# GitHub Auto-Save - Quick Start

## ✅ What Was Added

Your training script now **automatically commits and pushes results to GitHub**!

## 🎯 Default Behavior

When you run training:
```bash
python train_attention_lightning.py -c config_attention_large
```

At the **end of training**, the script automatically:
1. ✅ Commits `results/` folder (metrics, configs, artifacts)
2. ✅ Commits `logs/` folder (TensorBoard logs)
3. ✅ Pushes to your current branch
4. ❌ **Excludes** `.ckpt` files (they're too large - 100-500 MB each)

## 🔧 Quick Configuration

### Disable Auto-Save
```bash
python train_attention_lightning.py -c config_attention \
    github_autosave.enabled=false
```

### Commit After Each Epoch (Not Recommended)
```bash
python train_attention_lightning.py -c config_attention \
    github_autosave.mode=each_epoch
```
⚠️ Warning: Creates many commits, slows down training!

### Commit Locally, Don't Push
```bash
python train_attention_lightning.py -c config_attention \
    github_autosave.push_to_remote=false
```

## 📁 What Gets Saved to GitHub

✅ **Included**:
- `results/metrics/*.json` - Metrics files (~10 KB each)
- `results/rnn_models/*/config.yaml` - Config files (~5 KB each)
- `results/rnn_models/*/artifacts/*.joblib` - Vocabulary, scalers (~1-5 MB each)
- `logs/` - TensorBoard logs (~10-50 MB)

❌ **Excluded**:
- `*.ckpt` - Checkpoint files (~100-500 MB each)
- `outputs/` - Hydra outputs
- `data/` - Training data
- `__pycache__/` - Python cache

## 📋 Configuration Options

All in your config file under `github_autosave`:

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `enabled` | `true`/`false` | `true` | Enable auto-save |
| `mode` | `"end_of_training"`<br>`"each_epoch"` | `"end_of_training"` | When to commit |
| `push_to_remote` | `true`/`false` | `true` | Push to GitHub |
| `include_checkpoints` | `true`/`false` | `false` | Include .ckpt files |
| `results_dir` | path | `"results"` | Directory to commit |

## 💡 Recommended Settings

### For Most Users (Default)
```yaml
github_autosave:
  enabled: true
  mode: "end_of_training"
  push_to_remote: true
  include_checkpoints: false
```
**Result**: One clean commit when training completes ✅

### For Long Training Runs (> 24 hours)
```yaml
github_autosave:
  mode: "each_epoch"
```
**Result**: Backup progress periodically (creates many commits) ⚠️

### For Testing/Development
```yaml
github_autosave:
  enabled: false
```
**Result**: No automatic commits

## 📝 Example Commit Messages

### End of Training
```
Training completed: attention_lstm_large_20251017_143025 (2025-10-17 14:35:42)

Final metrics: val_f1=0.8120, val_acc=0.8045, val_loss=0.4523
```

### Each Epoch
```
Auto-save attention_lstm_large_20251017_143025 - Epoch 25 (2025-10-17 14:32:15)
Metrics: val_f1=0.8120, val_acc=0.8045
```

## 🚨 Important Notes

### 1. Large Files Are Excluded
- `.ckpt` files are automatically excluded via `.gitignore`
- This prevents repository bloat
- If you need checkpoints, use [Git LFS](https://git-lfs.github.com/)

### 2. Git Must Be Configured
- Git credentials must be set up (SSH or HTTPS)
- Otherwise, push will fail (but commit succeeds locally)

### 3. Each Epoch Mode Creates Many Commits
- 50 epochs = 50 commits!
- Only use for very long training runs
- Can slow down training

## 🔍 Verify It's Working

After training completes, check:

```bash
# View recent commits
git log --oneline -3

# Example output:
# abc1234 Training completed: attention_lstm_large_20251017_143025 (2025-10-17 14:35:42)
# def5678 Training completed: attention_lstm_deep_20251017_120000 (2025-10-17 12:15:30)
# ghi9012 Initial commit
```

## ❓ Troubleshooting

### Push Failed?
```bash
# Push manually
git push origin your-branch-name
```

### Want to Undo Last Commit?
```bash
# Undo commit (keep files)
git reset --soft HEAD~1

# Undo commit and discard changes
git reset --hard HEAD~1
```

### Too Many Commits from Each Epoch Mode?
```bash
# Squash last 50 commits into one
git rebase -i HEAD~50
# Mark all but first as 'squash' or 's'
```

## 📚 Full Documentation

See `GITHUB_AUTOSAVE.md` for complete documentation including:
- Detailed configuration options
- Git LFS setup for large files
- Advanced usage examples
- Troubleshooting guide

---

**Your experiments are now automatically backed up! 🎉**

