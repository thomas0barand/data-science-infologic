"""
GitHub Auto-Save Callback for PyTorch Lightning

Automatically commits and pushes results to GitHub after each epoch.

⚠️  WARNINGS:
- Creates many commits (one per epoch)
- Can slow down training significantly
- Large checkpoint files may bloat repository
- Requires git credentials to be configured

RECOMMENDATIONS:
- Use .gitignore to exclude large checkpoint files
- Consider committing only metrics/logs
- Use a separate branch for experiments
- Set up git LFS for large files
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class GitHubAutoSaveCallback(Callback):
    """
    PyTorch Lightning callback to automatically commit and push to GitHub after each epoch.
    
    Args:
        results_dir (str): Path to results directory to commit (default: "results")
        commit_message_prefix (str): Prefix for commit messages
        push_to_remote (bool): Whether to push to remote (default: True)
        branch (str): Branch name (default: current branch)
        include_checkpoints (bool): Whether to include checkpoint files (default: False)
        verbose (bool): Whether to print git commands (default: True)
    """
    
    def __init__(
        self,
        results_dir: str = "results",
        commit_message_prefix: str = "Auto-save training",
        push_to_remote: bool = True,
        branch: str = None,
        include_checkpoints: bool = False,
        verbose: bool = True
    ):
        super().__init__()
        self.results_dir = results_dir
        self.commit_message_prefix = commit_message_prefix
        self.push_to_remote = push_to_remote
        self.branch = branch
        self.include_checkpoints = include_checkpoints
        self.verbose = verbose
        self.repo_root = self._find_repo_root()
        
        if not self.repo_root:
            print("⚠️  Warning: Not in a git repository. GitHubAutoSaveCallback disabled.")
            self.enabled = False
        else:
            self.enabled = True
            if self.verbose:
                print(f"✓ GitHubAutoSaveCallback enabled for: {self.results_dir}")
                print(f"  Repository root: {self.repo_root}")
                if not self.include_checkpoints:
                    print(f"  ⚠️  Checkpoint files (.ckpt) will be excluded")
    
    def _find_repo_root(self):
        """Find the git repository root."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def _run_git_command(self, command, check=True):
        """Run a git command and return the result."""
        try:
            if self.verbose:
                print(f"  $ git {' '.join(command)}")
            
            result = subprocess.run(
                ["git"] + command,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=check
            )
            
            if result.stdout and self.verbose:
                print(f"    {result.stdout.strip()}")
            
            return result
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Git command failed: {e}")
            if e.stderr:
                print(f"     {e.stderr.strip()}")
            return None
    
    def _ensure_gitignore_has_checkpoints(self):
        """Ensure .gitignore excludes checkpoint files if needed."""
        if self.include_checkpoints:
            return
        
        gitignore_path = os.path.join(self.repo_root, ".gitignore")
        checkpoint_pattern = "*.ckpt"
        
        # Check if .gitignore exists and has the pattern
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                content = f.read()
                if checkpoint_pattern in content:
                    return
        
        # Add the pattern
        with open(gitignore_path, 'a') as f:
            f.write(f"\n# Auto-added by GitHubAutoSaveCallback\n")
            f.write(f"{checkpoint_pattern}\n")
        
        if self.verbose:
            print(f"  ✓ Added '{checkpoint_pattern}' to .gitignore")
    
    def _commit_and_push(self, epoch: int, metrics: dict = None):
        """Commit and push changes to GitHub."""
        if not self.enabled:
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Build commit message
        commit_msg = f"{self.commit_message_prefix} - Epoch {epoch} ({timestamp})"
        if metrics:
            # Add key metrics to commit message
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items() 
                                   if isinstance(v, (int, float))])
            if metric_str:
                commit_msg += f"\nMetrics: {metric_str}"
        
        if self.verbose:
            print(f"\n📝 Committing results to git...")
        
        # Ensure checkpoint files are ignored if needed
        if not self.include_checkpoints:
            self._ensure_gitignore_has_checkpoints()
        
        # Add results directory
        self._run_git_command(["add", self.results_dir])
        
        # Also add logs if they exist
        if os.path.exists(os.path.join(self.repo_root, "logs")):
            self._run_git_command(["add", "logs"])
        
        # Check if there are changes to commit
        status = self._run_git_command(["status", "--porcelain"], check=False)
        if not status or not status.stdout.strip():
            if self.verbose:
                print("  ℹ️  No changes to commit")
            return
        
        # Commit
        result = self._run_git_command(["commit", "-m", commit_msg])
        if not result:
            print("  ⚠️  Commit failed, skipping push")
            return
        
        # Push to remote
        if self.push_to_remote:
            if self.verbose:
                print(f"  🚀 Pushing to remote...")
            
            # Get current branch if not specified
            branch = self.branch
            if not branch:
                result = self._run_git_command(["branch", "--show-current"], check=False)
                if result and result.stdout:
                    branch = result.stdout.strip()
            
            if branch:
                push_result = self._run_git_command(["push", "origin", branch], check=False)
                if push_result and push_result.returncode == 0:
                    if self.verbose:
                        print(f"  ✓ Successfully pushed to origin/{branch}")
                else:
                    print(f"  ⚠️  Push failed. You may need to push manually later.")
            else:
                print(f"  ⚠️  Could not determine branch name. Skipping push.")
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when the train epoch ends."""
        # Get current metrics
        metrics = {}
        if trainer.callback_metrics:
            # Extract key metrics
            for key in ['train_loss', 'train_acc', 'train_f1', 
                       'val_loss', 'val_acc', 'val_f1']:
                if key in trainer.callback_metrics:
                    metrics[key] = float(trainer.callback_metrics[key])
        
        # Commit and push
        self._commit_and_push(trainer.current_epoch, metrics)


class GitHubEndOfTrainingCallback(Callback):
    """
    Simpler callback that only commits at the end of training (recommended).
    
    Args:
        results_dir (str): Path to results directory to commit
        commit_message (str): Commit message (default: auto-generated)
        push_to_remote (bool): Whether to push to remote (default: True)
        include_checkpoints (bool): Whether to include checkpoint files (default: False)
    """
    
    def __init__(
        self,
        results_dir: str = "results",
        commit_message: str = None,
        push_to_remote: bool = True,
        include_checkpoints: bool = False,
        model_name: str = None
    ):
        super().__init__()
        self.results_dir = results_dir
        self.commit_message = commit_message
        self.push_to_remote = push_to_remote
        self.include_checkpoints = include_checkpoints
        self.model_name = model_name
        self.repo_root = self._find_repo_root()
        
        if not self.repo_root:
            print("⚠️  Warning: Not in a git repository. GitHubEndOfTrainingCallback disabled.")
            self.enabled = False
        else:
            self.enabled = True
    
    def _find_repo_root(self):
        """Find the git repository root."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def _run_git_command(self, command):
        """Run a git command."""
        try:
            result = subprocess.run(
                ["git"] + command,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False
            )
            return result
        except Exception as e:
            print(f"  ❌ Git command failed: {e}")
            return None
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when training ends."""
        if not self.enabled:
            return
        
        print(f"\n" + "=" * 80)
        print("📝 Committing final results to GitHub...")
        print("=" * 80)
        
        # Ensure checkpoint files are ignored if needed
        if not self.include_checkpoints:
            gitignore_path = os.path.join(self.repo_root, ".gitignore")
            if os.path.exists(gitignore_path):
                with open(gitignore_path, 'r') as f:
                    if "*.ckpt" not in f.read():
                        with open(gitignore_path, 'a') as fa:
                            fa.write("\n# Exclude large checkpoint files\n*.ckpt\n")
        
        # Build commit message
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.commit_message:
            msg = self.commit_message
        else:
            model_name = self.model_name or "model"
            msg = f"Training completed: {model_name} ({timestamp})"
            
            # Add final metrics
            if trainer.callback_metrics:
                metrics = []
                for key in ['val_f1', 'val_acc', 'val_loss']:
                    if key in trainer.callback_metrics:
                        value = float(trainer.callback_metrics[key])
                        metrics.append(f"{key}={value:.4f}")
                if metrics:
                    msg += f"\n\nFinal metrics: {', '.join(metrics)}"
        
        # Add results
        self._run_git_command(["add", self.results_dir])
        if os.path.exists(os.path.join(self.repo_root, "logs")):
            self._run_git_command(["add", "logs"])
        
        # Check for changes
        status = self._run_git_command(["status", "--porcelain"])
        if not status or not status.stdout.strip():
            print("  ℹ️  No changes to commit")
            return
        
        # Commit
        print(f"  Committing changes...")
        result = self._run_git_command(["commit", "-m", msg])
        if not result or result.returncode != 0:
            print("  ⚠️  Commit failed")
            return
        
        print(f"  ✓ Committed successfully")
        
        # Push
        if self.push_to_remote:
            print(f"  Pushing to remote...")
            result = self._run_git_command(["branch", "--show-current"])
            if result and result.stdout:
                branch = result.stdout.strip()
                push_result = self._run_git_command(["push", "origin", branch])
                if push_result and push_result.returncode == 0:
                    print(f"  ✓ Pushed to origin/{branch}")
                else:
                    print(f"  ⚠️  Push failed. You may need to push manually.")
        
        print("=" * 80)

