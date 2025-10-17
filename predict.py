import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from utils import load_data, clean_data, tokenize_actions

@hydra.main(version_base=None, config_path="config", config_name="config_attention")
def main(config: DictConfig):
    with torch.no_grad():
        df_test = load_data(config.data.test_path, data_dir=config.data.data_dir, size=config.data.size)
        df_test_clean = clean_data(df_test, is_train=False)
        sequences, vocabulary = tokenize_actions(df_test_clean, is_train=False)


if __name__ == "__main__":
    main()