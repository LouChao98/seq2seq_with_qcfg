import dotenv
import hydra
from omegaconf import DictConfig
import sys

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="train.yaml", version_base="1.1")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.training_pipeline import train

    # # Get and apply hyperparameters from wandb
    # with open_dict(config):
    #     for key, value in sweep_config.items():
    #         cursur = config
    #         parts = key.split('.')
    #         for part in parts[:-1]:
    #             cursur = cursur[part]
    #         cursur[parts[-1]] = value

    # Applies optional utilities
    utils.extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    # hack sys.argv
    sys.argv = [item.lstrip('--') for item in sys.argv]
    main()
