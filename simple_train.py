import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

# import torch; torch.set_anomaly_enabled(True)
# import torch; torch.use_deterministic_algorithms(True)
# import torch; torch.backends.cudnn.benchmark = False


@hydra.main(config_path="configs/", config_name="train.yaml", version_base="1.1")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.simple_training_pipeline import train

    # Applies optional utilities
    # utils.extras(config)
    # Train model
    return train(config)


if __name__ == "__main__":
    main()
