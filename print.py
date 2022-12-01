from pprint import pprint

import dotenv
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import src

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):
    print(src)
    config = OmegaConf.to_container(config, resolve=True)
    pprint(config)

    print(hydra)
    config = OmegaConf.to_container(HydraConfig.get(), resolve=True)
    pprint(config)
    exit(0)


if __name__ == "__main__":
    main()
