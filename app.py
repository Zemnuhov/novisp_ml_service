import hydra
from omegaconf import DictConfig

from main import NovispMlService


app = None


def init_service(cfg: DictConfig):
    return NovispMlService(
        vessels_model=hydra.utils.instantiate(cfg.models.segmentation),
        invasion_model=hydra.utils.instantiate(cfg.models.invasion),
        prediction_device=cfg.setting.prediction_device,
        logs_dir=cfg.setting.file_path,
    )


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    global app
    app = init_service(cfg).app
    return app


main()
