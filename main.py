import multiprocessing
multiprocessing.set_start_method("forkserver", force=True)

from io import BytesIO
from pathlib import Path

import cv2
import hydra
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, APIRouter
from histomark_lib.torch_lib.models import Model
from histoprocess import InjectionSetting
from histoprocess.feature import Feature
from histoprocess.filters import PolygonAreaFilter, NormalizedPolygonAreaFilter
from omegaconf import DictConfig
from starlette.middleware.cors import CORSMiddleware

from src.model.pydentic_models import PydenticPolygons
from src.predictors.image_predictor import ImagePredictor
from src.predictors.slide_predictor import SlidePredictor
from src.utils import download_file


class NovispMlService:

    def __init__(
        self,
        vessels_model: Model,
        invasion_model: Model,
        prediction_device,
        logs_dir: str,
    ):
        self.app = FastAPI()
        self.app.include_router(self.init_router())
        self.logs_dir = Path(logs_dir)
        origins = ["*"]
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.image_predictor = ImagePredictor(
            vessels_model=vessels_model,
            invasion_model=invasion_model,
            device=prediction_device,
        )
        self.slide_predictor = SlidePredictor(
            vessels_model=vessels_model,
            invasion_model=invasion_model,
            device=prediction_device,
        )

    def init_router(self) -> APIRouter:
        router = APIRouter()
        router.add_api_route("/predict/image", self.image_predict, methods=["POST"])
        router.add_api_route(
            "/predict/nextcloud_link", self.handle_nextcloud_link, methods=["POST"]
        )
        return router

    def handle_nextcloud_link(self, file_url: str):
        InjectionSetting().set_backend("cucim")
        slide_name = Path(file_url).stem
        wsi_path = download_file(url=file_url, path=self.logs_dir / slide_name)
        if not (wsi_path.parent / f"{slide_name}.geojson").exists():
            polygons = self.slide_predictor.predict(wsi_path)
            polygons = Feature.init().filter_polygons_by_area(
                polygons=polygons,
                area_filter=NormalizedPolygonAreaFilter(
                    tissue_polygon=Feature.init().get_tissue_polygon(
                        wsi_path=str(wsi_path)
                    )
                ),
            )
        else:
            polygons = Feature.init().get_polygons_from_wsa(
                str(wsi_path.parent / f"{slide_name}.geojson")
            )

        return PydenticPolygons.form_polygon(polygons)

    async def image_predict(self, file: UploadFile = File(...)):
        photo_path = await self.save_photo(file)
        photo = cv2.cvtColor(cv2.imread(str(photo_path)), cv2.COLOR_BGR2RGB)
        polygons = self.image_predictor.predict(photo_path)
        polygons = Feature.init().filter_polygons_by_area(
            polygons=polygons,
            area_filter=PolygonAreaFilter(
                area_min=(photo.shape[0] * photo.shape[1]) * 0.0005
            ),
        )
        return PydenticPolygons.form_polygon(polygons)

    async def save_photo(self, uploaded_file: UploadFile) -> Path:
        image, name = await uploaded_file.read(), uploaded_file.filename
        name = Path(name)
        file_path = self.logs_dir / name.stem / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(BytesIO(image))
        image.save(file_path)
        return file_path


@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig):
    uvicorn.run(
        app=cfg.uvicorn.app,
        port=int(cfg.uvicorn.port),
        reload=cfg.uvicorn.reload,
        host=cfg.uvicorn.host,
    )


if __name__ == "__main__":
    main()
