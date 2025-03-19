import shutil
from io import BytesIO
from pathlib import Path
import cv2
import hydra
import uvicorn
import asyncio
from PIL import Image
from fastapi import FastAPI, UploadFile, File, APIRouter
from omegaconf import DictConfig
from starlette.middleware.cors import CORSMiddleware
from histomark_lib.torch_lib.models import Model
from histoprocess.feature import Feature
from histoprocess.filters import PolygonAreaFilter, NormalizedPolygonAreaFilter
from src.model.pydentic_models import PydenticPolygons
from src.predictors.image_predictor import ImagePredictor
from src.predictors.slide_predictor import SlidePredictor
from src.utils import download_file
from nc_py_api import Nextcloud


class NovispMlService:
    def __init__(
        self,
        vessels_model: Model,
        invasion_model: Model,
        prediction_device,
        logs_dir: str,
    ):
        self.logs_dir = Path(logs_dir)
        self.vessels_model = vessels_model
        self.invasion_model = invasion_model
        self.prediction_device = prediction_device
        self.app = self._create_app()
        self.nextcloud = Nextcloud(
            nextcloud_url="https://nextcloud.novisp.ru",
            nc_auth_user="zemnuhov2405",
            nc_auth_pass="6S6ka-NY82b-itCpA-9Ae3J-ndEHc",
            timeout=10000,
        )
        self.nextcloud_slides = self._get_nextcloud_slides()

    def _get_nextcloud_slides(self):
        files = self.nextcloud.files.find(
            req=["or", "like", "name", "%.tif", "like", "name", "%.svs"],
            path="/ML_DATA/histology/lung",
        )
        slides = {}
        for file in files:
            slides[Path(file.full_path).stem] = file
        return slides

    def _create_app(self) -> FastAPI:
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.include_router(self._init_router())
        return app

    def _init_router(self) -> APIRouter:
        router = APIRouter()
        router.add_api_route("/predict/image", self.image_predict, methods=["POST"])
        router.add_api_route(
            "/predict/nextcloud_file", self.handle_nextcloud_file, methods=["POST"]
        )
        router.add_api_route(
            "/predict/direct_link", self.handle_direct_link, methods=["POST"]
        )
        return router

    async def handle_nextcloud_file(self, file: str):
        if file in self.nextcloud_slides.keys():
            current_file = self.nextcloud_slides[file]
            wsi_path = self.logs_dir / "slides" / file / Path(current_file.full_path).name
            if not wsi_path.parent.exists():
                wsi_path.parent.mkdir(exist_ok=True, parents=True)
                with open(str(wsi_path), "wb") as file:
                    self.nextcloud.files.download2stream(
                        current_file,
                        file,
                    )
            return await self._predict_slide(wsi_path)
        return {"error": f"Файл с именем '{file}' не найден!"}

    async def _predict_slide(self, wsi_path: Path):
        slide_name = Path(wsi_path).stem
        geojson_path = wsi_path.parent / f"{slide_name}.geojson"

        if geojson_path.exists():
            polygons = await asyncio.to_thread(
                Feature.init().get_polygons_from_wsa, str(geojson_path)
            )
        else:
            slide_predictor = SlidePredictor(
                self.vessels_model, self.invasion_model, self.prediction_device
            )
            polygons = await asyncio.to_thread(slide_predictor.predict, wsi_path)
            polygons = await asyncio.to_thread(
                Feature.init().filter_polygons_by_area,
                polygons,
                NormalizedPolygonAreaFilter(
                    tissue_polygon=Feature.init().get_tissue_polygon(str(wsi_path))
                ),
            )

        return PydenticPolygons.form_polygon(polygons)

    async def handle_direct_link(self, file: str):
        slide_name = Path(file).stem
        wsi_path = await asyncio.to_thread(
            download_file, file, self.logs_dir / "slides" / slide_name
        )
        return await self._predict_slide(wsi_path)

    async def image_predict(self, file: UploadFile = File(...)):
        photo_path = await self._save_photo(file)
        photo = cv2.cvtColor(cv2.imread(str(photo_path)), cv2.COLOR_BGR2RGB)
        loop = asyncio.get_running_loop()
        image_predictor = ImagePredictor(
            self.vessels_model, self.invasion_model, self.prediction_device
        )
        polygons = await loop.run_in_executor(
            None, image_predictor.predict, photo_path
        )

        polygons = await loop.run_in_executor(
            None,
            Feature.init().filter_polygons_by_area,
            polygons,
            PolygonAreaFilter(area_min=(photo.shape[0] * photo.shape[1]) * 0.0005),
        )
        shutil.rmtree(photo_path.parent)

        return polygons



    async def _save_photo(self, uploaded_file: UploadFile) -> Path:
        image_data = await uploaded_file.read()
        name = Path(uploaded_file.filename)
        file_path = self.logs_dir / "image" / name.stem / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(BytesIO(image_data)) as image:
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
