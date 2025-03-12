from pathlib import Path

import albumentations
import torch
from albumentations import ToTensorV2
from histoprocess import InjectionSetting
from histoprocess._domain.model.polygon import Polygons
from histoprocess.feature import Feature
from histoprocess.filters import NormalizedPolygonAreaFilter

from src.data.slide_data_builder import SlideDataBuilder, GridType
from src.postprocess_preds import SegmentationPostprocess, InvasionSegmentationPostprocess
from src.predictors.predictor import Predictor
from histomark_lib.torch_lib.models import Model
from histomark_lib.torch_lib.prediction import torch_predict


class SlidePredictor(Predictor):

    def __init__(
        self, vessels_model: Model, invasion_model: Model, device: str = "cpu"
    ):
        self.vessels_model = vessels_model
        self.invasion_model = invasion_model
        self.device = device
        self.segmentation_transforms = [
            albumentations.Resize(width=512, height=512),
            albumentations.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(transpose_mask=True),
        ]
        self.invasion_transforms = [
            albumentations.Resize(width=256, height=256),
            albumentations.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(transpose_mask=True),
        ]

    def vessels_segmentation(self, wsi_path: Path):
        feature = Feature.init()
        path = wsi_path.parent
        data_builder = SlideDataBuilder(
            wsi_path=wsi_path,
            grid_type=GridType.FULL,
            transforms=self.segmentation_transforms,
            is_fill_without_tissue=True
        )
        vessels_prediction = torch_predict(
            model=self.vessels_model,
            data=data_builder.build(),
            accelerator=self.device
        )
        torch.cuda.empty_cache()
        segmentation_predict = SegmentationPostprocess().execute(
            vessels_prediction, data_builder.grid
        )
        polygons = feature.get_polygons_from_patches(segmentation_predict, data_builder.grid)
        tissue_polygon = feature.get_tissue_polygon(str(wsi_path))
        polygons = (
            feature.filter_polygons_by_area(
                polygons=polygons,
                area_filter=NormalizedPolygonAreaFilter(
                    tissue_polygon=tissue_polygon, area_min=0.0005
                ),
            )
            if len(polygons.value) > 1
            else polygons
        )
        feature.polygons_to_geojson(
            polygons=polygons,
            save_path=str(path),
            file_name="annotation",
        )
        data_builder = SlideDataBuilder(
            wsi_path=wsi_path,
            wsa_path=path/"annotation.geojson",
            grid_type=GridType.BBOX,
            transforms=self.segmentation_transforms,
            is_fill_without_tissue=True
        )
        vessels_prediction = torch_predict(
            model=self.vessels_model,
            data=data_builder.build(),
            accelerator=self.device
        )
        torch.cuda.empty_cache()
        segmentation_predict = SegmentationPostprocess().execute(
            vessels_prediction, data_builder.grid
        )
        polygons = feature.get_polygons_from_patches(segmentation_predict, data_builder.grid)
        tissue_polygon = feature.get_tissue_polygon(str(wsi_path))
        polygons = (
            feature.filter_polygons_by_area(
                polygons=polygons,
                area_filter=NormalizedPolygonAreaFilter(
                    tissue_polygon=tissue_polygon, area_min=0.0005
                ),
            )
            if len(polygons.value) > 1
            else polygons
        )
        feature.polygons_to_geojson(
            polygons=polygons,
            save_path=str(path),
            file_name="bbox_annotation",
        )
        return path/"bbox_annotation.geojson"

    def invasion_segmentation(self, wsi_path: Path):
        path = wsi_path.parent
        data_builder = SlideDataBuilder(
            wsi_path=wsi_path,
            wsa_path=path / "bbox_annotation.geojson",
            grid_type=GridType.ANNOTATION,
            transforms=self.invasion_transforms,
            is_fill_without_mask=True
        )
        invasion_prediction = torch_predict(
            model=self.invasion_model,
            data=data_builder.build(),
            accelerator=self.device,
            binary_threshold = 0.995
        )
        invasion_predict = InvasionSegmentationPostprocess().execute(
            invasion_prediction, data_builder.grid
        )
        Feature.init().patches_to_geojson(
            invasion_predict,
            data_builder.grid,
            str(path),
            file_name="invasion",
        )
        return path/"invasion.geojson"

    def predict(self, wsi_path: Path) -> Polygons:
        path = wsi_path.parent
        vessels_wsa_path = self.vessels_segmentation(wsi_path)
        invasion_wsa_path = self.invasion_segmentation(wsi_path)

        Feature.init().combine_prediction_results(
            segmentation_geojson_path=str(vessels_wsa_path),
            classification_geojson_path=str(invasion_wsa_path),
            classification_annotation_type=["Invasion"],
            file_name=wsi_path.stem,
            save_path=str(path),
        )
        poly = Feature.init().get_polygons_from_wsa(
            str(path / f"{wsi_path.stem}.geojson")
        )
        return poly


