from pathlib import Path
from histomark_lib.torch_lib.models import Model
from histomark_lib.torch_lib.prediction import torch_predict
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from histoprocess._presentation.feature import Feature
from src.predictors.predictor import Predictor
from src.data.image_data_builder import ImageDataBuilder, GridType
from src.postprocess_preds import (
    SegmentationPostprocess,
    InvasionSegmentationPostprocess,
)


class ImagePredictor(Predictor):

    def __init__(
        self, vessels_model: Model, invasion_model: Model, device: str = "cpu"
    ):
        self.vessels_model = vessels_model
        self.invasion_model = invasion_model
        self.device = device
        self.segmentation_transforms = [
            albumentations.Resize(width=512, height=512),
            albumentations.Blur((3, 3), p=1),
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

    def predict(self, image: Path):
        img_path = image
        segmentation_data_builder = ImageDataBuilder(
            img_path,
            grid_type=GridType.FULL,
            transforms=self.segmentation_transforms,
        )
        segmentation_predict = torch_predict(
            model=self.vessels_model,
            data=segmentation_data_builder.build(),
            accelerator=self.device,
            binary_threshold=0.5,
        )
        segmentation_predict = SegmentationPostprocess().execute(
            segmentation_predict, segmentation_data_builder.grid
        )
        Feature.init().patches_to_geojson(
            segmentation_predict, segmentation_data_builder.grid, str(img_path.parent)
        )
        invasion_data_builder = ImageDataBuilder(
            img_path,
            grid_type=GridType.ANNOTATION,
            transforms=self.invasion_transforms,
            wsa_path=str(img_path.parent / "annotation.geojson"),
            is_fill_without_mask=True,
        )
        if len(invasion_data_builder) > 0:
            invasion_predict = torch_predict(
                model=self.invasion_model,
                data=invasion_data_builder.build(),
                accelerator=self.device,
                binary_threshold=0.995,
            )
            invasion_predict = InvasionSegmentationPostprocess().execute(
                invasion_predict, invasion_data_builder.grid
            )
            Feature.init().patches_to_geojson(
                invasion_predict,
                invasion_data_builder.grid,
                str(img_path.parent),
                file_name="invasion",
            )
            Feature.init().combine_prediction_results(
                segmentation_geojson_path=str(img_path.parent / "annotation.geojson"),
                classification_geojson_path=str(img_path.parent / "invasion.geojson"),
                classification_annotation_type=["Invasion"],
                save_path=str(img_path.parent),
            )
            poly = Feature.init().get_polygons_from_wsa(
                str(img_path.parent / "combine_prediction.geojson")
            )
        else:
            poly = Feature.init().get_polygons_from_wsa(
                str(img_path.parent / "annotation.geojson")
            )
        return poly
