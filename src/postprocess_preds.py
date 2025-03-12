import albumentations
import cv2
import torch
import torchvision.transforms
import tqdm
from histomark_lib.postprocess_interface import Postprocess
from histoprocess import AnnotationType
from histoprocess._domain.model.grid import Grid
from histoprocess._domain.model.patch import Patch
from histoprocess.feature import Feature
from histoprocess.transforms.mask_transormer import MaskTransformer
from torchvision.transforms.functional import InterpolationMode


class SegmentationPostprocess(Postprocess):

    def __init__(self, device: str = "cpu"):
        self.device = device

    def execute(self, predictions, prediction_object: Grid, **kwargs):
        feature = Feature.init()
        is_postprocess_mask = kwargs.get("is_postprocess_mask", False)
        patches = [
            Patch(
                tile=None,
                mask=None,
                location=pos.location,
                size=pos.size if pos.size is not None else prediction_object.patch_size,
            )
            for pos in prediction_object.patch_positions
        ]
        for pred, patch in tqdm.tqdm(
            zip(predictions, patches),
            desc="Segmentation postprocess",
            total=len(predictions),
        ):
            pred = (
                self.cuda_resize(pred, patch)
                if self.device != "cpu"
                else self.cpu_resize(pred, patch)
            )
            pred = MaskTransformer(
                merge_classes={
                    AnnotationType.CLEAN_VESSEL: 1,
                    AnnotationType.C_INVASION_VESSEL: 2,
                }
            ).inverse_transform(pred)
            patch.mask = feature.correct_mask(pred) if is_postprocess_mask else pred
        return patches

    def cuda_resize(self, pred, patch):
        pred = pred.unsqueeze(0)
        pred = torchvision.transforms.Resize(
            [patch.size.height, patch.size.width],
            interpolation=InterpolationMode.NEAREST,
        )(pred)
        pred = pred.squeeze()
        return (pred.to(torch.uint8)).to("cpu", non_blocking=True).numpy()

    def cpu_resize(self, pred, patch):
        pred = (pred.to(torch.uint8)).to("cpu", non_blocking=True).numpy()
        pred = pred.squeeze(0) if len(pred.shape) == 3 else pred
        resize = albumentations.Resize(
            width=patch.size.width,
            height=patch.size.height,
            interpolation=cv2.INTER_NEAREST,
        )
        return resize(image=pred)["image"]



class InvasionSegmentationPostprocess(Postprocess):

    def __init__(self, device: str = "cpu"):
        self.device = device

    def execute(self, predictions, prediction_object: Grid, **kwargs):
        feature = Feature.init()
        is_postprocess_mask = kwargs.get("is_postprocess_mask", False)
        patches = [
            Patch(
                tile=None,
                mask=None,
                location=pos.location,
                size=pos.size if pos.size is not None else prediction_object.patch_size,
            )
            for pos in prediction_object.patch_positions
        ]
        for pred, patch in tqdm.tqdm(
            zip(predictions, patches),
            desc="Segmentation postprocess",
            total=len(predictions),
        ):
            pred = (
                self.cuda_resize(pred, patch)
                if self.device != "cpu"
                else self.cpu_resize(pred, patch)
            )
            pred = MaskTransformer(
                merge_classes={
                    AnnotationType.INVASION: 1,
                    AnnotationType.CLEAN_VESSEL: 100,
                }
            ).inverse_transform(pred)
            patch.mask = feature.correct_mask(pred) if is_postprocess_mask else pred
        return patches

    def cuda_resize(self, pred, patch):
        pred = pred.unsqueeze(0)
        pred = torchvision.transforms.Resize(
            [patch.size.height, patch.size.width],
            interpolation=InterpolationMode.NEAREST,
        )(pred)
        pred = pred.squeeze()
        return (pred.to(torch.uint8)).to("cpu", non_blocking=True).numpy()

    def cpu_resize(self, pred, patch):
        pred = (pred.to(torch.uint8)).to("cpu", non_blocking=True).numpy()
        pred = pred.squeeze(0) if len(pred.shape) == 3 else pred
        resize = albumentations.Resize(
            width=patch.size.width,
            height=patch.size.height,
            interpolation=cv2.INTER_NEAREST,
        )
        return resize(image=pred)["image"]
