from pathlib import Path
import re
import cv2
from histoprocess import AnnotationType
from histoprocess._domain.model.polygon import Polygons
import numpy as np
import requests
import tqdm


def draw_contours(annotations: Polygons, image: np.ndarray):
    image_copy = image.copy()
    image_copy2 = image.copy()
    for polygon in annotations.value:
        coord = tuple(np.array([polygon.coordinates]).astype(np.int32))
        cv2.drawContours(
            image=image_copy,
            contours=coord,
            contourIdx=-1,
            color=(
                (0, 255, 0)
                if polygon.annotation_type == AnnotationType.CLEAN_VESSEL
                else (255, 0, 0)
            ),
            thickness=cv2.FILLED,
            lineType=cv2.LINE_4,
        )
        cv2.drawContours(
            image=image_copy2,
            contours=coord,
            contourIdx=-1,
            color=(
                (0, 255, 0)
                if polygon.annotation_type == AnnotationType.CLEAN_VESSEL
                else (255, 0, 0)
            ),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    image_copy = cv2.addWeighted(image_copy2, 0.9, image_copy, 0.2, 0.0)
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    return image_copy


def download_file(url: str, path: Path) -> Path:
    response = requests.get(url, stream=True)
    if "Content-Disposition" in response.headers:
        file_name = re.findall(
            'filename="(.+)"', response.headers["Content-Disposition"]
        )[0]
    else:
        file_name = url.split("/")[-1]
    total = int(response.headers.get("content-length", 0))
    save_path = path / file_name
    if not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
    if not save_path.exists():
        with open(save_path, "wb") as file, tqdm.tqdm(
            desc=file_name,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    return save_path
