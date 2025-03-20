from pathlib import Path
import re
import cv2
from histoprocess import AnnotationType
from histoprocess._domain.model.polygon import Polygons
import numpy as np
import requests
import tqdm
import gdown
import html


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


def extract_file_id(google_drive_url):
    """Извлекает file_id из различных форматов ссылок"""
    patterns = [
        r"id=([a-zA-Z0-9_-]+)",  # Для ссылок вида https://drive.google.com/uc?export=download&id=...
        r"/d/([a-zA-Z0-9_-]+)",  # Для ссылок вида https://drive.google.com/file/d/...
    ]
    for pattern in patterns:
        match = re.search(pattern, google_drive_url)
        if match:
            return match.group(1)
    return None


def get_filename_from_content(response_content):
    """Извлекает имя файла из HTML-ответа Google Drive"""
    match = re.search(r"<a href=.*?>([^<]+)</a>", response_content.decode("utf-8"))
    if match:
        return html.unescape(
            match.group(1)
        )  # Декодируем HTML-сущности (например, `&amp;` → `&`)
    return None


def get_filename_from_shared_link(shared_link):
    """Получает имя файла по shared link без API"""
    file_id = extract_file_id(shared_link)
    if not file_id:
        print("Ошибка: Не удалось извлечь file_id.")
        return None

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        response = requests.get(download_url, stream=True, allow_redirects=True)
        file_name = get_filename_from_content(response.content)
        return file_name if file_name else None
    except requests.RequestException as e:
        print(f"Ошибка при получении данных: {e}")
        return None


def get_filename_from_id(file_id):
    """Получает имя файла по file_id"""
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        response = requests.get(download_url, stream=True, allow_redirects=True)
        file_name = get_filename_from_content(response.content)
        return file_name if file_name else None
    except requests.RequestException as e:
        print(f"Ошибка при получении данных: {e}")
        return None


def download_google_drive_file(shared_link, path: Path):
    """Скачивает файл с правильным именем"""
    file_id = extract_file_id(shared_link)
    if not file_id:
        print("Ошибка: file_id не найден.")
        return
    file_name = get_filename_from_shared_link(shared_link)
    file_path = path / Path(file_name).stem / file_name
    output_path = gdown.download(
        f"https://drive.google.com/uc?id={file_id}", str(file_path), quiet=False, fuzzy=True
    )
    return Path(output_path)
