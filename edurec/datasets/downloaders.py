import io
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path

import kagglehub
import requests
from tqdm import tqdm

from .. import settings
from .loaders import DatasetName

type DownloadFn = Callable[[], Path]

dataset_downloaders: dict[DatasetName, DownloadFn] = {}


def register_dataset(ds_name: DatasetName) -> Callable[[DownloadFn], DownloadFn]:
    def decorator(fn: DownloadFn) -> DownloadFn:
        dataset_downloaders[ds_name] = fn
        return fn

    return decorator


def _download_file(url: str, destination: Path, desc: str = "") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        with (
            open(destination, "wb") as out,
            tqdm(
                desc=desc or destination.name,
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                out.write(chunk)
                pbar.update(len(chunk))


def _download_and_extract_zip(url: str, extract_to: Path, desc: str = "") -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        buf = io.BytesIO()
        with tqdm(
            desc=desc,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                buf.write(chunk)
                pbar.update(len(chunk))
        buf.seek(0)
        with zipfile.ZipFile(buf) as z:
            z.extractall(extract_to)


def _download_mars() -> Path:
    dest = settings.RAW_DATA_FOLDER / "mars"
    if all((dest / f).exists() for f in settings.MARS_REQUIRED_FILES):
        return dest
    _download_and_extract_zip(settings.MARS_ZIP_URL, dest, desc="MARS dataset (ZIP)")
    return dest


@register_dataset(DatasetName.EXPLICIT_MARS)
def download_explicit_mars() -> Path:
    return _download_mars()


@register_dataset(DatasetName.IMPLICIT_MARS)
def download_implicit_mars() -> Path:
    return _download_mars()


@register_dataset(DatasetName.ITM)
def download_itm() -> Path:
    dest = settings.RAW_DATA_FOLDER / "itm"
    if all((dest / f).exists() for f in settings.ITM_REQUIRED_FILES):
        return dest

    src = Path(kagglehub.dataset_download(settings.KAGGLE_ITM_DATASET))
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name in settings.ITM_REQUIRED_FILES:
            shutil.copy2(f, dest / f.name)
    return dest


@register_dataset(DatasetName.DORIS)
def download_doris() -> Path:
    dest = settings.RAW_DATA_FOLDER / "doris"
    if all((dest / f).exists() for f in settings.DORIS_REQUIRED_FILES):
        return dest

    _download_and_extract_zip(settings.DORIS_ZIP_URL, dest, desc="DORIS dataset (ZIP)")

    subdir = dest / "datasetInEnglish"
    if subdir.is_dir():
        for f in subdir.iterdir():
            shutil.move(str(f), dest / f.name)
        shutil.rmtree(subdir)

    macosx = dest / "__MACOSX"
    if macosx.is_dir():
        shutil.rmtree(macosx)

    return dest


@register_dataset(DatasetName.MOOCCUBEX)
def download_mooccubex() -> Path:
    dest = settings.RAW_DATA_FOLDER / "mooccubex"
    if all((dest / f).exists() for f in settings.MOOCCUBEX_REQUIRED_FILES):
        return dest

    for relative_path in settings.MOOCCUBEX_REQUIRED_FILES:
        url = f"{settings.MOOCCUBEX_BASE_URL}/{relative_path}"
        destination = dest / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        _download_file(url, destination, desc=relative_path)

    return dest


def download_raw_data(dataset_name: DatasetName) -> Path:
    downloader = dataset_downloaders.get(dataset_name)
    if downloader is None:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return downloader()
