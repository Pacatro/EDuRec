import tempfile
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import requests
from tqdm import tqdm


def ensure_files(
    folder: Path,
    required: Sequence[str],
    fetch: Callable[[], None],
) -> Path:
    """Return ``folder``, running ``fetch`` first when any required file is missing."""
    if not all((folder / name).exists() for name in required):
        fetch()
    return folder


def stream_to_file(url: str, destination: Path, desc: str = "") -> None:
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


def extract_zip(url: str, extract_to: Path, desc: str = "") -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        archive = Path(folder) / "download.zip"
        stream_to_file(url, archive, desc=desc)
        with zipfile.ZipFile(archive) as z:
            z.extractall(extract_to)
