import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://lfs.aminer.cn/misc/moocdata/data/mooccube2"

MINIMAL_FILES = (
    "entities/user.json",
    "entities/course.json",
)

OPTIONAL_METADATA_FILES = (
    "entities/school.json",
    "entities/teacher.json",
    "relations/course-field.json",
    "relations/course-school.txt",
    "relations/course-teacher.txt",
)


def download_file(url: str, destination: Path, desc: str) -> None:
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get("Content-Length", 0))

        with (
            open(destination, "wb") as out_file,
            tqdm(
                desc=desc,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
            ) as pbar,
        ):
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga simplificada de los documentos de MOOCCubeX."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/mooccubex"),
        help="Directorio de destino.",
    )
    parser.add_argument(
        "--with-metadata",
        action="store_true",
        help="Descarga metadatos adicionales (profesores, escuelas, relaciones).",
    )
    args = parser.parse_args()

    files = list(MINIMAL_FILES)
    if args.with_metadata:
        files.extend(OPTIONAL_METADATA_FILES)

    for relative_path in files:
        url = f"{BASE_URL}/{relative_path}"
        destination = args.output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        download_file(url, destination, relative_path)

    print("\nDescarga completada con éxito.")


if __name__ == "__main__":
    main()
