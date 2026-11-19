#!/usr/bin/env python3
"""Extract gallery PNG thumbnails from the 16 executed chapter notebooks into docs/images/."""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_ROOT = ROOT / "notebooks"
IMG_DIR = ROOT / "docs" / "images"


def extract_notebook_images(nb_path: Path) -> list[bytes]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    images: list[bytes] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "image/png" in data:
                b64 = data["image/png"]
                if isinstance(b64, list):
                    b64 = "".join(b64)
                images.append(base64.b64decode(b64))
    return images


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    nb_files = sorted(NB_ROOT.glob("part*/*.ipynb"))
    for idx, nb_path in enumerate(nb_files, start=1):
        imgs = extract_notebook_images(nb_path)
        if not imgs:
            raise RuntimeError(f"No embedded PNG outputs found in {nb_path}")
        out_path = IMG_DIR / f"gallery_ch{idx:02d}.png"
        out_path.write_bytes(imgs[0])
        if idx == 1:
            (IMG_DIR / "checks_ch01.png").write_bytes(imgs[-1])
        print(f"Wrote {out_path.relative_to(ROOT)} ({len(imgs[0])} bytes) from {nb_path.name}")


if __name__ == "__main__":
    main()
