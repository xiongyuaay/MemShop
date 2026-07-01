from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from retail_thor.fbx_shelf_scene import render_fbx_store_shelf_scene


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHELF_FBX = ROOT.parent / "store-shelves" / "source" / "grocery_shelf.fbx"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a standalone FBX store-shelf screenshot with AI2-THOR common-object products."
    )
    parser.add_argument("--shelf-fbx", default=str(DEFAULT_SHELF_FBX))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "fbx_store_shelf_scene"))
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--product-count", type=int, default=40)
    parser.add_argument("--selected-module", default="first", choices=["first", "all"])
    args = parser.parse_args()

    manifest = render_fbx_store_shelf_scene(
        output_dir=Path(args.output_dir),
        shelf_fbx=Path(args.shelf_fbx),
        width=args.width,
        height=args.height,
        product_count=args.product_count,
        selected_module=args.selected_module,
    )
    print(f"generated standalone FBX shelf screenshots: {len(manifest['screenshots'])}")
    print(f"output directory: {Path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
