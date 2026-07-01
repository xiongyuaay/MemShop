from __future__ import annotations

import array
import json
import math
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


EXCLUDED_OBJECT_TYPES = {"Statue"}

AI2THOR_COMMON_PRODUCTS: list[dict[str, Any]] = [
    {"object_type": "Bottle", "shape": "cylinder", "color": (49, 111, 173), "size": (7.0, 7.0, 18.0)},
    {"object_type": "SoapBottle", "shape": "box", "color": (45, 151, 137), "size": (7.5, 7.5, 16.0)},
    {"object_type": "Cup", "shape": "cup", "color": (232, 232, 224), "size": (7.0, 7.0, 10.0)},
    {"object_type": "Mug", "shape": "mug", "color": (190, 54, 62), "size": (8.0, 8.0, 10.0)},
    {"object_type": "Bowl", "shape": "bowl", "color": (229, 185, 64), "size": (9.5, 9.5, 6.0)},
    {"object_type": "Plate", "shape": "plate", "color": (218, 221, 218), "size": (10.0, 10.0, 2.0)},
    {"object_type": "Book", "shape": "box", "color": (57, 85, 136), "size": (5.0, 11.0, 14.0)},
    {"object_type": "Apple", "shape": "sphere", "color": (196, 42, 52), "size": (8.0, 8.0, 8.0)},
    {"object_type": "Tomato", "shape": "sphere", "color": (212, 64, 45), "size": (8.0, 8.0, 7.0)},
    {"object_type": "Potato", "shape": "ellipsoid", "color": (163, 119, 76), "size": (9.0, 7.0, 6.0)},
    {"object_type": "Egg", "shape": "ellipsoid", "color": (236, 228, 204), "size": (5.5, 5.5, 7.5)},
    {"object_type": "Bread", "shape": "ellipsoid", "color": (194, 128, 63), "size": (11.0, 6.5, 6.5)},
    {"object_type": "SaltShaker", "shape": "cylinder", "color": (205, 213, 216), "size": (5.0, 5.0, 11.0)},
    {"object_type": "Vase", "shape": "vase", "color": (88, 130, 108), "size": (8.0, 8.0, 15.0)},
]


@dataclass(frozen=True)
class FbxNode:
    name: str
    properties: list[Any]
    children: list["FbxNode"]


@dataclass
class ShelfMesh:
    model_name: str
    root_model_name: str
    vertices: np.ndarray
    triangles: list[tuple[int, int, int]]
    color: tuple[int, int, int]


class StoreShelfMeshes(list[ShelfMesh]):
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        vertices = np.vstack([mesh.vertices for mesh in self])
        return vertices.min(axis=0), vertices.max(axis=0)


@dataclass(frozen=True)
class RenderFace:
    vertices: np.ndarray
    color: tuple[int, int, int]
    kind: str = "mesh"


def build_fbx_shelf_scene_spec(
    shelf_fbx: Path | str,
    output_dir: Path | str,
    product_count: int = 40,
    selected_module: str = "first",
) -> dict[str, Any]:
    shelf_path = Path(shelf_fbx).expanduser().resolve()
    products = _product_specs(product_count)
    return {
        "source": "fbx_store_shelf_scene",
        "scene_mode": "standalone_asset",
        "renderer": "python_software_renderer",
        "shelf_asset": {
            "path": str(shelf_path),
            "selected_module": selected_module,
            "textures_dir": str(shelf_path.parents[1] / "textures"),
        },
        "environment": {
            "room": None,
            "floor": "simple matte ground plane",
            "background": "plain studio background",
        },
        "product_source": {
            "type": "ai2thor_common_object_categories",
            "mesh_note": (
                "Products use AI2-THOR common object categories represented by simple "
                "Python-generated meshes; this renderer does not extract Unity asset meshes."
            ),
        },
        "excluded_object_types": sorted(EXCLUDED_OBJECT_TYPES),
        "products": products,
        "output_dir": str(Path(output_dir).expanduser().resolve()),
    }


def load_store_shelf_meshes(shelf_fbx: Path | str, selected_module: str = "first") -> StoreShelfMeshes:
    path = Path(shelf_fbx)
    nodes = _read_fbx(path)
    geometries: dict[int, dict[str, Any]] = {}
    models: dict[int, dict[str, Any]] = {}
    connections: list[tuple[int, int]] = []
    top_level_models: list[int] = []

    for node in nodes:
        if node.name == "Objects":
            for child in node.children:
                if child.name == "Geometry" and len(child.properties) >= 3:
                    geometry_id = int(child.properties[0])
                    vertices = _child_array(child, "Vertices")
                    polygon_indices = _child_array(child, "PolygonVertexIndex")
                    if vertices is not None and polygon_indices is not None:
                        geometries[geometry_id] = {
                            "vertices": np.asarray(vertices, dtype=float).reshape((-1, 3)),
                            "triangles": _triangulate_polygon_indices(polygon_indices),
                        }
                elif child.name == "Model" and len(child.properties) >= 3:
                    model_id = int(child.properties[0])
                    models[model_id] = {
                        "name": _clean_fbx_name(str(child.properties[1])),
                        "translation": np.zeros(3),
                        "geometric_translation": np.zeros(3),
                    }
                    for prop_node in _children_named(child, "Properties70"):
                        for prop in prop_node.children:
                            if not prop.properties:
                                continue
                            prop_name = prop.properties[0]
                            if prop_name == "Lcl Translation":
                                models[model_id]["translation"] = np.asarray(prop.properties[4:7], dtype=float)
                            elif prop_name == "GeometricTranslation":
                                models[model_id]["geometric_translation"] = np.asarray(prop.properties[4:7], dtype=float)
        elif node.name == "Connections":
            for child in node.children:
                if child.name == "C" and child.properties[:1] == ["OO"]:
                    connections.append((int(child.properties[1]), int(child.properties[2])))

    parent_by_child = {child: parent for child, parent in connections if parent != 0}
    geometry_model = {child: parent for child, parent in connections if child in geometries and parent in models}
    for child, parent in connections:
        if parent == 0 and child in models:
            top_level_models.append(child)

    @lru_cache(maxsize=None)
    def world_translation(model_id: int) -> np.ndarray:
        parent = parent_by_child.get(model_id)
        parent_translation = world_translation(parent) if parent in models else np.zeros(3)
        return parent_translation + models[model_id]["translation"]

    @lru_cache(maxsize=None)
    def root_model_name(model_id: int) -> str:
        current = model_id
        while parent_by_child.get(current) in models:
            current = parent_by_child[current]
        return models[current]["name"]

    if selected_module == "first":
        allowed_roots = {_first_shelf_root_name(top_level_models, models)}
    elif selected_module == "all":
        allowed_roots = {models[model_id]["name"] for model_id in top_level_models}
    else:
        allowed_roots = {selected_module}

    meshes = StoreShelfMeshes()
    for geometry_id, geometry in geometries.items():
        model_id = geometry_model.get(geometry_id)
        if model_id not in models:
            continue
        root_name = root_model_name(model_id)
        if root_name not in allowed_roots:
            continue
        model = models[model_id]
        vertices = geometry["vertices"] + world_translation(model_id) + model["geometric_translation"]
        meshes.append(
            ShelfMesh(
                model_name=model["name"],
                root_model_name=root_name,
                vertices=vertices,
                triangles=geometry["triangles"],
                color=_shelf_color(model["name"]),
            )
        )

    if not meshes:
        raise ValueError(f"No shelf meshes found in {path} for selected_module={selected_module!r}")
    return meshes


def render_fbx_store_shelf_scene(
    output_dir: Path | str,
    shelf_fbx: Path | str,
    width: int = 1200,
    height: int = 800,
    product_count: int = 40,
    selected_module: str = "first",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    spec = build_fbx_shelf_scene_spec(shelf_fbx, output_path, product_count, selected_module)
    meshes = load_store_shelf_meshes(shelf_fbx, selected_module=selected_module)
    min_corner, max_corner = meshes.bounds()

    faces: list[RenderFace] = []
    faces.extend(_floor_faces(min_corner, max_corner))
    for mesh in meshes:
        faces.extend(_mesh_faces(mesh))
    faces.extend(_product_faces(spec["products"], min_corner, max_corner))

    image = _render_faces(faces, min_corner, max_corner, width, height)
    image_name = "fbx_store_shelf_products.png"
    image.save(output_path / image_name)

    manifest = {
        **spec,
        "width": width,
        "height": height,
        "mesh_count": len(meshes),
        "screenshots": [
            {
                "scene_id": "fbx_store_shelf_front",
                "path": image_name,
                "caption": "Standalone store-shelves FBX asset filled with AI2-THOR common-object products.",
            }
        ],
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def render_fbx_store_shelf_image(
    shelf_fbx: Path | str,
    width: int = 1200,
    height: int = 800,
    product_count: int = 40,
    selected_module: str = "first",
    camera_eye: Sequence[float] | None = None,
    camera_target: Sequence[float] | None = None,
) -> Image.Image:
    spec = build_fbx_shelf_scene_spec(shelf_fbx, Path("."), product_count, selected_module)
    meshes = load_store_shelf_meshes(shelf_fbx, selected_module=selected_module)
    min_corner, max_corner = meshes.bounds()

    faces: list[RenderFace] = []
    faces.extend(_floor_faces(min_corner, max_corner))
    for mesh in meshes:
        faces.extend(_mesh_faces(mesh))
    faces.extend(_product_faces(spec["products"], min_corner, max_corner))

    return _render_faces(
        faces,
        min_corner,
        max_corner,
        width,
        height,
        camera_eye=camera_eye,
        camera_target=camera_target,
    )


def _read_fbx(path: Path) -> list[FbxNode]:
    data = path.read_bytes()
    if not data.startswith(b"Kaydara FBX Binary"):
        raise ValueError(f"Only binary FBX files are supported: {path}")
    version = struct.unpack_from("<I", data, 23)[0]
    if version >= 7500:
        raise ValueError(f"FBX version {version} uses 64-bit node headers; this lightweight reader supports < 7500.")
    nodes, _ = _parse_nodes(data, 27, len(data))
    return nodes


def _parse_nodes(data: bytes, offset: int, end_limit: int) -> tuple[list[FbxNode], int]:
    nodes: list[FbxNode] = []
    null_record_size = 13
    while offset < end_limit - null_record_size:
        end_offset, property_count, _property_list_len = struct.unpack_from("<III", data, offset)
        offset += 12
        if end_offset == 0:
            offset += 1
            break
        name_len = data[offset]
        offset += 1
        name = data[offset : offset + name_len].decode("utf-8", "replace")
        offset += name_len
        properties = []
        for _ in range(property_count):
            value, offset = _read_property(data, offset)
            properties.append(value)
        children: list[FbxNode] = []
        if offset < end_offset - null_record_size:
            children, offset = _parse_nodes(data, offset, end_offset)
        offset = end_offset
        nodes.append(FbxNode(name=name, properties=properties, children=children))
    return nodes, offset


def _read_property(data: bytes, offset: int) -> tuple[Any, int]:
    property_type = chr(data[offset])
    offset += 1
    if property_type == "Y":
        return struct.unpack_from("<h", data, offset)[0], offset + 2
    if property_type == "C":
        return bool(data[offset]), offset + 1
    if property_type == "I":
        return struct.unpack_from("<i", data, offset)[0], offset + 4
    if property_type == "F":
        return struct.unpack_from("<f", data, offset)[0], offset + 4
    if property_type == "D":
        return struct.unpack_from("<d", data, offset)[0], offset + 8
    if property_type == "L":
        return struct.unpack_from("<q", data, offset)[0], offset + 8
    if property_type in {"S", "R"}:
        length = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        raw = data[offset : offset + length]
        offset += length
        if property_type == "S":
            return raw.decode("utf-8", "replace"), offset
        return raw, offset
    if property_type in _ARRAY_TYPES:
        length, encoding, compressed_length = struct.unpack_from("<III", data, offset)
        offset += 12
        raw = data[offset : offset + compressed_length]
        offset += compressed_length
        if encoding:
            raw = zlib.decompress(raw)
        typecode, item_size = _ARRAY_TYPES[property_type]
        values = array.array(typecode)
        values.frombytes(raw)
        if values.itemsize != item_size:
            values.byteswap()
        return np.asarray(values), offset
    raise ValueError(f"Unsupported FBX property type {property_type!r}")


_ARRAY_TYPES = {
    "d": ("d", 8),
    "f": ("f", 4),
    "i": ("i", 4),
    "I": ("I", 4),
    "l": ("q", 8),
    "b": ("b", 1),
    "c": ("?", 1),
}


def _children_named(node: FbxNode, name: str) -> Iterable[FbxNode]:
    return (child for child in node.children if child.name == name)


def _child_array(node: FbxNode, name: str) -> np.ndarray | None:
    for child in node.children:
        if child.name == name and child.properties:
            return np.asarray(child.properties[0])
    return None


def _clean_fbx_name(name: str) -> str:
    return name.split("\x00", 1)[0]


def _triangulate_polygon_indices(indices: Sequence[int]) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    polygon: list[int] = []
    for raw_index in indices:
        index = int(raw_index)
        is_last = index < 0
        polygon.append(-index - 1 if is_last else index)
        if is_last:
            if len(polygon) >= 3:
                anchor = polygon[0]
                for i in range(1, len(polygon) - 1):
                    triangles.append((anchor, polygon[i], polygon[i + 1]))
            polygon = []
    return triangles


def _first_shelf_root_name(top_level_models: Sequence[int], models: dict[int, dict[str, Any]]) -> str:
    for model_id in top_level_models:
        name = models[model_id]["name"]
        if name.startswith("grocery_shelf"):
            return name
    if top_level_models:
        return models[top_level_models[0]]["name"]
    raise ValueError("No top-level shelf model found.")


def _shelf_color(model_name: str) -> tuple[int, int, int]:
    if "base" in model_name:
        return (53, 58, 62)
    return (79, 84, 88)


def _product_specs(product_count: int) -> list[dict[str, Any]]:
    available = [product for product in AI2THOR_COMMON_PRODUCTS if product["object_type"] not in EXCLUDED_OBJECT_TYPES]
    columns = 8
    products = []
    for index in range(max(0, product_count)):
        base = available[index % len(available)]
        products.append(
            {
                "object_type": base["object_type"],
                "shape": base["shape"],
                "slot": {"index": index, "row": index // columns, "column": index % columns},
            }
        )
    return products


def _floor_faces(min_corner: np.ndarray, max_corner: np.ndarray) -> list[RenderFace]:
    floor_z = float(min_corner[2] - 1.2)
    x0 = float(min_corner[0] - 70)
    x1 = float(max_corner[0] + 70)
    y0 = float(min_corner[1] - 80)
    y1 = float(max_corner[1] + 80)
    color = (213, 217, 218)
    p0 = np.asarray([x0, y0, floor_z])
    p1 = np.asarray([x1, y0, floor_z])
    p2 = np.asarray([x1, y1, floor_z])
    p3 = np.asarray([x0, y1, floor_z])
    return [RenderFace(np.vstack([p0, p1, p2]), color, "floor"), RenderFace(np.vstack([p0, p2, p3]), color, "floor")]


def _mesh_faces(mesh: ShelfMesh) -> list[RenderFace]:
    return [RenderFace(mesh.vertices[list(face)], mesh.color, "shelf") for face in mesh.triangles]


def _product_faces(products: Sequence[dict[str, Any]], min_corner: np.ndarray, max_corner: np.ndarray) -> list[RenderFace]:
    faces: list[RenderFace] = []
    columns = 8
    shelf_tops = np.asarray(
        [
            min_corner[2] + 17.5,
            min_corner[2] + 41.5,
            min_corner[2] + 65.5,
            min_corner[2] + 89.5,
            min_corner[2] + 113.5,
        ]
    )
    y_positions = np.linspace(min_corner[1] + 12.0, max_corner[1] - 12.0, columns)
    front_x = min_corner[0] + 9.0

    for index, product in enumerate(products):
        base = AI2THOR_COMMON_PRODUCTS[index % len(AI2THOR_COMMON_PRODUCTS)]
        if base["object_type"] in EXCLUDED_OBJECT_TYPES:
            continue
        row = (index // columns) % len(shelf_tops)
        column = index % columns
        size = np.asarray(base["size"], dtype=float)
        jitter = ((index * 17) % 9 - 4) * 0.35
        center = np.asarray(
            [
                front_x + ((index % 3) - 1) * 2.1,
                y_positions[column] + jitter,
                shelf_tops[row] + size[2] * 0.5 + 0.8,
            ],
            dtype=float,
        )
        faces.extend(_primitive_faces(base["shape"], center, size, base["color"], base["object_type"]))
    return faces


def _primitive_faces(
    shape: str,
    center: np.ndarray,
    size: np.ndarray,
    color: tuple[int, int, int],
    object_type: str,
) -> list[RenderFace]:
    if shape == "box":
        return _box_faces(center, size, color, object_type)
    if shape in {"sphere", "ellipsoid"}:
        return _sphere_faces(center, size * 0.5, color, object_type)
    if shape == "cup":
        return _cylinder_faces(center, size[0] * 0.45, size[2], color, object_type, segments=14)
    if shape == "mug":
        faces = _cylinder_faces(center, size[0] * 0.45, size[2], color, object_type, segments=14)
        handle_center = center + np.asarray([0.0, size[1] * 0.48, size[2] * 0.05])
        faces.extend(_box_faces(handle_center, np.asarray([size[0] * 0.22, size[1] * 0.18, size[2] * 0.62]), color, object_type))
        return faces
    if shape == "bowl":
        return _cylinder_faces(center, size[0] * 0.52, size[2], color, object_type, segments=18)
    if shape == "plate":
        return _cylinder_faces(center, size[0] * 0.58, size[2], color, object_type, segments=20)
    if shape == "vase":
        faces = _cylinder_faces(center + np.asarray([0.0, 0.0, -size[2] * 0.12]), size[0] * 0.42, size[2] * 0.72, color, object_type, segments=16)
        faces.extend(_cylinder_faces(center + np.asarray([0.0, 0.0, size[2] * 0.33]), size[0] * 0.25, size[2] * 0.34, _scale_color(color, 1.12), object_type, segments=16))
        return faces
    return _cylinder_faces(center, size[0] * 0.45, size[2], color, object_type, segments=16)


def _box_faces(center: np.ndarray, size: np.ndarray, color: tuple[int, int, int], kind: str) -> list[RenderFace]:
    hx, hy, hz = size / 2.0
    offsets = np.asarray(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ]
    )
    vertices = center + offsets
    triangles = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
    return [RenderFace(vertices[list(face)], color, kind) for face in triangles]


def _cylinder_faces(
    center: np.ndarray,
    radius: float,
    height: float,
    color: tuple[int, int, int],
    kind: str,
    segments: int = 16,
) -> list[RenderFace]:
    faces: list[RenderFace] = []
    bottom_z = center[2] - height / 2.0
    top_z = center[2] + height / 2.0
    bottom_center = np.asarray([center[0], center[1], bottom_z])
    top_center = np.asarray([center[0], center[1], top_z])
    ring_bottom = []
    ring_top = []
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        offset = np.asarray([math.cos(angle) * radius, math.sin(angle) * radius, 0.0])
        ring_bottom.append(bottom_center + offset)
        ring_top.append(top_center + offset)
    for i in range(segments):
        j = (i + 1) % segments
        faces.append(RenderFace(np.vstack([ring_bottom[i], ring_bottom[j], ring_top[j]]), color, kind))
        faces.append(RenderFace(np.vstack([ring_bottom[i], ring_top[j], ring_top[i]]), color, kind))
        faces.append(RenderFace(np.vstack([bottom_center, ring_bottom[i], ring_bottom[j]]), _scale_color(color, 0.82), kind))
        faces.append(RenderFace(np.vstack([top_center, ring_top[j], ring_top[i]]), _scale_color(color, 1.08), kind))
    return faces


def _sphere_faces(center: np.ndarray, radii: np.ndarray, color: tuple[int, int, int], kind: str) -> list[RenderFace]:
    faces: list[RenderFace] = []
    vertical_steps = 8
    horizontal_steps = 14
    vertices: list[list[np.ndarray]] = []
    for v in range(vertical_steps + 1):
        theta = math.pi * v / vertical_steps
        row = []
        for h in range(horizontal_steps):
            phi = 2.0 * math.pi * h / horizontal_steps
            row.append(
                center
                + np.asarray(
                    [
                        math.sin(theta) * math.cos(phi) * radii[0],
                        math.sin(theta) * math.sin(phi) * radii[1],
                        math.cos(theta) * radii[2],
                    ]
                )
            )
        vertices.append(row)
    for v in range(vertical_steps):
        for h in range(horizontal_steps):
            hn = (h + 1) % horizontal_steps
            faces.append(RenderFace(np.vstack([vertices[v][h], vertices[v + 1][h], vertices[v + 1][hn]]), color, kind))
            faces.append(RenderFace(np.vstack([vertices[v][h], vertices[v + 1][hn], vertices[v][hn]]), color, kind))
    return faces


def _render_faces(
    faces: Sequence[RenderFace],
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    width: int,
    height: int,
    camera_eye: Sequence[float] | None = None,
    camera_target: Sequence[float] | None = None,
) -> Image.Image:
    scale = 2
    render_width = width * scale
    render_height = height * scale
    image = Image.new("RGB", (render_width, render_height), (235, 238, 240))
    draw = ImageDraw.Draw(image)

    center = (min_corner + max_corner) / 2.0
    span = max_corner - min_corner
    if camera_eye is None or camera_target is None:
        eye = np.asarray([min_corner[0] - 175.0, min_corner[1] - span[1] * 0.78, center[2] + span[2] * 0.22])
        target = np.asarray([center[0] + 3.0, center[1], center[2] + span[2] * 0.02])
    else:
        eye = np.asarray(camera_eye, dtype=float)
        target = np.asarray(camera_target, dtype=float)
    camera = _camera_basis(eye, target)

    projected: list[tuple[float, list[tuple[float, float]], tuple[int, int, int], np.ndarray, str]] = []
    for face in faces:
        points, depths = _project(face.vertices, camera, render_width, render_height)
        if points is None or np.any(depths <= 1.0):
            continue
        avg_depth = float(np.mean(depths))
        color = _lit_color(face.vertices, face.color, camera["forward"], face.kind)
        projected.append((avg_depth, points, color, face.vertices, face.kind))

    for _depth, points, color, _verts, kind in sorted(projected, key=lambda item: item[0], reverse=True):
        if kind == "floor":
            draw.polygon(points, fill=color)
        else:
            draw.polygon(points, fill=color)

    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def _camera_basis(eye: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    forward = _normalize(target - eye)
    world_up = np.asarray([0.0, 0.0, 1.0])
    right = _normalize(np.cross(forward, world_up))
    up = _normalize(np.cross(right, forward))
    return {"eye": eye, "forward": forward, "right": right, "up": up}


def _project(
    vertices: np.ndarray,
    camera: dict[str, np.ndarray],
    width: int,
    height: int,
    fov_degrees: float = 38.0,
) -> tuple[list[tuple[float, float]] | None, np.ndarray]:
    relative = vertices - camera["eye"]
    cam_x = relative @ camera["right"]
    cam_y = relative @ camera["up"]
    cam_z = relative @ camera["forward"]
    if np.all(cam_z <= 1.0):
        return None, cam_z
    focal = (height * 0.5) / math.tan(math.radians(fov_degrees) * 0.5)
    screen_x = width * 0.5 + cam_x * focal / cam_z
    screen_y = height * 0.54 - cam_y * focal / cam_z
    return list(zip(screen_x.tolist(), screen_y.tolist())), cam_z


def _lit_color(
    vertices: np.ndarray,
    base_color: tuple[int, int, int],
    camera_forward: np.ndarray,
    kind: str,
) -> tuple[int, int, int]:
    if kind == "floor":
        return base_color
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    normal = _normalize(normal)
    light = _normalize(np.asarray([-0.55, -0.45, 0.7]))
    diffuse = max(0.0, float(np.dot(normal, light)))
    facing = 0.10 * abs(float(np.dot(normal, camera_forward)))
    shade = 0.58 + 0.32 * diffuse + facing
    return _scale_color(base_color, shade)


def _scale_color(color: tuple[int, int, int], scale: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel * scale))) for channel in color)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm
