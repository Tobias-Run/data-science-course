"""Headless Blender stage: height field -> .blend / glTF + diagnostic renders.

Runs as its own process (``python -m worldclaw.blender.build_terrain``) and
exits when done.  That is not tidiness -- it is the VRAM strategy from the
briefing: no generator model and no Blender session ever hold graphics memory at
the same time, and every hand-off goes through files in the run directory.

Materials here are deliberately flat grey.  Procedural PBR node graphs are M2;
what M1 needs from Blender is proof that the geometry is real, walkable and
correctly scaled.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from ..terrain import frame


def _require_bpy():
    try:
        import bpy  # noqa: F401
    except ImportError:
        sys.exit(
            "bpy is not importable. Install the Blender python module:\n"
            "    pip install 'worldclaw[blender]'\n"
            "or run this script with Blender's own interpreter:\n"
            "    blender --background --python -m worldclaw.blender.build_terrain -- ..."
        )
    import bpy

    return bpy


def build_mesh(bpy, height_m: np.ndarray, cell_size_m: float, name: str = "terrain"):
    """Create the terrain mesh from the height raster.

    Vertices are handed to Blender as one flat buffer via ``foreach_set``; the
    per-vertex Python path is orders of magnitude slower and would make a 1024²
    field take minutes.
    """
    h, w = height_m.shape
    xs = (np.arange(w, dtype=np.float32) - (w - 1) / 2.0) * cell_size_m
    ys = ((h - 1) / 2.0 - np.arange(h, dtype=np.float32)) * cell_size_m
    gx, gy = np.meshgrid(xs, ys)
    verts = np.stack([gx.ravel(), gy.ravel(), height_m.ravel()], axis=1).astype(np.float32)

    idx = np.arange(h * w, dtype=np.int32).reshape(h, w)
    quads = np.stack(
        [idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel(), idx[:-1, :-1].ravel()],
        axis=1,
    )

    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(quads.size)
    me.loops.foreach_set("vertex_index", quads.ravel())
    me.polygons.add(len(quads))
    me.polygons.foreach_set("loop_start", np.arange(len(quads), dtype=np.int32) * 4)
    me.polygons.foreach_set("loop_total", np.full(len(quads), 4, dtype=np.int32))
    me.update(calc_edges=True)
    me.validate()

    # UVs: one unit square across the whole terrain, ready for M2 materials.
    uv = me.uv_layers.new(name="UVMap")
    u = np.linspace(0.0, 1.0, w, dtype=np.float32)
    v = np.linspace(1.0, 0.0, h, dtype=np.float32)
    gu, gv = np.meshgrid(u, v)
    uv_per_vertex = np.stack([gu.ravel(), gv.ravel()], axis=1)
    uv.data.foreach_set("uv", uv_per_vertex[quads.ravel()].ravel())

    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


def add_material(bpy, ob, color=(0.25, 0.23, 0.20, 1.0)):
    mat = bpy.data.materials.new("terrain_placeholder")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.92
    ob.data.materials.append(mat)
    return mat


def add_sun(bpy, elevation_deg=42.0, azimuth_deg=315.0, strength=3.0):
    light = bpy.data.lights.new("sun", "SUN")
    light.energy = strength
    light.angle = math.radians(1.5)
    ob = bpy.data.objects.new("sun", light)
    bpy.context.collection.objects.link(ob)
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    ob.rotation_euler = (math.radians(90.0) - el, 0.0, az)
    return ob


def add_camera(bpy, name, location, look_at=(0.0, 0.0, 0.0), lens_mm=35.0, clip_end=None):
    """Camera aimed by construction -- no constraints, so the pose is exact.

    Stage 3 back-projects rays through these poses, so the extrinsics have to be
    something we compute rather than something an aim-constraint resolves at
    render time.
    """
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens_mm
    # Explicit rather than AUTO: AUTO fits the sensor to the larger render
    # dimension, which silently changes fx for portrait renders and would break
    # the intrinsics written to cameras.json.
    cam_data.sensor_fit = "HORIZONTAL"
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = location

    d = np.array(look_at, dtype=np.float64) - np.array(location, dtype=np.float64)
    n = np.linalg.norm(d)
    # Blender's default far clip is 100 m. At kilometre scale that silently
    # culls the entire terrain and renders bare sky, so it is set from the
    # actual viewing distance rather than left at the default.
    cam_data.clip_start = 0.1
    cam_data.clip_end = float(clip_end) if clip_end else max(10.0 * n, 1000.0)
    if n < 1e-9:
        raise ValueError(f"camera {name}: location and look_at coincide")
    d /= n
    # Blender cameras look down local -z with +y up.
    yaw = math.atan2(d[1], d[0]) - math.pi / 2.0
    pitch = math.acos(max(-1.0, min(1.0, -d[2])))
    cam.rotation_euler = (pitch, 0.0, yaw)
    return cam


def camera_intrinsics(bpy, cam, res_x: int, res_y: int) -> dict:
    """Pinhole intrinsics + world-to-camera extrinsics for this camera.

    Written out with every render because stage 3's ray-pair placement needs
    exactly ``K_t`` and ``E_t`` of the image it is editing.  Recovering them
    later from a .blend would be guesswork.
    """
    cd = cam.data
    sensor = cd.sensor_width
    fx = cd.lens * res_x / sensor
    fy = fx  # square pixels; sensor_fit is pinned to HORIZONTAL in add_camera
    cx, cy = res_x / 2.0, res_y / 2.0
    world_to_cam = np.array(cam.matrix_world.inverted(), dtype=np.float64)
    # Blender camera coords are x right / y up / z backward; computer-vision
    # convention is x right / y down / z forward.  Stage 3 does its ray algebra
    # in CV convention (u = fx*X/Z + cx with image y downward), so the converted
    # extrinsics are written alongside the raw ones instead of being derived --
    # incorrectly, eventually -- by some future reader.
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return {
        "name": cam.name,
        "lens_mm": cd.lens,
        "sensor_width_mm": sensor,
        "resolution": [res_x, res_y],
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "E_world_to_cam": world_to_cam.tolist(),
        "E_world_to_cam_cv": (flip @ world_to_cam).tolist(),
        "location": list(cam.location),
        "rotation_euler": list(cam.rotation_euler),
        "convention": "E_world_to_cam: blender cam looks along -z, +y up. "
        "E_world_to_cam_cv: x right, y down, z forward; K projects u=fx*X/Z+cx, "
        "v=fy*Y/Z+cy with pixel y downward.",
    }


def sample_height(height_m: np.ndarray, cell_size_m: float, x: float, y: float) -> float:
    """Terrain height at a world position, bilinearly interpolated."""
    return float(frame.sample_height_m(height_m, cell_size_m, x, y))


def _world_xy(height_m: np.ndarray, cell_size_m: float, row: int, col: int) -> tuple[float, float]:
    x, y = frame.world_from_raster(height_m.shape, cell_size_m, row, col)
    return (float(x), float(y))


def framed_camera(
    target: tuple[float, float, float],
    radius_m: float,
    azimuth_deg: float,
    elevation_deg: float,
    lens_mm: float,
    sensor_mm: float = 36.0,
    margin: float = 1.15,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Position a camera so a sphere of ``radius_m`` around ``target`` fits.

    Framing from the actual terrain bounds rather than from hand-tuned offsets:
    the previous fixed offsets aimed *below* a high plateau and rendered its
    surface at point-blank range, which tells you nothing about the terrain.
    """
    half_fov = math.atan(sensor_mm / (2.0 * lens_mm))
    dist = margin * radius_m / max(math.sin(half_fov), 1e-6)
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    loc = (
        target[0] + dist * math.cos(el) * math.sin(az),
        target[1] - dist * math.cos(el) * math.cos(az),
        target[2] + dist * math.sin(el),
    )
    return loc, target


def rim_viewpoint(
    height_m: np.ndarray, cell_size_m: float, scan_m: float = 160.0
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Find a standpoint on the edge of the sharpest drop, and what it overlooks.

    "Highest point looking at the lowest point" is the obvious heuristic and a
    poor one: on a canyon spec the highest point is a summit somewhere else
    entirely, and the view is its own dome filling the frame.  Maximum *local
    relief* instead lands on the rim of the gorge, which is where the height
    field has to be judged -- and where object placement will be hardest.
    """
    from scipy import ndimage

    k = max(3, int(round(scan_m / cell_size_m)) | 1)
    hi = ndimage.maximum_filter(height_m, size=k, mode="nearest")
    lo = ndimage.minimum_filter(height_m, size=k, mode="nearest")
    relief = hi - lo
    # Prefer a standpoint on the upper side of the drop, not down in it.
    upper = height_m >= (lo + 0.75 * relief)
    score = np.where(upper, relief, 0.0)

    row, col = np.unravel_index(int(np.argmax(score)), score.shape)
    half = k // 2
    r0, r1 = max(row - half, 0), min(row + half + 1, height_m.shape[0])
    c0, c1 = max(col - half, 0), min(col + half + 1, height_m.shape[1])
    window = height_m[r0:r1, c0:c1]
    lr, lc = np.unravel_index(int(np.argmin(window)), window.shape)

    return (
        _world_xy(height_m, cell_size_m, row, col),
        _world_xy(height_m, cell_size_m, r0 + lr, c0 + lc),
    )


def default_cameras(height_m: np.ndarray, cell_size_m: float) -> list[dict]:
    """Two framed overviews plus one eye-height view standing on the terrain.

    The eye-height camera is the one that answers the M1 acceptance question --
    is this walkable, is the scale believable -- and it is the camera type stage
    3 will render for regional composition, so it is worth having early.
    """
    h, w = height_m.shape
    world_x, world_y = w * cell_size_m, h * cell_size_m
    lo, hi = float(height_m.min()), float(height_m.max())
    centre = (0.0, 0.0, (lo + hi) / 2.0)
    radius = 0.5 * math.sqrt(world_x**2 + world_y**2 + (hi - lo) ** 2)

    cams = []
    for name, az, el, lens in (
        ("cam_overview", 200.0, 38.0, 32.0),
        ("cam_oblique", 115.0, 20.0, 50.0),
    ):
        loc, look = framed_camera(centre, radius, az, el, lens)
        cams.append({"name": name, "location": loc, "look_at": look, "lens_mm": lens})

    (hx, hy), (lx, ly) = rim_viewpoint(height_m, cell_size_m)
    eye = 1.7
    # Step back from the summit so the foreground is ground, not a cliff edge.
    d = np.array([hx - lx, hy - ly], dtype=np.float64)
    d = d / max(np.linalg.norm(d), 1e-6)
    # Stand *on* the rim, not back from it: a few tens of metres of convex
    # ground behind the edge is enough to occlude the drop entirely.
    sx, sy = hx + d[0] * 4.0, hy + d[1] * 4.0
    sz = sample_height(height_m, cell_size_m, sx, sy) + eye
    # Aim across the drop rather than into it, and only slightly downwards: a
    # standing observer aimed at a point far below sees nothing but the ground
    # at their feet, whereas a near-level view puts the far side and the horizon
    # in frame at a believable eye height.
    tx, ty = sx + (lx - sx) * 5.0, sy + (ly - sy) * 5.0
    cams.append(
        {
            "name": "cam_ground",
            "location": (sx, sy, sz),
            "look_at": (tx, ty, sz - 0.06 * float(np.hypot(tx - sx, ty - sy))),
            "lens_mm": 35.0,
        }
    )
    return cams


def backdrop_height(height_m: np.ndarray, clearance_m: float = 0.5) -> float:
    """Height for the horizon plane: below every terrain sample.

    Split out from ``add_backdrop`` so the invariant that makes the difference
    between a landscape and a black frame is testable without Blender.
    """
    return float(height_m.min()) - clearance_m


def add_backdrop(bpy, height_m: np.ndarray, cell_size_m: float, material, extent_factor: float = 12.0):
    """A large plane at the terrain's edge height, continuing to the horizon.

    Without it the world simply stops at the tile boundary and the lower half of
    the sky shows through as a dark band that reads as open water.  That is not
    only ugly: stage 3 feeds these renders to an image-edit model, and a false
    sea invites it to populate the scene with boats.  It is excluded from
    placement -- nothing is ever scattered or anchored on it.

    The plane sits just *below the lowest ground in the scene*.  An earlier
    version used the median height of the terrain's border, which put it at
    plateau level in a canyon scene: a camera standing on the gorge floor was
    then under a horizon-to-horizon ceiling and rendered pure black.  Only a
    plane beneath every terrain sample is guaranteed never to enclose a
    viewpoint that stands on the terrain.
    """
    z = backdrop_height(height_m)
    size = max(height_m.shape) * cell_size_m * extent_factor

    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, z))
    ob = bpy.context.active_object
    ob.name = "backdrop"
    if material is not None:
        ob.data.materials.append(material)
    return ob


def enable_depth_pass(bpy, out_dir: Path):
    """Route the Z pass to an EXR file per render.

    The gate in ``placement/depth_gate.py`` needs a *true* depth map for the
    pre-edit render; only the edited image has to make do with an estimator.
    Rendering it here rather than estimating both sides removes half the
    uncertainty from the sharpest metric in the acceptance criteria.
    """
    scene = bpy.context.scene
    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = True
    # Blender 5 moved the compositor from scene.node_tree to a node group;
    # 4.x still exposes the old attribute, so both are accepted.
    tree = getattr(scene, "node_tree", None)
    if tree is None:
        tree = scene.compositing_node_group
        if tree is None:
            tree = bpy.data.node_groups.new("compositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    rl = tree.nodes.new("CompositorNodeRLayers")
    # Blender 5 dropped CompositorNodeComposite in favour of the node group's
    # own output; 4.x still has it. Either way the colour image must keep a path
    # to the output, or enabling the depth pass silently blanks every render.
    if hasattr(bpy.types, "CompositorNodeComposite"):
        comp = tree.nodes.new("CompositorNodeComposite")
        tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
    else:
        if not tree.interface.items_tree:
            tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        group_out = tree.nodes.new("NodeGroupOutput")
        tree.links.new(rl.outputs["Image"], group_out.inputs[0])

    out = tree.nodes.new("CompositorNodeOutputFile")
    # Blender 5's file-output node only offers the multilayer EXR variant, and
    # the RNA enum still advertises both, so this is decided by trying.
    try:
        out.format.file_format = "OPEN_EXR"
    except TypeError:
        out.format.file_format = "OPEN_EXR_MULTILAYER"
    out.format.color_depth = "32"
    if hasattr(out, "file_slots"):  # Blender 4.x
        out.base_path = str(out_dir)
        out.file_slots.clear()
        out.file_slots.new("depth")
    else:  # Blender 5.x: named items, directory taken from scene.render.filepath
        out.file_output_items.new("FLOAT", "depth")
    tree.links.new(rl.outputs["Depth"], out.inputs["depth"])
    return out


def set_depth_output_name(node, name: str) -> None:
    """Name the depth file for the camera currently being rendered."""
    if hasattr(node, "file_slots"):
        node.file_slots[0].path = f"{name}_"
    else:
        node.file_name = f"{name}_depth"


def read_exr_depth(bpy, path: Path, res_x: int, res_y: int) -> np.ndarray:
    """Load an EXR depth file through Blender and return it as a numpy array.

    Blender is already loaded, so it is also the EXR reader -- no extra
    dependency for a format that would otherwise need one.  Blender images are
    stored bottom-up, hence the flip.
    """
    img = bpy.data.images.load(str(path))
    try:
        buf = np.array(img.pixels[:], dtype=np.float32)
        depth = buf.reshape(res_y, res_x, 4)[..., 0]
        return np.flipud(depth).copy()
    finally:
        bpy.data.images.remove(img)


def add_sky(bpy, strength: float = 0.15):
    """A physical sky, not a black void.

    Diagnostic renders are read by eye now and by the image-edit model in M4;
    both need a horizon and sky lighting to judge scale against.
    """
    world = bpy.data.worlds.new("world")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    bg.inputs["Strength"].default_value = strength
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sun_elevation = math.radians(42.0)
    sky.sun_rotation = math.radians(315.0)
    nt.links.new(sky.outputs[0], bg.inputs["Color"])
    return world


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heightfield", required=True, help="terrain/heightfield.npz from stage 2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="terrain")
    ap.add_argument("--render", action="store_true", help="render the diagnostic views")
    ap.add_argument("--render-samples", type=int, default=24)
    ap.add_argument("--render-width", type=int, default=960)
    ap.add_argument("--render-height", type=int, default=540)
    ap.add_argument("--decimate", type=int, default=1, help="raster stride before meshing")
    ap.add_argument("--export-gltf", action="store_true")
    ap.add_argument("--save-blend", action="store_true")
    ap.add_argument("--splat", help="terrain/splat.json; enables the blended material")
    ap.add_argument("--scatter", help="terrain/scatter.json; instances the props")
    ap.add_argument("--spec", help="terrain spec, for per-region material colours")
    ap.add_argument("--depth", action="store_true",
                    help="also write a true depth map per render (the M4 gate's reference)")
    ap.add_argument("--no-backdrop", action="store_true",
                    help="omit the horizon plane; the tile edge then shows bare sky")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    # Resolve every path argument to absolute before bpy touches any of them.
    # bpy.data.images.load() and friends resolve a relative path against
    # Blender's own notion of "current file directory" -- meaningful for a
    # saved .blend, undefined for the empty in-memory scene this script
    # creates -- rather than the OS process's working directory the way
    # open() or pathlib would. That mismatch is platform-dependent: it did
    # not surface on Linux, and did on the first real Windows run, on
    # exactly this call. Absolute paths sidestep the ambiguity entirely.
    args.heightfield = str(Path(args.heightfield).resolve())
    args.out_dir = str(Path(args.out_dir).resolve())
    for attr in ("splat", "scatter", "spec"):
        if getattr(args, attr):
            setattr(args, attr, str(Path(getattr(args, attr)).resolve()))

    # Material colours come from the spec's MaterialSpec so the layout map stays
    # the single source for what a region looks like.
    args.region_colors, args.region_roughness = {}, {}
    if args.spec and Path(args.spec).exists():
        for r in json.loads(Path(args.spec).read_text(encoding="utf-8"))["regions"]:
            mat = r.get("material", {})
            args.region_colors[r["name"]] = tuple(mat.get("base_color", (0.35, 0.33, 0.30)))
            args.region_roughness[r["name"]] = float(mat.get("roughness", 0.9))

    bpy = _require_bpy()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(args.heightfield) as z:
        height = np.asarray(z["height_m"], dtype=np.float32)
        cell = float(z["cell_size_m"])
    if args.decimate > 1:
        height = height[:: args.decimate, :: args.decimate]
        cell *= args.decimate

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0

    ob = build_mesh(bpy, height, cell, args.name)
    if args.splat and Path(args.splat).exists():
        from .materials import build_terrain_material

        doc = json.loads(Path(args.splat).read_text(encoding="utf-8"))
        first = doc["maps"][0]
        # splat.json is always written by pipeline.py at "<run>/terrain/splat.json"
        # with image paths recorded relative to "<run>/", via RunContext.rel() --
        # so two levels up from splat.json is always the right base, never a guess.
        splat_path = Path(args.splat).parent.parent / first["path"]
        regions = [
            {
                "name": name,
                "channel": ch,
                "base_color": args.region_colors.get(name, (0.35, 0.33, 0.30)),
                "roughness": args.region_roughness.get(name, 0.9),
            }
            for ch, name in sorted(first["channels"].items())
        ]
        terrain_mat = build_terrain_material(bpy, "terrain_blend", str(splat_path), regions)
        ob.data.materials.append(terrain_mat)
    else:
        terrain_mat = add_material(bpy, ob)
    if not args.no_backdrop:
        add_backdrop(bpy, height, cell, terrain_mat)
    add_sun(bpy)
    add_sky(bpy)

    scatter_count = 0
    if args.scatter and Path(args.scatter).exists():
        from .materials import instance_scatter

        insts = json.loads(Path(args.scatter).read_text(encoding="utf-8"))["instances"]
        instance_scatter(bpy, insts)
        scatter_count = len(insts)

    world_size = cell * height.shape[1]
    hmax = float(height.max())
    cams = [
        add_camera(bpy, c["name"], c["location"], c["look_at"], c["lens_mm"])
        for c in default_cameras(height, cell)
    ]

    scene.render.resolution_x = args.render_width
    scene.render.resolution_y = args.render_height
    scene.render.image_settings.file_format = "PNG"
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"  # the GPU belongs to the generator models
    scene.cycles.samples = args.render_samples
    scene.cycles.use_denoising = True
    # Sun plus physical sky overexposes a bright surface on the default
    # settings; AgX with a little negative exposure keeps the slopes readable.
    scene.view_settings.view_transform = "AgX"
    # Calibrated so a lit slope lands near mid-grey: the sun lamp and the
    # physical sky both contribute, and at default strengths the terrain renders
    # as a featureless white sheet.
    scene.view_settings.exposure = -2.0

    intrinsics = [camera_intrinsics(bpy, c, args.render_width, args.render_height) for c in cams]
    (out / "cameras.json").write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")

    depth_node = enable_depth_pass(bpy, out / "_depth_tmp") if args.depth else None
    rendered, depth_maps, depth_warnings = [], [], []
    if args.render:
        for cam in cams:
            scene.camera = cam
            path = out / f"render_{cam.name}.png"
            scene.render.filepath = str(path)
            if depth_node is not None:
                set_depth_output_name(depth_node, cam.name)
            bpy.ops.render.render(write_still=True)
            rendered.append(str(path))

            if depth_node is not None:
                exrs = sorted(out.glob(f"{cam.name}_*.exr")) + \
                       sorted((out / "_depth_tmp").glob(f"{cam.name}_*.exr"))
                if exrs:
                    npy = out / f"depth_{cam.name}.npy"
                    np.save(npy, read_exr_depth(bpy, exrs[-1], args.render_width,
                                                args.render_height))
                    depth_maps.append(str(npy))
                    for f in exrs:
                        f.unlink()
                else:
                    depth_warnings.append(cam.name)

    if args.export_gltf:
        bpy.ops.export_scene.gltf(filepath=str(out / f"{args.name}.glb"), export_format="GLB")
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(out / f"{args.name}.blend"))

    summary = {
        "vertices": len(ob.data.vertices),
        "polygons": len(ob.data.polygons),
        "world_size_m": world_size,
        "height_max_m": hmax,
        "cell_size_m": cell,
        "renders": rendered,
        "depth_maps": depth_maps,
        "depth_unavailable_for": depth_warnings,
        "scatter_instances": scatter_count,
        "blender": bpy.app.version_string,
    }
    if depth_warnings:
        # Loud, because a silently missing reference map would leave the M4
        # gate comparing an estimate against nothing and calling it a pass.
        print(
            f"WARNING: no depth file was produced for {depth_warnings}. "
            "Blender's compositor file output is in flux across versions; the "
            "gate's supported path is running the same depth estimator on the "
            "before and after images, where the estimator's bias cancels.",
            file=sys.stderr,
        )
    (out / "blender_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
