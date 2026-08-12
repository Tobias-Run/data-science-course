"""Procedural terrain material driven by the splat map.

One material, not one per region: the regions blend continuously along
``m_tilde_r``, and a shader mix is the only place that blend can happen without
re-cutting the mesh.  Colour and roughness are mixed per channel, then a
slope-driven rock layer goes on top -- steep ground shows rock whatever the
region says, which is what stops a grass plateau from running vertically up a
cliff face.

Purely procedural: no baked texture files. glTF export therefore carries
geometry and UVs but not this material; the Unreal path (M5) rebuilds an
equivalent landscape material from the same splat map.
"""

from __future__ import annotations

import math


def _srgb(rgb, alpha: float = 1.0):
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), alpha)


def build_terrain_material(bpy, name: str, splat_path: str, regions: list[dict],
                           rock_color=(0.30, 0.28, 0.26), rock_slope_deg: float = 38.0):
    """Build the blended terrain material.

    ``regions`` is a list of ``{"name", "channel", "base_color", "roughness"}``
    in splat-channel order, at most four -- the channels of one RGBA image.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes["Principled BSDF"]

    tex_coord = nodes.new("ShaderNodeTexCoord")
    splat_img = nodes.new("ShaderNodeTexImage")
    splat_img.image = bpy.data.images.load(splat_path)
    # Weights are data, not colour: an sRGB transfer curve would bend the blend.
    splat_img.image.colorspace_settings.name = "Non-Color"
    splat_img.interpolation = "Linear"
    splat_img.extension = "EXTEND"
    links.new(tex_coord.outputs["UV"], splat_img.inputs["Vector"])

    sep = nodes.new("ShaderNodeSeparateColor")
    sep.mode = "RGB"
    links.new(splat_img.outputs["Color"], sep.inputs["Color"])
    channel_out = {
        "R": sep.outputs["Red"],
        "G": sep.outputs["Green"],
        "B": sep.outputs["Blue"],
        "A": splat_img.outputs["Alpha"],
    }

    # Fine-grain variation so a region is not a flat wash of one colour.
    detail = nodes.new("ShaderNodeTexNoise")
    detail.inputs["Scale"].default_value = 260.0
    detail.inputs["Detail"].default_value = 6.0

    colour = None
    rough = None
    for r in regions:
        weight = channel_out[r["channel"]]

        # Noise drives the *factor* between two tones of the region colour.
        # Feeding it into a colour input instead mixes grey into every region
        # and washes the whole terrain out to one beige.
        base = nodes.new("ShaderNodeMixRGB")
        base.blend_type = "MIX"
        links.new(detail.outputs["Fac"], base.inputs["Fac"])
        base.inputs["Color1"].default_value = _srgb(r["base_color"])
        base.inputs["Color2"].default_value = _srgb([c * 0.72 for c in r["base_color"]])

        if colour is None:
            colour, rough = base.outputs["Color"], r["roughness"]
            rough_node = nodes.new("ShaderNodeValue")
            rough_node.outputs[0].default_value = float(r["roughness"])
            rough = rough_node.outputs[0]
            continue

        mix_c = nodes.new("ShaderNodeMixRGB")
        mix_c.blend_type = "MIX"
        links.new(weight, mix_c.inputs["Fac"])
        links.new(colour, mix_c.inputs["Color1"])
        links.new(base.outputs["Color"], mix_c.inputs["Color2"])
        colour = mix_c.outputs["Color"]

        mix_r = nodes.new("ShaderNodeMix")
        mix_r.data_type = "FLOAT"
        links.new(weight, mix_r.inputs["Factor"])
        links.new(rough, mix_r.inputs[2])
        mix_r.inputs[3].default_value = float(r["roughness"])
        rough = mix_r.outputs[0]

    # Slope mask: geometry normal z falls off as ground steepens.
    geom = nodes.new("ShaderNodeNewGeometry")
    sep_n = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geom.outputs["Normal"], sep_n.inputs["Vector"])
    slope = nodes.new("ShaderNodeMapRange")
    slope.inputs["From Min"].default_value = math.cos(math.radians(rock_slope_deg + 10.0))
    slope.inputs["From Max"].default_value = math.cos(math.radians(rock_slope_deg - 10.0))
    slope.inputs["To Min"].default_value = 1.0
    slope.inputs["To Max"].default_value = 0.0
    slope.clamp = True
    links.new(sep_n.outputs["Z"], slope.inputs["Value"])

    mix_rock = nodes.new("ShaderNodeMixRGB")
    mix_rock.blend_type = "MIX"
    links.new(slope.outputs["Result"], mix_rock.inputs["Fac"])
    links.new(colour, mix_rock.inputs["Color1"])
    mix_rock.inputs["Color2"].default_value = _srgb(rock_color)

    links.new(mix_rock.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(rough, bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.25
    return mat


def placeholder_asset(bpy, asset_class: str):
    """A stand-in mesh per asset class, origin at its base.

    Origin at the base, not the centre: the scatter table stores the contact
    point, so an asset centred on its origin would sink half its height into the
    ground and every contact number would be a lie.
    """
    name = f"proto_{asset_class}"
    if name in bpy.data.objects:
        return bpy.data.objects[name]

    kind = asset_class.lower()
    if any(k in kind for k in ("tree", "pine", "shrub", "bush", "grass")):
        bpy.ops.mesh.primitive_cone_add(radius1=0.9, depth=3.2, vertices=7)
        ob = bpy.context.active_object
        colour = (0.11, 0.20, 0.09, 1.0)
        height = 3.2
    elif any(k in kind for k in ("rock", "boulder", "stone", "pebble")):
        bpy.ops.mesh.primitive_ico_sphere_add(radius=0.8, subdivisions=1)
        ob = bpy.context.active_object
        ob.scale = (1.0, 0.85, 0.7)
        colour = (0.24, 0.22, 0.20, 1.0)
        height = 1.12
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.4)
        ob = bpy.context.active_object
        colour = (0.35, 0.30, 0.24, 1.0)
        height = 1.4

    ob.name = name
    ob.data.name = name
    # Lift the geometry so the object origin sits on its base.
    for v in ob.data.vertices:
        v.co.z += height / 2.0
    ob.location = (0.0, 0.0, 0.0)
    ob.scale = ob.scale  # keep any per-kind squash

    mat = bpy.data.materials.new(f"mat_{asset_class}")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = colour
    mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    ob.data.materials.append(mat)

    ob.hide_render = True
    ob.hide_viewport = True
    return ob


def instance_scatter(bpy, instances: list[dict], collection_name: str = "scatter"):
    """Link one object per scatter instance, sharing mesh data per asset class.

    Shared mesh data rather than copies: a few thousand props would otherwise
    duplicate their geometry and dominate both memory and the .blend size.
    """
    coll = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(coll)

    protos: dict[str, object] = {}
    for inst in instances:
        cls = inst["asset_class"]
        if cls not in protos:
            protos[cls] = placeholder_asset(bpy, cls)
        proto = protos[cls]

        ob = bpy.data.objects.new(f"{cls}_{inst['region']}_{inst['index']}", proto.data)
        coll.objects.link(ob)
        ob.location = tuple(inst["position_m"])
        ob.scale = (inst["scale"],) * 3

        nx, ny, nz = inst["normal"]
        # Tilt the local z axis onto the blended normal, then spin about it.
        ob.rotation_mode = "ZYX"
        ob.rotation_euler = (
            math.atan2(-ny, max(nz, 1e-6)),
            math.atan2(nx, max(nz, 1e-6)),
            math.radians(inst["yaw_deg"]),
        )
    return coll
