"""Structured intermediate representations.

The three pipeline stages communicate only through these artefacts -- never
through shared Python state.  Everything here round-trips through JSON so a run
can be inspected, cached and resumed stage by stage.

Stage 1 (planning)  -> SceneSpec, TerrainSpec, layout map
Stage 2 (terrain)   -> TerrainArtifacts (height field, masks, meshes)
Stage 3 (objects)   -> RegionalPlan, PlacementRecord   [M4, declared here so the
                       terrain stage already writes what stage 3 will consume]
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Stage 1 -- intent + planning
# --------------------------------------------------------------------------


class SceneIntent(Base):
    """Result of intent analysis.

    Per the paper this stage *extracts only what the prompt states* and invents
    nothing; unstated fields stay ``None`` so the planner can tell "the user
    said nothing" apart from "the user asked for this".
    """

    prompt: str
    biome: str | None = None
    landforms: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    mood: str | None = None
    time_of_day: str | None = None
    scale_hint: str | None = None
    explicit_constraints: list[str] = Field(default_factory=list)


class SceneSpec(Base):
    """Planned scene: intent plus the gaps the planner filled in."""

    name: str
    intent: SceneIntent
    terrain: "TerrainSpec"
    inferred_fields: list[str] = Field(
        default_factory=list,
        description="Field paths the planner supplied rather than the prompt; "
        "keeps user intent distinguishable from defaults.",
    )


# --------------------------------------------------------------------------
# Noise stack:  sum_k w_rk * N_rk(x)
# --------------------------------------------------------------------------


class NoiseComponent(Base):
    """One band of the regional noise stack.

    ``frequency`` is in cycles across the full world extent, so a value of 4
    means four lattice cells edge to edge regardless of raster resolution.
    """

    kind: Literal["perlin", "ridged", "billow"] = "perlin"
    frequency: float = Field(gt=0.0, description="cycles across the world extent")
    weight: float = Field(description="w_rk, in normalised height units")
    octaves: int = Field(default=1, ge=1, le=10)
    lacunarity: float = Field(default=2.0, gt=1.0)
    gain: float = Field(default=0.5, gt=0.0, lt=1.0)


# --------------------------------------------------------------------------
# Geomorphological operators:  sum_j alpha_rj * G_rj(x)
#
# The paper names peak / dune / terrace / erosion but does not formulate them.
# These parameter sets describe our own formulations (see terrain/operators.py).
# --------------------------------------------------------------------------


class PeakOperator(Base):
    kind: Literal["peak"] = "peak"
    alpha: float = Field(description="alpha_rj, in normalised height units")
    count: int = Field(default=3, ge=1, le=64)
    radius: float = Field(default=0.18, gt=0.0, le=1.0, description="fraction of world extent")
    radius_jitter: float = Field(default=0.35, ge=0.0, le=1.0)
    sharpness: float = Field(default=2.0, gt=0.0, description="profile exponent; >1 = pointier")
    amplitude_jitter: float = Field(default=0.3, ge=0.0, le=1.0)
    ridge_amount: float = Field(default=0.35, ge=0.0, le=1.0, description="ridged-noise modulation")
    ridge_frequency: float = Field(default=9.0, gt=0.0)
    inset: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
        description="keep peak centres this far inside the region (fraction of its inradius)",
    )


class DuneOperator(Base):
    kind: Literal["dune"] = "dune"
    alpha: float
    wavelength: float = Field(default=0.09, gt=0.0, le=1.0, description="fraction of world extent")
    direction_deg: float = 30.0
    asymmetry: float = Field(
        default=0.55, ge=0.0, lt=1.0, description="windward/lee skew; 0 = plain sine"
    )
    crest_sharpness: float = Field(default=1.6, gt=0.0)
    meander: float = Field(default=0.35, ge=0.0, description="lateral phase warp, in wavelengths")
    meander_frequency: float = Field(default=2.5, gt=0.0)


class TerraceOperator(Base):
    kind: Literal["terrace"] = "terrace"
    alpha: float = Field(default=1.0, description="1.0 = full terracing, 0.5 = half")
    steps: float = Field(default=7.0, gt=0.0, description="steps across one normalised height unit")
    sharpness: float = Field(default=0.75, ge=0.0, le=0.98, description="0 = no-op, ->1 = hard step")
    warp: float = Field(default=0.06, ge=0.0, description="noise warp of the step edges")
    warp_frequency: float = Field(default=6.0, gt=0.0)


class ErosionOperator(Base):
    kind: Literal["erosion"] = "erosion"
    alpha: float = Field(default=1.0)
    iterations: int = Field(default=40, ge=1, le=2000)
    talus_deg: float = Field(default=34.0, gt=0.0, lt=89.0, description="repose angle")
    strength: float = Field(default=0.5, gt=0.0, le=1.0)
    smooth_sigma_px: float = Field(default=0.0, ge=0.0, description="post-relaxation blur")


TerrainOperator = Annotated[
    Union[PeakOperator, DuneOperator, TerraceOperator, ErosionOperator],
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------
# Stage 2 -- terrain
# --------------------------------------------------------------------------


class MaterialSpec(Base):
    """Consumed in M2; carried through M1 so the layout stays the single source."""

    base_color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    roughness: float = Field(default=0.9, ge=0.0, le=1.0)
    slope_rock_blend: bool = True


class ScatterSpec(Base):
    """One scattered asset class within a region."""

    asset_class: str
    density_per_km2: float = Field(gt=0.0)
    max_slope_deg: float = Field(default=30.0, gt=0.0, lt=90.0)
    height_range_m: tuple[float, float] | None = None
    scale_range: tuple[float, float] = (0.8, 1.2)
    align_to_normal: float = Field(default=0.6, ge=0.0, le=1.0)
    footprint_radius_m: float = Field(
        default=1.0, gt=0.0, description="contact-bearing radius at scale 1.0"
    )
    min_spacing_m: float = Field(default=3.0, gt=0.0, description="centre-to-centre minimum")
    embed: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="fraction of the footprint's local relief to sink the base into "
        "the ground; trades a little burial for contact all round on a slope",
    )
    contact_tolerance_m: float = Field(
        default=0.15, gt=0.0, description="base-to-terrain distance still counted as contact"
    )


class ScatterInstance(Base):
    """One placed prop. Also the Unreal instanced-mesh export row (M5)."""

    asset_class: str
    region: str
    index: int
    position_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    yaw_deg: float
    scale: float
    footprint_radius_m: float
    contact_ratio: float = Field(ge=0.0, le=1.0)
    gap_m: float = Field(ge=0.0, description="largest float above the terrain")
    penetration_m: float = Field(ge=0.0, description="largest terrain poke above the base")


class RegionSpec(Base):
    """One terrain category of the layout map."""

    name: str
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$", description="layout-map colour code")
    base_height: float = Field(description="h_r, in normalised height units")
    blend_width_px: float | None = Field(
        default=None, ge=0.0, description="boundary softening; falls back to the global value"
    )
    noise: list[NoiseComponent] = Field(default_factory=list)
    operators: list[TerrainOperator] = Field(default_factory=list)
    material: MaterialSpec = Field(default_factory=MaterialSpec)
    scatter: list[ScatterSpec] = Field(default_factory=list)

    @property
    def rgb(self) -> tuple[int, int, int]:
        c = self.color.lstrip("#")
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


class TerrainSpec(Base):
    """Everything stage 2 needs: the layout map plus per-region parameters."""

    name: str
    seed: int = 0
    world_size_m: float = Field(default=1024.0, gt=0.0, description="edge length of the square world")
    height_scale_m: float = Field(
        default=200.0, gt=0.0, description="metres per normalised height unit"
    )
    resolution: int = Field(default=512, ge=32, le=8192, description="height-field samples per edge")
    blend_width_px: float = Field(default=8.0, ge=0.0, description="default boundary softening")
    despeckle_px: int = Field(
        default=0, ge=0, description="morphological cleanup of a hand-painted layout"
    )
    layout_map: str | None = Field(default=None, description="path to the layout bitmap")
    layout_template: str | None = Field(
        default=None,
        description="synthetic layout to paint when layout_map is missing; a "
        "stand-in for the hand-painted map of M1 and for the image model of M3",
    )
    regions: list[RegionSpec]

    @field_validator("regions")
    @classmethod
    def _unique(cls, v: list[RegionSpec]) -> list[RegionSpec]:
        if not v:
            raise ValueError("at least one region is required")
        for field in ("name", "color"):
            seen = [getattr(r, field).lower() for r in v]
            dupes = {x for x in seen if seen.count(x) > 1}
            if dupes:
                raise ValueError(f"duplicate region {field}: {sorted(dupes)}")
        return v

    @property
    def cell_size_m(self) -> float:
        """World metres per height-field sample."""
        return self.world_size_m / self.resolution

    def region_index(self, name: str) -> int:
        for i, r in enumerate(self.regions):
            if r.name == name:
                return i
        raise KeyError(name)


class RegionInstance(Base):
    """A connected component of one region category.

    Stage 3 selects *these*, not whole categories: "the clearing in the north",
    not "all forest".
    """

    region: str
    index: int
    area_px: int
    area_m2: float
    centroid_px: tuple[float, float]
    centroid_m: tuple[float, float]
    bbox_px: tuple[int, int, int, int]
    mean_height_m: float
    min_height_m: float
    max_height_m: float
    mean_slope_deg: float


class TerrainArtifacts(Base):
    """Manifest of what stage 2 produced -- the hand-off to stage 3."""

    spec_name: str
    run_id: str
    resolution: int
    world_size_m: float
    height_scale_m: float
    height_min_m: float
    height_max_m: float
    heightfield_npz: str
    weights_npz: str
    labels_png: str
    heightmap_png16: str
    mesh_obj: str
    preview_png: str | None = None
    region_instances: list[RegionInstance] = Field(default_factory=list)
    splat_maps: list[str] = Field(default_factory=list)
    scatter_json: str | None = None
    scatter_statistics: dict[str, float] = Field(default_factory=dict)
    checksum: str = Field(default="", description="sha256 of the height field, for reproducibility")


# --------------------------------------------------------------------------
# Stage 3 -- regional objects  [M4; declared for schema completeness]
# --------------------------------------------------------------------------


class RegionalPlan(Base):
    """One planned population zone.

    Per the paper's pipeline figure these are *semantic* zones the planner draws
    onto the layout map ("Cabin Village", "Fishing Village"), typically a
    rectangular window -- not identical with the terrain-category patches of
    ``RegionInstance``.  ``bbox_px`` carries that window; the terrain-category
    link stays optional context.
    """

    name: str = Field(description="semantic zone name, e.g. 'cabin village'")
    bbox_px: tuple[int, int, int, int] | None = Field(
        default=None, description="planned window on the layout map (x0, y0, x1, y1)"
    )
    region: str | None = Field(
        default=None, description="dominant terrain category, if one applies"
    )
    instance_index: int | None = None
    camera_name: str = ""
    object_prompts: list[str] = Field(default_factory=list)
    max_objects: int = Field(
        default=50, ge=1, description="detection cap per region; the figure shows ~50"
    )
    iteration_budget: int = Field(
        default=15,
        ge=1,
        description="hard per-region refinement cap, enforced in code; the "
        "paper's pipeline figure shows a /loop budget of 15 per region",
    )


class PlacementRecord(Base):
    """One placed object: the output of the ray-pair placement of section 3.2."""

    object_id: str
    category: str
    region: str
    mesh_path: str
    # ray pair
    anchor_world_m: tuple[float, float, float]
    reference_point_cam: tuple[float, float, float]
    depth_anchor_m: float
    depth_object_m: float
    # resolved transform
    scale: float
    rotation_matrix: list[list[float]]
    translation_m: tuple[float, float, float]
    l2c_baked_into_vertices: bool = Field(
        default=False,
        description="if true the T_l2c factor is already in the vertices and must not be reapplied",
    )
    # quality gates
    calibration_scale: float = 1.0
    contact_ratio: float = 0.0
    contact_passed: bool = False
    refine_iterations: int = 0

    @model_validator(mode="after")
    def _check_rotation(self) -> "PlacementRecord":
        if len(self.rotation_matrix) != 3 or any(len(r) != 3 for r in self.rotation_matrix):
            raise ValueError("rotation_matrix must be 3x3")
        return self


SceneSpec.model_rebuild()
