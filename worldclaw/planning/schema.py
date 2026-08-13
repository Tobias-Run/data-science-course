"""The planning intermediate representation.

Deliberately *not* the full ``TerrainSpec``.  Asking a local model to emit noise
frequencies, octave counts and erosion iteration budgets invites confident
nonsense and strains schema-constrained decoding with deeply nested unions.

Instead the model states intent -- "this region is a rugged plateau, high, about
a third of the map" -- and ``expand.py`` turns intent into numbers by rule.
That is also what the paper's planner does: it designs regions, materials and
terrain character, not sampling parameters.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..schemas import Base

# Kept small and closed: every value maps to a rule in expand.py, so a category
# the planner invents would silently produce a default landform.
TerrainCategory = Literal[
    "water", "basin", "plain", "sand", "dune", "slope", "forest", "plateau", "rock", "peak"
]

Character = Literal[
    "smooth", "rolling", "rugged", "terraced", "eroded", "peaked", "dunes", "flat"
]

# A closed vocabulary rather than free text or a hex code: schema-constrained
# decoding then guarantees a value expand.py can resolve, and small local models
# are unreliable at inventing hex but perfectly reliable at picking a word.
ColorHint = Literal[
    "red", "orange", "ochre", "yellow", "brown", "grey", "white", "black",
    "green", "blue", "teal", "purple", "pink",
]

Composition = Literal[
    "channel",  # a valley or river cutting across the map
    "bands",  # parallel strips, e.g. coast -> plain -> mountains
    "patches",  # scattered blobs of one category inside another
    "radial",  # a central massif or crater with surrounding rings
    "island",  # one landmass surrounded by a border category
]


class RegionPlan(Base):
    """One planned terrain region."""

    name: str = Field(description="short human name, e.g. 'canyon floor'")
    category: TerrainCategory = Field(
        description="what the ground IS, which sets its landform and roughness. "
        "water=open water, basin=low bare floor, plain=GRASSLAND (green), "
        "sand=flat sandy ground, dune=wind-blown dunes, slope=talus/scree flank, "
        "forest=wooded, plateau=high flat tableland, rock=bare rock, peak=summit. "
        "Match it to the biome: an arid scene uses sand/dune/slope/plateau/rock, "
        "not plain or forest."
    )
    relative_area: float = Field(
        ge=0.01, le=1.0, description="share of the map, roughly; normalised later"
    )
    elevation: float = Field(
        ge=0.0, le=1.0, description="0 = lowest ground in the scene, 1 = highest"
    )
    character: list[Character] = Field(
        default_factory=list, max_length=3, description="landform character, not parameters"
    )
    color_hint: ColorHint | None = Field(
        default=None,
        description="the colour this ground actually is. Set it for EVERY region, "
        "reasoning from the biome: an arid canyon rim is ochre or brown, never "
        "green. Null falls back to the category's own colour, which assumes a "
        "temperate biome -- so 'plain' would render as grass green even in a "
        "desert. Only leave it null if the category's own colour is already right.",
    )
    scatter: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="asset classes to scatter here, e.g. 'boulder', 'shrub'",
    )


class LayoutPlan(Base):
    """How the regions are arranged, and the prompt that draws them."""

    composition: Composition
    image_prompt: str = Field(
        description="prompt for the layout-map image model: flat colour regions, "
        "top-down, no text, no shading"
    )
    notes: str = ""


class ScenePlan(Base):
    """The planner's output. Expanded into a TerrainSpec by rule."""

    name: str = Field(description="short slug, lowercase, no spaces")
    biome: str
    world_size_m: float = Field(ge=256.0, le=8192.0)
    relief_m: float = Field(
        ge=5.0, le=2000.0, description="height difference between lowest and highest ground"
    )
    regions: list[RegionPlan] = Field(min_length=2, max_length=8)
    layout: LayoutPlan
    inferred_fields: list[str] = Field(
        default_factory=list,
        description="fields you filled in that the prompt did not state; "
        "keeps user intent separable from planner defaults",
    )
