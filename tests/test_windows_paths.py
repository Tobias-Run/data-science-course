"""Scene names come straight from a local model's freeform text and become
directory names (new_run_id uses spec.name as its prefix). Windows forbids a
specific set of characters in a path component and reserves a handful of
device names outright; a model is free to produce any of them.

expand.py's _slugify is the single place that sanitises this, so every
downstream path built from a planned scene's name inherits the guarantee
rather than each call site re-deriving it.
"""

from __future__ import annotations

import pytest

from worldclaw.planning.expand import _slugify


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Red Sandstone Canyon", "red-sandstone-canyon"),
        ("  leading and trailing  ", "leading-and-trailing"),
        ("multiple   spaces\tand\ttabs", "multiple-spaces-and-tabs"),
    ],
)
def test_ordinary_names_become_readable_slugs(raw, expected):
    assert _slugify(raw, fallback="scene") == expected


@pytest.mark.parametrize("forbidden", list('<>:"/\\|?*'))
def test_windows_forbidden_characters_are_stripped(forbidden):
    slug = _slugify(f"canyon{forbidden}floor", fallback="scene")
    assert forbidden not in slug


def test_trailing_dot_is_removed():
    """Windows silently strips a trailing dot from a real path; matching it
    here avoids a slug that quietly becomes a different string once it hits
    the filesystem than the one recorded in terrain_spec.json."""
    assert not _slugify("canyon.", fallback="scene").endswith(".")


@pytest.mark.parametrize("reserved", ["con", "PRN", "Aux", "nul", "COM1", "lpt9"])
def test_windows_reserved_device_names_are_replaced(reserved):
    assert _slugify(reserved, fallback="scene") == "scene"


def test_empty_or_whitespace_only_falls_back():
    assert _slugify("", fallback="scene") == "scene"
    assert _slugify("   ", fallback="scene") == "scene"
    assert _slugify("...", fallback="scene") == "scene"


def test_long_names_are_capped():
    slug = _slugify("x" * 300, fallback="scene", max_len=60)
    assert len(slug) <= 60


def test_non_latin_unicode_is_kept_not_stripped():
    """Accents and non-Latin scripts are legal on NTFS; only the specific
    forbidden character set actually breaks path creation, so nothing here
    should be discarded just for not being ASCII."""
    slug = _slugify("Schlucht Grönland", fallback="scene")
    assert "grönland" in slug


def test_slug_is_reused_for_the_layout_map_path():
    """The spec's name and its layout_map filename must agree, or the
    terrain stage looks for a layout file under a different name than the
    one the spec actually carries."""
    from worldclaw.planning.expand import expand
    from worldclaw.planning.schema import LayoutPlan, RegionPlan, ScenePlan

    plan = ScenePlan(
        name="Weird: Name/With*Chars",
        biome="b", world_size_m=1024.0, relief_m=100.0,
        regions=[
            RegionPlan(name="a", category="plain", relative_area=0.5, elevation=0.2),
            RegionPlan(name="b", category="rock", relative_area=0.5, elevation=0.8),
        ],
        layout=LayoutPlan(composition="bands", image_prompt="x"),
    )
    spec = expand(plan)
    assert spec.layout_map == f"layouts/{spec.name}.png"
