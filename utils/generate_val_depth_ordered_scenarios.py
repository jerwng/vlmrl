#!/usr/bin/env python3
"""
Generate validation scenarios with strict depth-ordered object placement.

Objects are placed in size tiers from nearest to farthest:
  Tier 1 — small:   cardboard_box, trash_can    (x ≈ 1.5–2.5 m)
  Tier 2 — medium:  chair, couch, table          (x ≈ 3.5–5.0 m)
  Tier 3 — large:   bookshelf                    (x ≈ 6.0–8.0 m)

Within every scene, no smaller-tier object is placed behind a larger-tier
object in depth (x-axis). This makes the scenes visually "natural" (small
objects occlude large ones) and provides a systematic validation condition
absent from the existing FT1–FT6 scenarios.

Output: loco-mujoco/loco_mujoco/models/unitree_go2/finetune_scenarios/<ID>/scene_room.xml
IDs:    FT7_<group>_<num>
"""

import os

BASE_DIR = (
    "/home/chenyuanwang01/VLM+RL/loco-mujoco/loco_mujoco/models/"
    "unitree_go2/finetune_val_scenarios"
)

# Reuse room header/footer and helpers from generate_finetune_scenarios.py
SIDE_EULER = {
    "F":  1.5708, "FR": 0.7854, "R":  0.0,   "BR": 5.4978,
    "Bk": 4.7124, "BL": 3.9270, "L":  3.1416, "FL": 2.3562,
}

F  = SIDE_EULER["F"]
FR = SIDE_EULER["FR"]
R  = SIDE_EULER["R"]
BR = SIDE_EULER["BR"]
BK = SIDE_EULER["Bk"]
BL = SIDE_EULER["BL"]
L  = SIDE_EULER["L"]
FL = SIDE_EULER["FL"]

ROOM_HEADER = """\
<mujoco model="scene">

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.871 0.616 0.4 0.5"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture builtin="gradient" height="100" rgb1="0.9 0.9 0.9" rgb2="0.55 0.25 0.0" type="skybox" width="100"/>
    <include file="../../room/objects/materials.xml"/>
  </asset>

    <worldbody>
        <geom name="floor" friction="1 .1 .1" pos="0 0 0" size="0 0 0.125" type="plane" material="mat_floor" condim="3" conaffinity="1" contype="1" group="2"/>
        <light cutoff="1000" diffuse="1.5 1.5 1.5" dir="-0 0 -1.3" directional="true" exponent="10" pos="0 0 10.3" specular=".1 .1 .1" castshadow="false"/>

        <geom name="wall_negx" type="box" pos="-15.0 0.0 1.25" size="0.05 15.0 1.25" material="mat_wall"/>
        <geom name="wall_posx" type="box" pos=" 15.0 0.0 1.25" size="0.05 15.0 1.25" material="mat_wall"/>
        <geom name="wall_negy" type="box" pos="0.0 -15.0 1.25" size="15.0 0.05 1.25" material="mat_wall"/>
        <geom name="wall_posy" type="box" pos="0.0  15.0 1.25" size="15.0 0.05 1.25" material="mat_wall"/>

        <geom name="ceiling" type="box"
          pos="0 0 2.55"
          size="15.0 15.0 0.05"
          material="mat_wall"
          contype="1" conaffinity="1"
        />
"""

ROOM_FOOTER = "    </worldbody>\n</mujoco>"


def deuler(xml_name):
    """Default euler: 0 for rotationally invariant objects, Front otherwise."""
    return 0.0 if xml_name in ("cardboard_box", "trash_can") else F


def make_xml(scenario_id, description, bodies):
    """
    bodies: list of (body_name, xml_file, x, y, euler_z, comment_or_None)
    Handles duplicate xml_file names by appending _2, _3, etc.
    """
    parts = [ROOM_HEADER]
    parts.append(f"        <!-- SCENARIO: {scenario_id} | {description} -->\n")
    has_dist = any("DISTRACTOR" in (b[5] or "") for b in bodies)
    xml_file_counts = {}
    for bname, xml_file, x, y, euler_z, comment in bodies:
        xml_file_counts[xml_file] = xml_file_counts.get(xml_file, 0) + 1
        count = xml_file_counts[xml_file]
        unique_xml = xml_file if count == 1 else f"{xml_file}_{count}"
        if comment:
            parts.append(f"        <!-- {comment} -->")
        parts.append(
            f'        <body name="{bname}" pos="{x:.4f} {y:.4f} 0.0" euler="0 0 {euler_z:.4f}">'
        )
        parts.append(f'          <include file="../../room/objects/{unique_xml}.xml"/>')
        parts.append(f"        </body>\n")
    if not has_dist:
        parts.append("        <!-- DISTRACTORS: none -->\n")
    parts.append(ROOM_FOOTER)
    return "\n".join(parts)


def write_xml(scenario_id, xml):
    d = os.path.join(BASE_DIR, scenario_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "scene_room.xml"), "w") as f:
        f.write(xml)


def build_bodies(obj_list):
    """
    obj_list: list of (xml_name, x, y, euler_z)
    Auto-assigns body names: <xml>_target for first instance, <xml>_distractor<N> for rest.
    Returns list of body tuples for make_xml().
    """
    counts = {}
    bodies = []
    for xml, x, y, euler in obj_list:
        counts[xml] = counts.get(xml, 0) + 1
        n = counts[xml]
        if n == 1:
            bname = f"{xml}_target"
            comment = f"TARGET: {xml}"
        else:
            bname = f"{xml}_distractor{n - 1}"
            comment = f"DISTRACTOR: {xml}"
        bodies.append((bname, xml, x, y, euler, comment))
    return bodies


# ════════════════════════════════════════════════════════════════════════════
#  DEPTH ORDERING CONSTRAINT (enforced by construction):
#
#   Tier 1 (small)  — cardboard_box, trash_can  →  x ∈ [1.5, 2.5]
#   Tier 2 (medium) — chair, couch, table        →  x ∈ [3.5, 5.0]
#   Tier 3 (large)  — bookshelf                  →  x ∈ [6.0, 8.0]
#
#  Every entry below is manually verified: all tier-1 x < tier-2 x < tier-3 x.
# ════════════════════════════════════════════════════════════════════════════

# Each scenario entry: (sid, description, [(xml, x, y, euler), ...])
# Objects listed front-to-back within each entry.

# ── Group A: 1 small + 1 medium + 1 bookshelf ────────────────────────────
# 2 small types × 3 medium types × 3 spatial configs = 18 scenarios
GROUP_A = [
    # cardboard_box + chair + bookshelf
    ("FT7_A_01", "BX near + CH mid + B far, center",
     [("cardboard_box", 2.0,  0.0,  0.0),
      ("chair",         4.0,  0.0,  F),
      ("bookshelf",     7.0,  0.0,  F)]),
    ("FT7_A_02", "BX near + CH mid + B far, laterally spread",
     [("cardboard_box", 2.0,  1.0,  0.0),
      ("chair",         4.0, -0.5,  F),
      ("bookshelf",     6.5,  0.5,  F)]),
    ("FT7_A_03", "BX near + CH mid + B far, offset right",
     [("cardboard_box", 2.5, -0.8,  0.0),
      ("chair",         4.5, -1.5,  F),
      ("bookshelf",     7.5, -1.0,  F)]),
    # cardboard_box + couch + bookshelf
    ("FT7_A_04", "BX near + C mid + B far, center",
     [("cardboard_box", 2.0,  0.0,  0.0),
      ("couch",         4.0,  0.0,  F),
      ("bookshelf",     7.0,  0.5,  F)]),
    ("FT7_A_05", "BX near + C mid + B far, offset left",
     [("cardboard_box", 1.8,  1.5,  0.0),
      ("couch",         3.5,  0.5,  F),
      ("bookshelf",     6.5,  1.5,  F)]),
    ("FT7_A_06", "BX near + C mid + B far, staggered depths",
     [("cardboard_box", 2.5, -0.5,  0.0),
      ("couch",         5.0,  1.0,  F),
      ("bookshelf",     8.0, -0.5,  F)]),
    # cardboard_box + table + bookshelf
    ("FT7_A_07", "BX near + T mid + B far, center",
     [("cardboard_box", 2.0,  0.0,  0.0),
      ("table",         4.0,  0.0,  F),
      ("bookshelf",     7.0,  0.0,  F)]),
    ("FT7_A_08", "BX near + T mid + B far, lateral spread",
     [("cardboard_box", 2.0, -1.0,  0.0),
      ("table",         4.0,  0.5,  F),
      ("bookshelf",     7.0,  1.5,  F)]),
    ("FT7_A_09", "BX near + T mid + B far, all offset right",
     [("cardboard_box", 2.5, -1.5,  0.0),
      ("table",         4.5, -1.0,  F),
      ("bookshelf",     7.0, -1.5,  F)]),
    # trash_can + chair + bookshelf
    ("FT7_A_10", "TC near + CH mid + B far, center",
     [("trash_can",    2.0,  0.0,  0.0),
      ("chair",        4.0,  0.0,  F),
      ("bookshelf",    7.0,  0.0,  F)]),
    ("FT7_A_11", "TC near + CH mid + B far, offset left",
     [("trash_can",    1.5,  0.8,  0.0),
      ("chair",        3.5,  1.5,  F),
      ("bookshelf",    6.0,  0.5,  F)]),
    ("FT7_A_12", "TC near + CH mid + B far, offset right",
     [("trash_can",    2.0, -1.0,  0.0),
      ("chair",        4.0, -1.5,  F),
      ("bookshelf",    7.5, -0.5,  F)]),
    # trash_can + couch + bookshelf
    ("FT7_A_13", "TC near + C mid + B far, center",
     [("trash_can",    2.0,  0.0,  0.0),
      ("couch",        4.0,  0.0,  F),
      ("bookshelf",    7.0,  0.5,  F)]),
    ("FT7_A_14", "TC near + C mid + B far, laterally spread",
     [("trash_can",    1.5,  1.5,  0.0),
      ("couch",        3.5, -0.5,  F),
      ("bookshelf",    6.5,  1.0,  F)]),
    ("FT7_A_15", "TC near + C mid + B far, deep stagger",
     [("trash_can",    2.5, -0.3,  0.0),
      ("couch",        5.0,  0.8,  F),
      ("bookshelf",    8.0, -0.5,  F)]),
    # trash_can + table + bookshelf
    ("FT7_A_16", "TC near + T mid + B far, center",
     [("trash_can",    2.0,  0.0,  0.0),
      ("table",        4.0,  0.0,  F),
      ("bookshelf",    7.0,  0.0,  F)]),
    ("FT7_A_17", "TC near + T mid + B far, left side",
     [("trash_can",    1.8,  1.0,  0.0),
      ("table",        3.5,  0.5,  F),
      ("bookshelf",    6.5,  1.5,  F)]),
    ("FT7_A_18", "TC near + T mid + B far, all near-ish",
     [("trash_can",    1.5,  0.2,  0.0),
      ("table",        3.5,  0.0,  F),
      ("bookshelf",    6.0,  0.3,  F)]),
]

# ── Group B: 2 small objects + 1 medium + 1 bookshelf ────────────────────
# BX+TC paired; varied mediums and depths; 13 scenarios
GROUP_B = [
    # BX + TC + chair + bookshelf
    ("FT7_B_01", "BX+TC near + CH mid + B far, aligned center",
     [("cardboard_box", 2.0,  0.5,  0.0),
      ("trash_can",     2.0, -0.5,  0.0),
      ("chair",         4.0,  0.0,  F),
      ("bookshelf",     7.0,  0.0,  F)]),
    ("FT7_B_02", "BX+TC near + CH mid + B far, spread laterally",
     [("cardboard_box", 1.5,  1.2,  0.0),
      ("trash_can",     2.0, -0.8,  0.0),
      ("chair",         4.0,  0.5,  F),
      ("bookshelf",     6.5, -0.5,  F)]),
    ("FT7_B_03", "BX+TC near + CH mid + B far, depth-staggered small",
     [("cardboard_box", 1.5,  0.0,  0.0),
      ("trash_can",     2.5,  1.0,  0.0),
      ("chair",         4.5,  0.5,  F),
      ("bookshelf",     7.5,  0.0,  F)]),
    # BX + TC + couch + bookshelf
    ("FT7_B_04", "BX+TC near + C mid + B far, center",
     [("cardboard_box", 2.0,  0.5,  0.0),
      ("trash_can",     1.5, -0.3,  0.0),
      ("couch",         4.0,  0.0,  F),
      ("bookshelf",     7.0,  0.5,  F)]),
    ("FT7_B_05", "BX+TC near + C mid + B far, spread right",
     [("cardboard_box", 2.0, -1.0,  0.0),
      ("trash_can",     1.5, -0.3,  0.0),
      ("couch",         4.0, -1.5,  F),
      ("bookshelf",     7.0, -0.5,  F)]),
    ("FT7_B_06", "BX+TC near + C mid + B far, small flanking medium",
     [("cardboard_box", 2.0,  1.5,  0.0),
      ("trash_can",     2.0, -1.5,  0.0),
      ("couch",         4.5,  0.0,  F),
      ("bookshelf",     7.5,  0.0,  F)]),
    # BX + TC + table + bookshelf
    ("FT7_B_07", "BX+TC near + T mid + B far, center",
     [("cardboard_box", 2.0,  0.4,  0.0),
      ("trash_can",     2.0, -0.4,  0.0),
      ("table",         4.0,  0.0,  F),
      ("bookshelf",     7.0,  0.0,  F)]),
    ("FT7_B_08", "BX+TC near + T mid + B far, offset left",
     [("cardboard_box", 1.5,  1.0,  0.0),
      ("trash_can",     2.5,  2.0,  0.0),
      ("table",         4.0,  1.0,  F),
      ("bookshelf",     6.5,  1.5,  F)]),
    ("FT7_B_09", "BX+TC near + T mid + B far, diagonal placement",
     [("cardboard_box", 1.5, -0.5,  0.0),
      ("trash_can",     2.0,  0.5,  0.0),
      ("table",         4.0, -0.5,  F),
      ("bookshelf",     7.5,  0.5,  F)]),
    # 2× cardboard_box + medium + bookshelf
    ("FT7_B_10", "2×BX near + CH mid + B far",
     [("cardboard_box", 1.5,  0.5,  0.0),
      ("cardboard_box", 2.0, -0.5,  0.0),
      ("chair",         4.0,  0.0,  F),
      ("bookshelf",     7.0,  0.0,  F)]),
    ("FT7_B_11", "2×BX near + T mid + B far, spread",
     [("cardboard_box", 1.5,  1.0,  0.0),
      ("cardboard_box", 2.0, -1.0,  0.0),
      ("table",         4.5,  0.0,  F),
      ("bookshelf",     7.0,  0.5,  F)]),
    # 2× trash_can + medium + bookshelf
    ("FT7_B_12", "2×TC near + C mid + B far",
     [("trash_can",    1.5,  0.5,  0.0),
      ("trash_can",    2.0, -0.5,  0.0),
      ("couch",        4.0,  0.0,  F),
      ("bookshelf",    6.5,  0.0,  F)]),
    ("FT7_B_13", "2×TC near + T mid + B far, spread",
     [("trash_can",    1.5,  1.2,  0.0),
      ("trash_can",    1.5, -1.2,  0.0),
      ("table",        4.0,  0.0,  F),
      ("bookshelf",    7.0,  0.0,  F)]),
]

# ── Group C: 1 small + 2 medium + 1 bookshelf ────────────────────────────
# 13 scenarios
GROUP_C = [
    # BX + chair + couch + bookshelf
    ("FT7_C_01", "BX near + CH+C mid + B far, center",
     [("cardboard_box", 2.0,  0.0,  0.0),
      ("chair",         3.5,  1.0,  F),
      ("couch",         5.0, -0.5,  F),
      ("bookshelf",     7.0,  0.0,  F)]),
    ("FT7_C_02", "BX near + CH+C mid + B far, left-spread",
     [("cardboard_box", 1.5,  1.5,  0.0),
      ("chair",         4.0,  1.0,  F),
      ("couch",         4.0, -0.5,  F),
      ("bookshelf",     7.0,  0.5,  F)]),
    ("FT7_C_03", "BX near + CH+C mid + B far, depth-staggered medium",
     [("cardboard_box", 2.0, -0.3,  0.0),
      ("chair",         3.5,  0.5,  F),
      ("couch",         5.0,  0.0,  F),
      ("bookshelf",     7.5, -0.5,  F)]),
    # BX + chair + table + bookshelf
    ("FT7_C_04", "BX near + CH+T mid + B far, center",
     [("cardboard_box", 2.0,  0.0,  0.0),
      ("chair",         3.5, -0.5,  F),
      ("table",         4.5,  1.0,  F),
      ("bookshelf",     7.0,  0.0,  F)]),
    ("FT7_C_05", "BX near + CH+T mid + B far, right spread",
     [("cardboard_box", 2.5, -1.0,  0.0),
      ("chair",         4.0, -1.5,  F),
      ("table",         4.5, -0.5,  F),
      ("bookshelf",     7.0, -1.0,  F)]),
    ("FT7_C_06", "BX near + CH+T mid + B far, lateral stagger",
     [("cardboard_box", 1.5,  0.5,  0.0),
      ("table",         3.5, -0.5,  F),
      ("chair",         5.0,  1.5,  F),
      ("bookshelf",     7.5,  0.0,  F)]),
    # BX + couch + table + bookshelf
    ("FT7_C_07", "BX near + C+T mid + B far, center",
     [("cardboard_box", 2.0,  0.0,  0.0),
      ("table",         3.5,  1.0,  F),
      ("couch",         5.0, -0.5,  F),
      ("bookshelf",     7.5,  0.0,  F)]),
    ("FT7_C_08", "BX near + C+T mid + B far, left-heavy",
     [("cardboard_box", 2.0,  1.5,  0.0),
      ("couch",         4.0,  1.0,  F),
      ("table",         4.5,  2.0,  F),
      ("bookshelf",     7.0,  1.0,  F)]),
    ("FT7_C_09", "BX near + C+T mid + B far, spread out",
     [("cardboard_box", 1.8, -0.5,  0.0),
      ("couch",         3.5,  0.5,  F),
      ("table",         5.0, -1.5,  F),
      ("bookshelf",     7.5,  0.0,  F)]),
    # trash_can + chair + couch + bookshelf
    ("FT7_C_10", "TC near + CH+C mid + B far, center",
     [("trash_can",    2.0,  0.0,  0.0),
      ("chair",        3.5,  1.0,  F),
      ("couch",        5.0, -0.5,  F),
      ("bookshelf",    7.0,  0.0,  F)]),
    ("FT7_C_11", "TC near + CH+C mid + B far, right side",
     [("trash_can",    1.5, -0.5,  0.0),
      ("chair",        4.0, -1.0,  F),
      ("couch",        5.0, -1.5,  F),
      ("bookshelf",    7.0, -0.5,  F)]),
    # trash_can + couch + table + bookshelf
    ("FT7_C_12", "TC near + C+T mid + B far, center",
     [("trash_can",    2.0,  0.0,  0.0),
      ("couch",        3.5,  0.5,  F),
      ("table",        5.0, -0.5,  F),
      ("bookshelf",    7.5,  0.0,  F)]),
    ("FT7_C_13", "TC near + C+T mid + B far, spread",
     [("trash_can",    1.5,  1.0,  0.0),
      ("table",        3.5,  0.0,  F),
      ("couch",        5.0,  1.5,  F),
      ("bookshelf",    7.0,  0.5,  F)]),
]

# ── Group D: 2 small + 2 medium + 1 bookshelf (dense scenes) ─────────────
# 11 scenarios
GROUP_D = [
    ("FT7_D_01", "BX+TC near + CH+C mid + B far, center cluster",
     [("cardboard_box", 2.0,  0.5,  0.0),
      ("trash_can",     1.5, -0.5,  0.0),
      ("chair",         3.5,  0.5,  F),
      ("couch",         5.0, -0.5,  F),
      ("bookshelf",     7.0,  0.0,  F)]),
    ("FT7_D_02", "BX+TC near + CH+C mid + B far, spread",
     [("cardboard_box", 2.0,  1.5,  0.0),
      ("trash_can",     2.0, -1.5,  0.0),
      ("chair",         4.0,  1.0,  F),
      ("couch",         4.0, -1.5,  F),
      ("bookshelf",     7.0,  0.0,  F)]),
    ("FT7_D_03", "BX+TC near + CH+C mid + B far, depth stagger all",
     [("trash_can",     1.5,  0.0,  0.0),
      ("cardboard_box", 2.5,  1.0,  0.0),
      ("chair",         3.5, -0.5,  F),
      ("couch",         5.0,  1.5,  F),
      ("bookshelf",     8.0, -0.5,  F)]),
    ("FT7_D_04", "BX+TC near + CH+T mid + B far, center",
     [("cardboard_box", 2.0,  0.5,  0.0),
      ("trash_can",     1.5, -0.5,  0.0),
      ("chair",         3.5,  1.0,  F),
      ("table",         5.0, -0.5,  F),
      ("bookshelf",     7.0,  0.0,  F)]),
    ("FT7_D_05", "BX+TC near + CH+T mid + B far, left-skewed",
     [("cardboard_box", 2.0,  1.5,  0.0),
      ("trash_can",     1.5,  0.5,  0.0),
      ("chair",         3.5,  1.5,  F),
      ("table",         5.0,  0.5,  F),
      ("bookshelf",     7.5,  1.0,  F)]),
    ("FT7_D_06", "BX+TC near + CH+T mid + B far, diagonal",
     [("trash_can",     1.5, -1.0,  0.0),
      ("cardboard_box", 2.5,  0.5,  0.0),
      ("table",         3.5, -0.5,  F),
      ("chair",         5.0,  1.0,  F),
      ("bookshelf",     7.0, -0.5,  F)]),
    ("FT7_D_07", "BX+TC near + C+T mid + B far, center",
     [("cardboard_box", 2.0,  0.5,  0.0),
      ("trash_can",     2.0, -0.5,  0.0),
      ("table",         3.5,  0.0,  F),
      ("couch",         5.0,  0.8,  F),
      ("bookshelf",     7.0,  0.0,  F)]),
    ("FT7_D_08", "BX+TC near + C+T mid + B far, right-skewed",
     [("cardboard_box", 2.0, -0.5,  0.0),
      ("trash_can",     1.5, -1.5,  0.0),
      ("couch",         4.0, -1.0,  F),
      ("table",         4.5, -2.0,  F),
      ("bookshelf",     7.0, -1.0,  F)]),
    ("FT7_D_09", "BX+TC near + C+T mid + B far, wide spread",
     [("cardboard_box", 1.5,  2.0,  0.0),
      ("trash_can",     2.0, -2.0,  0.0),
      ("couch",         4.0,  1.5,  F),
      ("table",         4.5, -1.5,  F),
      ("bookshelf",     7.5,  0.0,  F)]),
    ("FT7_D_10", "2×BX near + CH+T mid + B far",
     [("cardboard_box", 1.5,  0.8,  0.0),
      ("cardboard_box", 2.5, -0.8,  0.0),
      ("chair",         4.0,  0.5,  F),
      ("table",         4.5, -0.5,  F),
      ("bookshelf",     7.0,  0.0,  F)]),
    ("FT7_D_11", "2×TC near + C+T mid + B far, center line",
     [("trash_can",    1.5,  0.5,  0.0),
      ("trash_can",    2.5, -0.5,  0.0),
      ("couch",        4.0,  0.0,  F),
      ("table",        5.0,  1.0,  F),
      ("bookshelf",    7.5, -0.5,  F)]),
]

# ── Group E: no bookshelf — small + medium only (2-tier depth ordering) ───
# 13 scenarios
GROUP_E = [
    # BX + chair
    ("FT7_E_01", "BX near + CH mid only, center",
     [("cardboard_box", 2.0,  0.0,  0.0),
      ("chair",         4.0,  0.0,  F)]),
    ("FT7_E_02", "BX near + CH mid only, offset left",
     [("cardboard_box", 1.5,  1.0,  0.0),
      ("chair",         3.5,  1.5,  F)]),
    ("FT7_E_03", "BX near + CH mid only, far medium",
     [("cardboard_box", 2.0, -0.5,  0.0),
      ("chair",         5.0,  0.0,  F)]),
    # BX + couch
    ("FT7_E_04", "BX near + C mid only, center",
     [("cardboard_box", 2.0,  0.0,  0.0),
      ("couch",         4.0,  0.0,  F)]),
    ("FT7_E_05", "BX near + C mid only, offset right",
     [("cardboard_box", 2.0, -1.0,  0.0),
      ("couch",         4.5, -1.5,  F)]),
    ("FT7_E_06", "BX near + C mid only, wide couch far",
     [("cardboard_box", 1.5,  0.5,  0.0),
      ("couch",         5.0,  0.0,  F)]),
    # TC + table
    ("FT7_E_07", "TC near + T mid only, center",
     [("trash_can",    2.0,  0.0,  0.0),
      ("table",        4.0,  0.0,  F)]),
    ("FT7_E_08", "TC near + T mid only, offset left",
     [("trash_can",    1.5,  1.0,  0.0),
      ("table",        3.5,  0.8,  F)]),
    ("FT7_E_09", "TC near + T mid only, far table",
     [("trash_can",    2.0, -0.5,  0.0),
      ("table",        5.0, -0.5,  F)]),
    # BX + TC + chair
    ("FT7_E_10", "BX+TC near + CH mid, center",
     [("cardboard_box", 2.0,  0.5,  0.0),
      ("trash_can",     1.5, -0.5,  0.0),
      ("chair",         4.0,  0.0,  F)]),
    ("FT7_E_11", "BX+TC near + CH mid, spread",
     [("cardboard_box", 1.5,  1.5,  0.0),
      ("trash_can",     2.0, -1.5,  0.0),
      ("chair",         4.0,  0.0,  F)]),
    # BX + TC + couch + table
    ("FT7_E_12", "BX+TC near + C+T mid, center",
     [("cardboard_box", 2.0,  0.5,  0.0),
      ("trash_can",     1.5, -0.3,  0.0),
      ("couch",         4.0,  0.5,  F),
      ("table",         5.0, -0.5,  F)]),
    ("FT7_E_13", "BX+TC near + C+T mid, left-heavy",
     [("cardboard_box", 2.0,  1.5,  0.0),
      ("trash_can",     1.5,  0.5,  0.0),
      ("table",         3.5,  1.0,  F),
      ("couch",         5.0,  1.5,  F)]),
]


def gen_all():
    total = 0
    for group_name, group_data in [
        ("A", GROUP_A),
        ("B", GROUP_B),
        ("C", GROUP_C),
        ("D", GROUP_D),
        ("E", GROUP_E),
    ]:
        count = 0
        for sid, desc, obj_list in group_data:
            bodies = build_bodies(obj_list)
            xml = make_xml(sid, desc, bodies)
            write_xml(sid, xml)
            count += 1
        print(f"Group {group_name}: {count} scenarios")
        total += count
    print(f"Total FT7 scenarios: {total}")
    return total


if __name__ == "__main__":
    gen_all()
