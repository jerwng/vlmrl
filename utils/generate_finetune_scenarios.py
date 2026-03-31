#!/usr/bin/env python3
"""
Generate all ~1000 VLM LoRA finetune scenario scene_room.xml files.
Output: loco-mujoco/loco_mujoco/models/unitree_go2/finetune_scenarios/<ID>/scene_room.xml

Follows the plan in .claude/plans/2026-03-28-vlm-lora-finetune-scenarios.md
"""

import os

BASE_DIR = (
    "/home/chenyuanwang01/VLM+RL/loco-mujoco/loco_mujoco/models/"
    "unitree_go2/finetune_scenarios"
)

# Side code → euler Z (radians)
SIDE_EULER = {
    "F":  1.5708, "FR": 0.7854, "R":  0.0,   "BR": 5.4978,
    "Bk": 4.7124, "BL": 3.9270, "L":  3.1416, "FL": 2.3562,
}

# Distance code → x (m)
DIST_X = {"VN": 1.5, "N": 2.5, "M": 3.5, "F": 5.5, "VF": 7.5}

# Object code → XML include name
OBJ_XML = {
    "C":  "couch",
    "T":  "table",
    "CH": "chair",
    "B":  "bookshelf",
    "BX": "cardboard_box",
    "TC": "trash_can",
}

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
    """Default distractor euler: 0 for rotationally invariant objects, Front otherwise."""
    return 0.0 if xml_name in ("cardboard_box", "trash_can") else 1.5708


def make_xml(scenario_id, description, bodies):
    """
    bodies: list of (body_name, xml_file, x, y, euler_z, comment_or_None)
    If the same xml_file appears more than once, subsequent uses get _2, _3, etc.
    (e.g. chair → chair, chair_2, chair_3).  Assumes all variant XMLs exist.
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


# ════════════════════════════════════════════════════════════════════════════
#  TIER 1  6 × 8 × 5 = 240
# ════════════════════════════════════════════════════════════════════════════
def gen_tier1():
    objs = [
        ("C",  "couch"),
        ("T",  "table"),
        ("CH", "chair"),
        ("B",  "bookshelf"),
        ("BX", "cardboard_box"),
        ("TC", "trash_can"),
    ]
    sides = ["F", "FR", "R", "BR", "Bk", "BL", "L", "FL"]
    dists = [("VN", 1.5), ("N", 2.5), ("M", 3.5), ("F", 5.5), ("VF", 7.5)]
    count = 0
    for obj_code, obj_xml in objs:
        for side in sides:
            for dist_code, x in dists:
                sid = f"FT1_{obj_code}_{side}_{dist_code}"
                euler = SIDE_EULER[side]
                bodies = [
                    (f"{obj_xml}_target", obj_xml, x, 0.0, euler, f"TARGET: {obj_xml}"),
                ]
                xml = make_xml(sid, f"Single {obj_xml}, {side} view, x={x}m", bodies)
                write_xml(sid, xml)
                count += 1
    print(f"Tier 1: {count} scenarios")
    return count


# ════════════════════════════════════════════════════════════════════════════
#  TIER 2  30 ordered pairs × 8 configs = 240
# ════════════════════════════════════════════════════════════════════════════
# config: (target_xy, target_euler, distractor_xy)
T2_CONFIGS = [
    ("A", (3.5,  0.0), SIDE_EULER["F"],  (3.5,  2.0)),
    ("B", (3.5,  0.0), SIDE_EULER["F"],  (3.5, -2.0)),
    ("C", (3.5,  0.0), SIDE_EULER["F"],  (5.5,  0.5)),
    ("D", (5.0,  1.5), SIDE_EULER["F"],  (3.0, -0.5)),
    ("E", (3.5,  0.0), SIDE_EULER["R"],  (3.5,  2.0)),
    ("F", (3.5,  0.0), SIDE_EULER["Bk"], (3.5, -2.0)),
    ("G", (3.5,  0.0), SIDE_EULER["L"],  (5.5,  0.5)),
    ("H", (5.0,  1.5), SIDE_EULER["FR"], (3.0, -0.5)),
]

def gen_tier2():
    obj_list = [
        ("C",  "couch"),
        ("T",  "table"),
        ("CH", "chair"),
        ("B",  "bookshelf"),
        ("BX", "cardboard_box"),
        ("TC", "trash_can"),
    ]
    count = 0
    for tgt_code, tgt_xml in obj_list:
        for dst_code, dst_xml in obj_list:
            if tgt_code == dst_code:
                continue
            for cfg_name, tpos_xy, teuler, dpos_xy in T2_CONFIGS:
                sid = f"FT2_{tgt_code}_{dst_code}_{cfg_name}"
                bodies = [
                    (f"{tgt_xml}_target",     tgt_xml, tpos_xy[0], tpos_xy[1], teuler,         f"TARGET: {tgt_xml}"),
                    (f"{dst_xml}_distractor1", dst_xml, dpos_xy[0], dpos_xy[1], deuler(dst_xml), f"DISTRACTOR: {dst_xml}"),
                ]
                xml = make_xml(sid, f"{tgt_xml}+{dst_xml} cfg={cfg_name}", bodies)
                write_xml(sid, xml)
                count += 1
    print(f"Tier 2: {count} scenarios")
    return count


# ════════════════════════════════════════════════════════════════════════════
#  TIER 3  6 × 25 = 150  (all explicitly specified in the plan)
# ════════════════════════════════════════════════════════════════════════════
# Each entry: (scenario_id, tgt_xml, tx, ty, teuler,
#              d1_xml, d1x, d1y,  d2_xml, d2x, d2y,  notes)
# Distractor euler = deuler(xml) unless overridden (not needed here).

F = SIDE_EULER["F"]
FR = SIDE_EULER["FR"]
R = SIDE_EULER["R"]
BR = SIDE_EULER["BR"]
BK = SIDE_EULER["Bk"]
BL = SIDE_EULER["BL"]
L = SIDE_EULER["L"]
FL = SIDE_EULER["FL"]

TIER3_DATA = [
    # ── Couch target ──────────────────────────────────────────────────────
    ("FT3_C_01", "couch", 2.0, 0.0,  F,  "chair",        2.5,  1.5,  "trash_can",    1.5, -0.5, "Near, center"),
    ("FT3_C_02", "couch", 2.0, 0.8,  F,  "chair",        3.0, -0.5,  "cardboard_box",1.5,  0.2, "Near, left"),
    ("FT3_C_03", "couch", 2.0,-0.8,  F,  "table",        3.0,  1.0,  "bookshelf",    4.0,  0.0, "Near, right"),
    ("FT3_C_04", "couch", 2.0, 0.0, BK,  "chair",        2.5,  1.0,  "chair",        2.5, -1.0, "Near, back, confusion flanking"),
    ("FT3_C_05", "couch", 2.0, 0.0,  L,  "bookshelf",    3.0,  1.5,  "trash_can",    2.5, -1.0, "Near, side"),
    ("FT3_C_06", "couch", 4.0, 0.0,  F,  "chair",        3.5,  1.5,  "table",        5.0, -1.0, "Medium, center"),
    ("FT3_C_07", "couch", 4.0, 1.5,  F,  "table",        4.0, -0.5,  "bookshelf",    5.5,  1.0, "Medium, left"),
    ("FT3_C_08", "couch", 4.0,-1.5,  F,  "chair",        2.5,  0.5,  "trash_can",    3.0, -1.0, "Medium, right"),
    ("FT3_C_09", "couch", 4.0, 0.0,  R,  "chair",        3.5,  1.0,  "cardboard_box",2.5,  0.0, "Medium, side"),
    ("FT3_C_10", "couch", 4.0, 0.0, BK,  "table",        3.0, -0.5,  "bookshelf",    5.0,  1.0, "Medium, back"),
    ("FT3_C_11", "couch", 7.5, 0.0,  F,  "chair",        5.0,  1.0,  "table",        6.0, -0.8, "Far, center"),
    ("FT3_C_12", "couch", 7.5, 2.5,  F,  "chair",        5.0,  0.0,  "bookshelf",    6.5,  2.0, "Far, left"),
    ("FT3_C_13", "couch", 7.5,-2.5,  F,  "table",        5.5, -1.0,  "trash_can",    4.0,  0.5, "Far, right"),
    ("FT3_C_14", "couch", 7.5, 0.0, BK,  "chair",        5.0,  1.5,  "chair",        5.0, -1.5, "Far, back, two chair distractors"),
    ("FT3_C_15", "couch", 7.5, 0.0,  L,  "bookshelf",    6.0,  2.0,  "cardboard_box",5.0, -0.5, "Far, side"),
    ("FT3_C_16", "couch", 3.5, 0.0, FR,  "bookshelf",    2.5,  1.5,  "table",        5.0, -1.0, "Medium, FR diagonal"),
    ("FT3_C_17", "couch", 3.5, 0.0, BR,  "chair",        4.0,  1.0,  "table",        2.5,  0.0, "Medium, BR diagonal, confusion"),
    ("FT3_C_18", "couch", 7.5, 0.0, FR,  "trash_can",    2.0,  0.0,  "chair",        6.0, -1.0, "Far, near distractor in front"),
    ("FT3_C_19", "couch", 2.0, 0.0,  R,  "chair",        2.5,  1.0,  "table",        1.5,  0.5, "Near, all brown-wood"),
    ("FT3_C_20", "couch", 6.0, 2.0, FL,  "chair",        4.5,  0.0,  "bookshelf",    7.0,  1.0, "Far-medium, FL diagonal, off-center"),
    ("FT3_C_21", "couch", 2.5, 0.0, FL,  "table",        3.5,  1.5,  "trash_can",    1.5, -0.3, "Near, FL diagonal, clutter near"),
    ("FT3_C_22", "couch", 3.5,-2.0,  F,  "chair",        2.5, -0.5,  "bookshelf",    5.0, -1.5, "Medium, target far right in frame"),
    ("FT3_C_23", "couch", 5.5, 0.0,  F,  "trash_can",    2.0,  0.3,  "table",        4.5,  1.5, "Medium-far, near small distractor"),
    ("FT3_C_24", "couch", 3.5, 0.0,  R,  "trash_can",    2.5,  0.3,  "chair",        5.0,  1.5, "Medium, side, small near occluder"),
    ("FT3_C_25", "couch", 4.5, 0.0,  F,  "chair",        3.0,  1.5,  "table",        6.0, -0.5, "Medium, front, three depth planes"),
    # ── Table target ──────────────────────────────────────────────────────
    ("FT3_T_01", "table", 2.0, 0.0,  F,  "bookshelf",    2.5,  1.5,  "trash_can",    1.5, -0.5, "Near, center"),
    ("FT3_T_02", "table", 2.0, 0.8,  F,  "chair",        1.5, -0.5,  "cardboard_box",2.5,  0.5, "Near, left"),
    ("FT3_T_03", "table", 2.0,-0.8,  F,  "bookshelf",    3.0,  1.0,  "couch",        3.5, -0.5, "Near, right"),
    ("FT3_T_04", "table", 2.0, 0.0,  R,  "chair",        2.5,  1.0,  "trash_can",    1.5, -0.3, "Near, side (narrow face)"),
    ("FT3_T_05", "table", 2.0, 0.0, BK,  "bookshelf",    3.0,  1.5,  "cardboard_box",1.5, -0.5, "Near, back"),
    ("FT3_T_06", "table", 4.0, 0.0,  F,  "bookshelf",    3.5,  1.5,  "chair",        5.0, -1.0, "Medium, center"),
    ("FT3_T_07", "table", 4.0, 1.5,  F,  "couch",        4.0, -0.5,  "trash_can",    3.0,  1.0, "Medium, left"),
    ("FT3_T_08", "table", 4.0,-1.5,  F,  "bookshelf",    3.5,  0.5,  "cardboard_box",5.0, -1.5, "Medium, right"),
    ("FT3_T_09", "table", 4.0, 0.0,  L,  "chair",        3.0, -1.0,  "couch",        5.0,  1.5, "Medium, side"),
    ("FT3_T_10", "table", 4.0, 0.0, BK,  "bookshelf",    5.5,  0.8,  "trash_can",    3.0,  0.0, "Medium, back"),
    ("FT3_T_11", "table", 7.5, 0.0,  F,  "bookshelf",    5.5,  1.0,  "chair",        6.0, -0.8, "Far, center"),
    ("FT3_T_12", "table", 7.5, 2.5,  F,  "bookshelf",    6.0,  1.0,  "couch",        5.0,  2.0, "Far, left"),
    ("FT3_T_13", "table", 7.5,-2.5,  F,  "chair",        5.5, -1.5,  "cardboard_box",6.5,  0.5, "Far, right"),
    ("FT3_T_14", "table", 7.5, 0.0,  R,  "bookshelf",    6.0,  2.0,  "trash_can",    5.0,  0.0, "Far, side"),
    ("FT3_T_15", "table", 7.5, 0.0, BK,  "chair",        5.0,  1.0,  "couch",        9.0,  0.5, "Far, back"),
    ("FT3_T_16", "table", 3.5, 0.0, FR,  "chair",        4.5,  1.5,  "bookshelf",    2.5,  0.5, "Medium, FR diagonal"),
    ("FT3_T_17", "table", 3.5, 0.0, BL,  "bookshelf",    4.5, -1.0,  "couch",        2.5,  1.0, "Medium, BL diagonal"),
    ("FT3_T_18", "table", 7.5, 0.0,  F,  "cardboard_box",2.0,  0.0,  "chair",        5.5,  1.0, "Far table, very near box"),
    ("FT3_T_19", "table", 2.0, 0.0,  L,  "bookshelf",    2.5, -1.0,  "chair",        3.5,  1.0, "Near, side, brown trio"),
    ("FT3_T_20", "table", 5.5,-1.5, FL,  "chair",        4.0, -0.5,  "couch",        7.0, -2.0, "Medium-far, FL diagonal, off-center"),
    ("FT3_T_21", "table", 2.5, 0.0, BR,  "couch",        3.5,  1.5,  "trash_can",    1.5, -0.3, "Near, BR diagonal"),
    ("FT3_T_22", "table", 3.5,-2.0,  F,  "bookshelf",    2.5, -0.5,  "couch",        5.0, -1.5, "Medium, target far right"),
    ("FT3_T_23", "table", 5.5, 0.0,  F,  "trash_can",    2.0,  0.3,  "bookshelf",    4.5, -1.5, "Medium-far, near small distractor"),
    ("FT3_T_24", "table", 3.5, 0.0, FL,  "couch",        5.0,  1.5,  "trash_can",    2.0,  0.0, "Medium, FL diagonal, mixed sizes"),
    ("FT3_T_25", "table", 4.5, 0.0, BK,  "chair",        3.0,  1.0,  "bookshelf",    6.0, -0.5, "Medium, back, confusion trio"),
    # ── Chair target ──────────────────────────────────────────────────────
    ("FT3_CH_01", "chair", 2.0, 0.0,  F,  "couch",        2.5,  1.5,  "trash_can",    1.5, -0.5, "Near, confusion pair"),
    ("FT3_CH_02", "chair", 2.0, 0.8,  F,  "couch",        3.5, -0.5,  "table",        1.5,  0.3, "Near, left"),
    ("FT3_CH_03", "chair", 2.0,-0.8,  F,  "bookshelf",    2.5,  1.0,  "cardboard_box",1.5, -0.2, "Near, right"),
    ("FT3_CH_04", "chair", 2.0, 0.0, BK,  "couch",        2.5,  1.0,  "couch",        2.5, -1.0, "Near, back, flanked by confusion pair"),
    ("FT3_CH_05", "chair", 2.0, 0.0,  L,  "table",        3.0,  1.0,  "trash_can",    2.5, -0.5, "Near, side"),
    ("FT3_CH_06", "chair", 4.0, 0.0,  F,  "couch",        3.5,  1.5,  "table",        5.0, -0.8, "Medium, center"),
    ("FT3_CH_07", "chair", 4.0, 1.5,  F,  "couch",        4.0, -0.5,  "cardboard_box",3.0,  1.0, "Medium, left"),
    ("FT3_CH_08", "chair", 4.0,-1.5,  F,  "bookshelf",    3.5,  0.5,  "trash_can",    5.0, -1.5, "Medium, right"),
    ("FT3_CH_09", "chair", 4.0, 0.0,  R,  "couch",        3.5,  2.0,  "table",        5.0, -0.5, "Medium, side"),
    ("FT3_CH_10", "chair", 4.0, 0.0, BK,  "table",        3.0, -1.0,  "bookshelf",    5.0,  1.0, "Medium, back"),
    ("FT3_CH_11", "chair", 7.5, 0.0,  F,  "couch",        5.0,  1.5,  "table",        6.5, -0.8, "Far, center"),
    ("FT3_CH_12", "chair", 7.5, 2.5,  F,  "couch",        5.0,  0.5,  "bookshelf",    6.0,  2.0, "Far, left"),
    ("FT3_CH_13", "chair", 7.5,-2.5,  F,  "table",        5.5, -1.5,  "trash_can",    6.0,  0.5, "Far, right"),
    ("FT3_CH_14", "chair", 7.5, 0.0, BK,  "couch",        5.0,  1.5,  "couch",        5.0, -1.5, "Far, back, flanked by couches"),
    ("FT3_CH_15", "chair", 7.5, 0.0,  L,  "bookshelf",    6.5,  2.0,  "cardboard_box",5.5, -0.5, "Far, side"),
    ("FT3_CH_16", "chair", 3.5, 0.0, FR,  "couch",        4.5,  1.5,  "bookshelf",    2.5, -0.5, "Medium, FR diagonal, confusion"),
    ("FT3_CH_17", "chair", 3.5, 0.0, BL,  "table",        4.0, -1.0,  "couch",        2.5,  1.0, "Medium, BL diagonal"),
    ("FT3_CH_18", "chair", 7.5, 0.0,  F,  "cardboard_box",2.0,  0.0,  "couch",        5.5,  1.0, "Far, near distractor in front"),
    ("FT3_CH_19", "chair", 2.0, 0.0, FL,  "couch",        3.5,  1.5,  "trash_can",    1.5, -0.3, "Near, FL diagonal"),
    ("FT3_CH_20", "chair", 5.0,-2.0,  R,  "couch",        3.5,  0.0,  "table",        6.5, -1.0, "Medium, off-center side view"),
    ("FT3_CH_21", "chair", 2.5, 0.0, BR,  "table",        3.5, -1.5,  "trash_can",    1.5,  0.3, "Near, BR diagonal"),
    ("FT3_CH_22", "chair", 3.5,-2.0,  F,  "bookshelf",    2.5, -0.5,  "couch",        5.0, -2.0, "Medium, chair far right in frame"),
    ("FT3_CH_23", "chair", 5.5, 0.0,  F,  "cardboard_box",2.0,  0.3,  "table",        4.5, -1.5, "Medium-far, near box in front"),
    ("FT3_CH_24", "chair", 4.0, 0.0, FL,  "couch",        5.5,  1.5,  "table",        2.5, -0.5, "Medium, FL diagonal"),
    ("FT3_CH_25", "chair", 4.5, 0.0,  F,  "couch",        3.0,  2.0,  "bookshelf",    6.0, -0.5, "Medium, confusion pair at different depths"),
    # ── Bookshelf target ──────────────────────────────────────────────────
    ("FT3_B_01", "bookshelf", 2.0, 0.0,  F,  "table",        2.5,  1.5,  "trash_can",    1.5, -0.5, "Near, center"),
    ("FT3_B_02", "bookshelf", 2.0, 0.8,  F,  "chair",        1.5, -0.5,  "couch",        3.0,  1.0, "Near, left"),
    ("FT3_B_03", "bookshelf", 2.0,-0.8,  F,  "table",        2.5,  0.5,  "cardboard_box",1.5, -0.2, "Near, right"),
    ("FT3_B_04", "bookshelf", 2.0, 0.0,  R,  "chair",        2.5,  1.0,  "trash_can",    1.5, -0.3, "Near, side (narrow face)"),
    ("FT3_B_05", "bookshelf", 2.0, 0.0, BK,  "table",        3.0,  1.0,  "couch",        3.5, -1.0, "Near, back"),
    ("FT3_B_06", "bookshelf", 4.0, 0.0,  F,  "table",        3.5,  1.5,  "chair",        5.0, -1.0, "Medium, center, brown-wood distractors"),
    ("FT3_B_07", "bookshelf", 4.0, 1.5,  F,  "chair",        4.0, -0.5,  "trash_can",    3.0,  1.0, "Medium, left"),
    ("FT3_B_08", "bookshelf", 4.0,-1.5,  F,  "table",        5.0,  0.5,  "couch",        4.5, -2.0, "Medium, right"),
    ("FT3_B_09", "bookshelf", 4.0, 0.0,  L,  "table",        3.0,  1.5,  "chair",        5.0, -0.5, "Medium, side"),
    ("FT3_B_10", "bookshelf", 4.0, 0.0, BK,  "chair",        5.0,  0.8,  "cardboard_box",3.0,  0.0, "Medium, back"),
    ("FT3_B_11", "bookshelf", 7.5, 0.0,  F,  "table",        5.5,  1.0,  "chair",        6.0, -0.8, "Far, center, all-brown"),
    ("FT3_B_12", "bookshelf", 7.5, 2.5,  F,  "chair",        6.0,  1.0,  "table",        5.0,  2.5, "Far, left"),
    ("FT3_B_13", "bookshelf", 7.5,-2.5,  F,  "table",        6.0, -1.0,  "trash_can",    5.0,  0.5, "Far, right"),
    ("FT3_B_14", "bookshelf", 7.5, 0.0,  R,  "table",        6.5,  1.5,  "couch",        5.0, -1.0, "Far, side"),
    ("FT3_B_15", "bookshelf", 7.5, 0.0, BK,  "chair",        5.0,  1.0,  "table",        9.0, -0.5, "Far, back"),
    ("FT3_B_16", "bookshelf", 3.5, 0.0, FR,  "table",        4.5,  1.5,  "chair",        2.5, -0.5, "Medium, FR diagonal (very narrow)"),
    ("FT3_B_17", "bookshelf", 3.5, 0.0, BR,  "chair",        4.5, -1.0,  "table",        2.5,  1.0, "Medium, BR diagonal"),
    ("FT3_B_18", "bookshelf", 7.5, 0.0,  F,  "cardboard_box",2.0,  0.5,  "chair",        5.5, -1.0, "Far shelf, near box"),
    ("FT3_B_19", "bookshelf", 2.0, 0.0, FL,  "table",        3.0,  1.5,  "trash_can",    1.5, -0.3, "Near, FL diagonal"),
    ("FT3_B_20", "bookshelf", 5.0, 2.0, BK,  "chair",        4.0,  0.5,  "table",        6.5,  2.5, "Medium, off-center back"),
    ("FT3_B_21", "bookshelf", 2.5, 0.0, BL,  "couch",        4.0,  1.5,  "trash_can",    1.5, -0.3, "Near, BL diagonal"),
    ("FT3_B_22", "bookshelf", 3.5,-2.0,  F,  "chair",        2.5, -0.8,  "table",        5.0, -1.8, "Medium, target far right, brown trio"),
    ("FT3_B_23", "bookshelf", 5.5, 0.0,  F,  "cardboard_box",2.0,  0.3,  "couch",        4.5, -1.5, "Medium-far, near box in foreground"),
    ("FT3_B_24", "bookshelf", 4.0, 0.0, FR,  "couch",        5.5,  1.5,  "trash_can",    2.5,  0.0, "Medium, FR diagonal, cross-material"),
    ("FT3_B_25", "bookshelf", 4.5, 0.0,  L,  "table",        3.0,  1.5,  "chair",        6.0, -0.5, "Medium, all-brown side view"),
    # ── Cardboard Box target ───────────────────────────────────────────────
    ("FT3_BX_01", "cardboard_box", 1.5, 0.0,  F,  "trash_can",    1.5,  1.0,  "table",        3.0,  0.5, "Near, two small objects"),
    ("FT3_BX_02", "cardboard_box", 1.5, 0.8,  F,  "chair",        2.0, -0.5,  "trash_can",    1.0,  0.3, "Near, left"),
    ("FT3_BX_03", "cardboard_box", 1.5,-0.8,  F,  "table",        2.5,  0.5,  "couch",        3.0, -1.5, "Near, right"),
    ("FT3_BX_04", "cardboard_box", 1.5, 0.0,  R,  "table",        2.5,  1.0,  "bookshelf",    3.0,  0.0, "Near, side (slight rotation)"),
    ("FT3_BX_05", "cardboard_box", 1.5, 0.0, BK,  "chair",        2.5,  1.0,  "trash_can",    2.0, -0.5, "Near, back"),
    ("FT3_BX_06", "cardboard_box", 3.5, 0.0,  F,  "trash_can",    2.5,  0.8,  "table",        5.0, -1.0, "Medium, center"),
    ("FT3_BX_07", "cardboard_box", 3.5, 1.5,  F,  "chair",        3.5, -0.5,  "couch",        5.0,  2.0, "Medium, left"),
    ("FT3_BX_08", "cardboard_box", 3.5,-1.5,  F,  "table",        4.0,  0.5,  "bookshelf",    3.0, -2.5, "Medium, right"),
    ("FT3_BX_09", "cardboard_box", 3.5, 0.0,  L,  "chair",        2.5,  1.0,  "trash_can",    4.0, -0.5, "Medium, side"),
    ("FT3_BX_10", "cardboard_box", 3.5, 0.0, BK,  "couch",        5.0,  1.0,  "table",        2.5,  0.0, "Medium, back"),
    ("FT3_BX_11", "cardboard_box", 7.5, 0.0,  F,  "table",        5.5,  1.0,  "chair",        6.5, -0.8, "Far, box tiny in frame"),
    ("FT3_BX_12", "cardboard_box", 7.5, 2.5,  F,  "trash_can",    5.5,  0.5,  "bookshelf",    7.0,  2.0, "Far, left"),
    ("FT3_BX_13", "cardboard_box", 7.5,-2.5,  F,  "couch",        5.0, -0.5,  "chair",        6.5, -2.0, "Far, right"),
    ("FT3_BX_14", "cardboard_box", 7.5, 0.0,  R,  "table",        5.5,  1.5,  "trash_can",    6.0,  0.0, "Far, side"),
    ("FT3_BX_15", "cardboard_box", 7.5, 0.0,  F,  "cardboard_box",7.5,  2.0,  "cardboard_box",7.5, -2.0, "Far, 3 identical boxes"),
    ("FT3_BX_16", "cardboard_box", 3.5, 0.0, FR,  "trash_can",    4.0,  1.0,  "bookshelf",    2.5, -0.5, "Medium, diagonal"),
    ("FT3_BX_17", "cardboard_box", 3.5, 0.0, BK,  "table",        2.5,  1.0,  "chair",        5.0, -0.5, "Medium, back"),
    ("FT3_BX_18", "cardboard_box", 7.5, 0.0,  F,  "trash_can",    2.0,  0.0,  "bookshelf",    5.5,  1.0, "Far box, near can in front"),
    ("FT3_BX_19", "cardboard_box", 2.0, 0.0,  L,  "couch",        3.5,  1.5,  "trash_can",    1.5, -0.3, "Near, side"),
    ("FT3_BX_20", "cardboard_box", 5.0,-1.5, FR,  "table",        4.0, -0.5,  "trash_can",    6.0, -2.0, "Medium-far, off-center FR"),
    ("FT3_BX_21", "cardboard_box", 1.5, 0.0, FL,  "couch",        3.5,  1.5,  "table",        2.5,  0.5, "Near, FL diagonal"),
    ("FT3_BX_22", "cardboard_box", 3.5,-2.0,  F,  "trash_can",    2.5, -0.5,  "couch",        5.0, -2.0, "Medium, target far right"),
    ("FT3_BX_23", "cardboard_box", 5.5, 0.0,  F,  "trash_can",    2.0,  0.3,  "chair",        4.5, -1.5, "Medium-far, near small can"),
    ("FT3_BX_24", "cardboard_box", 3.5, 0.0, BL,  "couch",        5.0,  1.5,  "table",        2.0, -0.5, "Medium, BL diagonal"),
    ("FT3_BX_25", "cardboard_box", 4.5, 0.0,  F,  "table",        3.0,  1.5,  "trash_can",    6.0, -0.5, "Medium, two depth planes"),
    # ── Trash Can target ──────────────────────────────────────────────────
    ("FT3_TC_01", "trash_can", 1.5, 0.0,  F,  "cardboard_box",1.5,  1.0,  "chair",        3.0,  0.5, "Near, small objects together"),
    ("FT3_TC_02", "trash_can", 1.5, 0.8,  F,  "couch",        3.0,  1.5,  "cardboard_box",1.0,  0.0, "Near, left"),
    ("FT3_TC_03", "trash_can", 1.5,-0.8,  F,  "bookshelf",    2.5,  0.5,  "table",        3.0, -1.5, "Near, right"),
    ("FT3_TC_04", "trash_can", 1.5, 0.0,  R,  "cardboard_box",2.0,  1.0,  "chair",        3.0,  0.0, "Near, side"),
    ("FT3_TC_05", "trash_can", 1.5, 0.0, BK,  "couch",        2.5,  1.0,  "table",        3.0, -0.5, "Near, back (rotationally invariant)"),
    ("FT3_TC_06", "trash_can", 3.5, 0.0,  F,  "cardboard_box",2.5,  0.8,  "couch",        5.0, -1.0, "Medium, center"),
    ("FT3_TC_07", "trash_can", 3.5, 1.5,  F,  "chair",        3.0, -0.5,  "bookshelf",    5.0,  2.0, "Medium, left"),
    ("FT3_TC_08", "trash_can", 3.5,-1.5,  F,  "table",        4.0,  0.5,  "cardboard_box",2.5, -1.5, "Medium, right"),
    ("FT3_TC_09", "trash_can", 3.5, 0.0,  L,  "bookshelf",    2.5,  1.5,  "couch",        4.5, -1.0, "Medium, side"),
    ("FT3_TC_10", "trash_can", 3.5, 0.0, BK,  "cardboard_box",5.0,  0.5,  "table",        2.5,  0.0, "Medium, back"),
    ("FT3_TC_11", "trash_can", 7.5, 0.0,  F,  "table",        5.5,  1.0,  "bookshelf",    6.5, -0.8, "Far, can is tiny in frame"),
    ("FT3_TC_12", "trash_can", 7.5, 2.5,  F,  "cardboard_box",5.5,  0.5,  "chair",        7.0,  2.0, "Far, left"),
    ("FT3_TC_13", "trash_can", 7.5,-2.5,  F,  "couch",        5.0, -0.5,  "table",        6.5, -2.0, "Far, right"),
    ("FT3_TC_14", "trash_can", 7.5, 0.0,  R,  "bookshelf",    6.0,  1.5,  "cardboard_box",5.5,  0.0, "Far, side"),
    ("FT3_TC_15", "trash_can", 7.5, 0.0,  F,  "trash_can",    7.5,  2.0,  "trash_can",    7.5, -2.0, "Far, 3 identical cans"),
    ("FT3_TC_16", "trash_can", 3.5, 0.0,  F,  "bookshelf",    4.5,  1.5,  "cardboard_box",2.5, -0.5, "Medium, small vs tall confusion"),
    ("FT3_TC_17", "trash_can", 3.5, 0.0, BK,  "chair",        2.5,  1.0,  "table",        5.0, -0.5, "Medium, back (rotationally invariant)"),
    ("FT3_TC_18", "trash_can", 7.5, 0.0,  F,  "cardboard_box",2.0,  0.5,  "table",        5.5, -1.0, "Far can, near box in front"),
    ("FT3_TC_19", "trash_can", 2.0, 0.0,  F,  "couch",        3.5,  1.5,  "bookshelf",    3.0, -1.0, "Near, large distractors dwarfing can"),
    ("FT3_TC_20", "trash_can", 5.0, 2.0,  F,  "cardboard_box",4.0,  0.5,  "chair",        6.0,  2.5, "Medium-far, off-center"),
    ("FT3_TC_21", "trash_can", 1.5, 0.0,  F,  "chair",        2.5,  1.5,  "couch",        4.0, -1.0, "Near, tiny can vs large confusion"),
    ("FT3_TC_22", "trash_can", 3.5,-2.0,  F,  "cardboard_box",2.5, -0.5,  "bookshelf",    5.0, -2.0, "Medium, far right"),
    ("FT3_TC_23", "trash_can", 5.5, 0.0,  F,  "cardboard_box",2.0,  0.3,  "couch",        4.5, -1.5, "Medium-far, near box in front"),
    ("FT3_TC_24", "trash_can", 3.5, 0.0, FR,  "table",        5.0,  1.5,  "cardboard_box",2.5,  0.0, "Medium, FR diagonal"),
    ("FT3_TC_25", "trash_can", 4.5, 0.0,  F,  "cardboard_box",3.0,  1.5,  "bookshelf",    6.0, -0.5, "Medium, two depth planes"),
]

def gen_tier3():
    count = 0
    for row in TIER3_DATA:
        sid, tgt_xml, tx, ty, teuler, d1_xml, d1x, d1y, d2_xml, d2x, d2y, notes = row
        bodies = [
            (f"{tgt_xml}_target",     tgt_xml, tx,  ty,  teuler,         f"TARGET: {tgt_xml}"),
            (f"{d1_xml}_distractor1", d1_xml,  d1x, d1y, deuler(d1_xml), f"DISTRACTOR 1: {d1_xml}"),
            (f"{d2_xml}_distractor2", d2_xml,  d2x, d2y, deuler(d2_xml), f"DISTRACTOR 2: {d2_xml}"),
        ]
        xml = make_xml(sid, notes, bodies)
        write_xml(sid, xml)
        count += 1
    print(f"Tier 3: {count} scenarios")
    return count


# ════════════════════════════════════════════════════════════════════════════
#  TIER 4 — Dense Scenes 4-6 objects (100 scenarios)
# ════════════════════════════════════════════════════════════════════════════
# Each entry: (scenario_id, description, [(xml, x, y, euler, is_target), ...])
# Bodies format: (xml_name, x, y, euler_z, label)  label = "TARGET" or name

def _t4_body(xml, x, y, euler=None, idx=None):
    """Helper: build a body tuple for Tier 4 (no single designated target)."""
    e = euler if euler is not None else deuler(xml)
    name = f"{xml}_{idx}" if idx is not None else xml
    return (name, xml, x, y, e, f"OBJECT: {xml}")

def t4_make(sid, desc, obj_list):
    """obj_list: list of (xml, x, y) — euler auto-assigned via deuler."""
    bodies = []
    counts = {}
    for xml, x, y in obj_list:
        counts[xml] = counts.get(xml, 0) + 1
        bname = f"{xml}_{counts[xml]}"
        bodies.append((bname, xml, x, y, deuler(xml), f"OBJECT: {xml}"))
    return make_xml(sid, desc, bodies)

# All 16 all-objects scenes (FT4_01–16) — explicitly specified
T4_ALL_OBJ = [
    ("FT4_01", "All 6, cluster 4-6m",          [("couch",5,2),("table",4,0),("chair",3,1),("bookshelf",6,-1),("cardboard_box",2,0.5),("trash_can",3,-1)]),
    ("FT4_02", "All 6, arc left-to-right",      [("couch",4,3),("table",4,1),("chair",4,-1),("bookshelf",4,-3),("cardboard_box",3,0),("trash_can",5,0)]),
    ("FT4_03", "All 6, near-far line",           [("trash_can",1.5,0),("cardboard_box",2.5,0.5),("chair",4,0),("table",5.5,0.5),("bookshelf",7,-1),("couch",8,0)]),
    ("FT4_04", "All 6, living room offset left", [("couch",4,2),("table",3,1.5),("chair",4,3),("bookshelf",5,1),("cardboard_box",2.5,0.5),("trash_can",2,2)]),
    ("FT4_05", "All 6, storage corner",          [("bookshelf",3,-3),("cardboard_box",2,-2),("trash_can",2,-3),("couch",5,1),("table",4,-1),("chair",3,0)]),
    ("FT4_06", "All 6, office layout right",     [("table",3,-2),("chair",4,-3),("bookshelf",5,-2),("couch",3,1),("cardboard_box",2,-0.5),("trash_can",1.5,0)]),
    ("FT4_07", "All 6, wide spread",             [("couch",9,3),("table",8,-2),("chair",7,0),("bookshelf",9,-3),("cardboard_box",2,0),("trash_can",3,2)]),
    ("FT4_08", "All 6, tight cluster medium",    [("couch",4,0),("table",4.5,1.5),("chair",3,1.5),("bookshelf",5,-0.5),("cardboard_box",3.5,0.5),("trash_can",3,-0.5)]),
    ("FT4_09", "All 6, near small + far large",  [("cardboard_box",1.5,0.5),("trash_can",1.5,-0.5),("chair",5,1),("table",6,0),("couch",7,-1),("bookshelf",6,2)]),
    ("FT4_10", "All 6, Z-shaped arrangement",    [("trash_can",2,-2),("cardboard_box",2,0),("chair",4,1),("table",5,0),("couch",7,2),("bookshelf",7,-1)]),
    ("FT4_11", "All 6, living room right-side",  [("couch",4,3),("table",3,2),("chair",5,2.5),("bookshelf",6,1),("cardboard_box",2,1),("trash_can",3,3)]),
    ("FT4_12", "All 6, warehouse scatter",       [("bookshelf",3,0),("cardboard_box",2.5,-1.5),("trash_can",2,1),("table",5,-2),("chair",6,1),("couch",8,0)]),
    ("FT4_13", "All 6, near cluster stress test",[("couch",3,1),("table",2.5,-0.5),("chair",2,1.5),("bookshelf",4,-0.5),("cardboard_box",1.5,0),("trash_can",1.5,-0.8)]),
    ("FT4_14", "All 6, depth stagger center",    [("trash_can",1.5,0),("cardboard_box",2.5,-0.5),("chair",3.5,0.5),("bookshelf",5,0),("table",6.5,-0.5),("couch",8,0.5)]),
    ("FT4_15", "All 6, far scatter",             [("couch",7,2),("table",8,-1),("chair",9,1),("bookshelf",7,-2),("cardboard_box",6,0),("trash_can",5,2)]),
    ("FT4_16", "All 6, asymmetric left-heavy",   [("couch",5,-1),("table",4,-2),("chair",3,-2.5),("bookshelf",6,-3),("cardboard_box",2,-1),("trash_can",3,1)]),
]

# ── Tier 4 themed: Living Room (FT4_17–37), 21 scenes ──────────────────────
# Core: couch + table + chair + bookshelf; some add box/trash_can
T4_LIVING = [
    # Near cluster (7)
    ("FT4_17", "Living room near cluster 1",     [("couch",2,0),("table",2.5,1.5),("chair",3,0.5),("bookshelf",2,-1.5)]),
    ("FT4_18", "Living room near cluster 2",     [("couch",2,0.5),("table",3,0),("chair",2,-0.8),("bookshelf",3,-1.5)]),
    ("FT4_19", "Living room near cluster 3",     [("couch",2.5,0),("table",2,-1),("chair",1.5,0.5),("bookshelf",3,1.5)]),
    ("FT4_20", "Living room near 5 objects",     [("couch",2,1),("table",2,-0.5),("chair",3,0),("bookshelf",1.5,-1),("cardboard_box",2.5,-0.5)]),
    ("FT4_21", "Living room near 5 w/ can",      [("couch",2.5,-0.5),("table",2,1),("chair",1.5,-0.3),("bookshelf",3.5,0),("trash_can",1.5,1)]),
    ("FT4_22", "Living room near wide",          [("couch",2,0),("table",3,-1.5),("chair",3,1.5),("bookshelf",2,2)]),
    ("FT4_23", "Living room near offset",        [("couch",2.5,0),("table",2,1.5),("chair",1.5,0),("bookshelf",3,1)]),
    # Medium spread (7)
    ("FT4_24", "Living room medium spread 1",    [("couch",4,0),("table",4,2),("chair",3,0.5),("bookshelf",5,-1)]),
    ("FT4_25", "Living room medium spread 2",    [("couch",4,-1),("table",3,1),("chair",5,0.5),("bookshelf",4,2)]),
    ("FT4_26", "Living room medium spread 3",    [("couch",4,0),("table",3,-1.5),("chair",5,1.5),("bookshelf",3,1.5)]),
    ("FT4_27", "Living room medium 5 objects",   [("couch",4,1),("table",3,-0.5),("chair",5,0),("bookshelf",4,-2),("cardboard_box",2,0)]),
    ("FT4_28", "Living room medium 5 w/ can",    [("couch",4,-0.5),("table",3,1.5),("chair",4,2),("bookshelf",5,-0.5),("trash_can",2.5,0)]),
    ("FT4_29", "Living room medium L-shape",     [("couch",3,2),("table",4,0),("chair",3,-1.5),("bookshelf",5,2)]),
    ("FT4_30", "Living room medium diagonal",    [("couch",5,2),("table",4,0.5),("chair",3,-0.5),("bookshelf",5,-2)]),
    # Wide/far (7)
    ("FT4_31", "Living room far spread 1",       [("couch",7,1),("table",6,-1),("chair",8,0.5),("bookshelf",7,-2)]),
    ("FT4_32", "Living room far spread 2",       [("couch",8,0),("table",7,2),("chair",6,-1),("bookshelf",8,-2.5)]),
    ("FT4_33", "Living room far spread 3",       [("couch",7,-1),("table",8,1.5),("chair",9,0),("bookshelf",6,2)]),
    ("FT4_34", "Living room far 5 objects",      [("couch",7,0),("table",6,2),("chair",8,-1),("bookshelf",7,-2.5),("cardboard_box",5,0)]),
    ("FT4_35", "Living room far 5 w/ can",       [("couch",7,1.5),("table",8,-0.5),("chair",6,0),("bookshelf",8,2.5),("trash_can",5,1)]),
    ("FT4_36", "Living room far scattered",      [("couch",9,2),("table",7,-1.5),("chair",8,1),("bookshelf",9,-2)]),
    ("FT4_37", "Living room far center line",    [("couch",8,0),("table",6,0.5),("chair",5,-0.5),("bookshelf",7,1.5)]),
]

# ── Storage Area (FT4_38–58): bookshelf + box + trash_can + chair ± table ──
T4_STORAGE = [
    # Near cluster (7)
    ("FT4_38", "Storage near cluster 1",    [("bookshelf",2,0),("cardboard_box",1.5,0.8),("trash_can",1.5,-0.5),("chair",2.5,1.5)]),
    ("FT4_39", "Storage near cluster 2",    [("bookshelf",2,-1),("cardboard_box",1.5,0),("trash_can",2,1),("chair",3,-0.5)]),
    ("FT4_40", "Storage near cluster 3",    [("bookshelf",3,0),("cardboard_box",2,1),("trash_can",2,-0.8),("chair",1.5,0)]),
    ("FT4_41", "Storage near 5 w/ table",   [("bookshelf",2.5,0),("cardboard_box",1.5,0.5),("trash_can",1.5,-0.5),("chair",3,1),("table",3,-1.5)]),
    ("FT4_42", "Storage near wide",         [("bookshelf",2,1.5),("cardboard_box",1.5,-0.5),("trash_can",2,-1.5),("chair",3,0.5)]),
    ("FT4_43", "Storage near tight",        [("bookshelf",2,0),("cardboard_box",1.5,0.4),("trash_can",1.5,-0.4),("chair",2.5,-1)]),
    ("FT4_44", "Storage near offset",       [("bookshelf",3,-1),("cardboard_box",2,-0.5),("trash_can",1.5,0.5),("chair",2,1)]),
    # Medium spread (7)
    ("FT4_45", "Storage medium spread 1",   [("bookshelf",4,0),("cardboard_box",3,1),("trash_can",3,-1),("chair",5,0.5)]),
    ("FT4_46", "Storage medium spread 2",   [("bookshelf",4,-1.5),("cardboard_box",2.5,0),("trash_can",3.5,1),("chair",5,-0.5)]),
    ("FT4_47", "Storage medium spread 3",   [("bookshelf",3.5,1),("cardboard_box",2.5,-0.5),("trash_can",4.5,-1),("chair",4.5,2)]),
    ("FT4_48", "Storage medium 5 w/ table", [("bookshelf",4,0),("cardboard_box",2.5,0.5),("trash_can",3,-1),("chair",5,1.5),("table",4.5,-2)]),
    ("FT4_49", "Storage medium L-shape",    [("bookshelf",4,2),("cardboard_box",3,0),("trash_can",2.5,-0.5),("chair",5,0.5)]),
    ("FT4_50", "Storage medium cluster",    [("bookshelf",3.5,0.5),("cardboard_box",2.5,-0.5),("trash_can",4,-0.5),("chair",3,2)]),
    ("FT4_51", "Storage medium diagonal",   [("bookshelf",5,-1),("cardboard_box",3.5,0),("trash_can",2,0.5),("chair",4,1.5)]),
    # Wide/far (7)
    ("FT4_52", "Storage far spread 1",      [("bookshelf",7,0),("cardboard_box",5,-0.5),("trash_can",6,1),("chair",8,-1)]),
    ("FT4_53", "Storage far spread 2",      [("bookshelf",7,-2),("cardboard_box",5.5,0.5),("trash_can",6.5,-0.5),("chair",8,1)]),
    ("FT4_54", "Storage far spread 3",      [("bookshelf",8,1),("cardboard_box",6,-1),("trash_can",7,2),("chair",9,-0.5)]),
    ("FT4_55", "Storage far 5 w/ table",    [("bookshelf",7,0),("cardboard_box",5,1),("trash_can",6,-1),("chair",8,0.5),("table",7,-2.5)]),
    ("FT4_56", "Storage far scatter",       [("bookshelf",8,-1),("cardboard_box",6,0.5),("trash_can",5,2),("chair",9,-2)]),
    ("FT4_57", "Storage far center",        [("bookshelf",7,0.5),("cardboard_box",5.5,-0.5),("trash_can",6.5,1.5),("chair",8,-0.5)]),
    ("FT4_58", "Storage far offset",        [("bookshelf",6.5,2),("cardboard_box",5,-0.5),("trash_can",7,-1.5),("chair",7.5,0.5)]),
]

# ── Office Cluster (FT4_59–79): table + chair + bookshelf + trash_can ± couch
T4_OFFICE = [
    # Near cluster (7)
    ("FT4_59", "Office near cluster 1",     [("table",2,0),("chair",2.5,1),("bookshelf",3,-0.5),("trash_can",1.5,0.5)]),
    ("FT4_60", "Office near cluster 2",     [("table",2.5,-0.5),("chair",2,1),("bookshelf",2,-1.5),("trash_can",1.5,-0.3)]),
    ("FT4_61", "Office near cluster 3",     [("table",2,0.5),("chair",1.5,-0.5),("bookshelf",3,0.5),("trash_can",2.5,-1)]),
    ("FT4_62", "Office near 5 w/ couch",    [("table",2.5,0),("chair",2,-0.5),("bookshelf",3.5,0.5),("trash_can",1.5,0.5),("couch",4.5,-1)]),
    ("FT4_63", "Office near wide",          [("table",2,-1.5),("chair",3,0.5),("bookshelf",2,1.5),("trash_can",1.5,-0.5)]),
    ("FT4_64", "Office near tight",         [("table",2,0),("chair",2.5,0.8),("bookshelf",1.5,-0.8),("trash_can",1.5,0.5)]),
    ("FT4_65", "Office near offset right",  [("table",2,-1),("chair",2.5,-1.5),("bookshelf",3,-2),("trash_can",1.5,-0.8)]),
    # Medium spread (7)
    ("FT4_66", "Office medium spread 1",    [("table",4,0),("chair",3.5,1.5),("bookshelf",5,-0.5),("trash_can",3,-0.5)]),
    ("FT4_67", "Office medium spread 2",    [("table",3.5,-1),("chair",4.5,0.5),("bookshelf",4,2),("trash_can",2.5,0)]),
    ("FT4_68", "Office medium spread 3",    [("table",4,1),("chair",3,-0.5),("bookshelf",5,0),("trash_can",4.5,-1.5)]),
    ("FT4_69", "Office medium 5 w/ couch",  [("table",4,0),("chair",3.5,1),("bookshelf",5,-1),("trash_can",3,-1),("couch",5.5,2)]),
    ("FT4_70", "Office medium diagonal",    [("table",5,-1),("chair",4,0),("bookshelf",3,1),("trash_can",2.5,-0.5)]),
    ("FT4_71", "Office medium L-shape",     [("table",4,2),("chair",3,0),("bookshelf",4,-1.5),("trash_can",2,1)]),
    ("FT4_72", "Office medium cluster",     [("table",3.5,0.5),("chair",4.5,-0.5),("bookshelf",4,1.5),("trash_can",3,-1)]),
    # Wide/far (7)
    ("FT4_73", "Office far spread 1",       [("table",7,0),("chair",6,1.5),("bookshelf",8,-0.5),("trash_can",5.5,0.5)]),
    ("FT4_74", "Office far spread 2",       [("table",7,-1.5),("chair",8,0.5),("bookshelf",6,2),("trash_can",7,2)]),
    ("FT4_75", "Office far spread 3",       [("table",8,1),("chair",7,-1),("bookshelf",9,0),("trash_can",6.5,-1.5)]),
    ("FT4_76", "Office far 5 w/ couch",     [("table",7,0),("chair",6.5,2),("bookshelf",8,-1),("trash_can",6,-1),("couch",8,2.5)]),
    ("FT4_77", "Office far scattered",      [("table",9,-1),("chair",8,1.5),("bookshelf",7,0),("trash_can",5.5,2)]),
    ("FT4_78", "Office far center",         [("table",7,0.5),("chair",6,-0.5),("bookshelf",8,1),("trash_can",7,-1.5)]),
    ("FT4_79", "Office far offset",         [("table",7.5,2),("chair",6.5,0.5),("bookshelf",8.5,-1),("trash_can",6,2.5)]),
]

# ── Mixed Spread (FT4_80–100): varied 4-5 object combinations ─────────────
T4_MIXED = [
    # Near (7)
    ("FT4_80", "Mixed near 1",   [("couch",2.5,0),("trash_can",1.5,1),("bookshelf",3,-1),("cardboard_box",1.5,-0.5)]),
    ("FT4_81", "Mixed near 2",   [("table",2,0),("trash_can",1.5,-0.5),("chair",3,1),("couch",3.5,-1)]),
    ("FT4_82", "Mixed near 3",   [("bookshelf",2.5,0),("cardboard_box",1.5,0.5),("couch",3,1.5),("chair",1.5,-0.8)]),
    ("FT4_83", "Mixed near 4",   [("chair",2,0),("table",3,1),("trash_can",1.5,-0.5),("bookshelf",3.5,-1)]),
    ("FT4_84", "Mixed near 5 obj",[("couch",2,0.5),("table",2.5,-1),("cardboard_box",1.5,0),("trash_can",3,1),("bookshelf",3.5,-1.5)]),
    ("FT4_85", "Mixed near 6",   [("trash_can",1.5,0.5),("chair",2.5,-0.5),("table",3,1.5),("bookshelf",2,-1.5)]),
    ("FT4_86", "Mixed near 7",   [("couch",2,0),("cardboard_box",1.5,0.8),("chair",2.5,-1),("table",3.5,0.5)]),
    # Medium (7)
    ("FT4_87",  "Mixed medium 1",  [("couch",4,0),("cardboard_box",3,1.5),("trash_can",3,-1),("bookshelf",5,0.5)]),
    ("FT4_88",  "Mixed medium 2",  [("table",4,1),("chair",3,-0.5),("trash_can",5,-1),("couch",5,2)]),
    ("FT4_89",  "Mixed medium 3",  [("bookshelf",4,-1),("couch",3.5,1.5),("cardboard_box",2.5,0),("trash_can",4.5,0.5)]),
    ("FT4_90",  "Mixed medium 4",  [("chair",4,0),("bookshelf",5,-1.5),("couch",3,1.5),("cardboard_box",2.5,-0.5)]),
    ("FT4_91",  "Mixed medium 5 obj",[("table",4,0),("trash_can",3,-0.5),("couch",5,1.5),("chair",3.5,1),("cardboard_box",2.5,0)]),
    ("FT4_92",  "Mixed medium 6",  [("bookshelf",4.5,1),("table",3.5,-1),("trash_can",5,0),("chair",3,1.5)]),
    ("FT4_93",  "Mixed medium 7",  [("couch",4,1),("cardboard_box",2.5,-0.5),("table",5,-1),("trash_can",3.5,2)]),
    # Wide/far (7)
    ("FT4_94",  "Mixed far 1",     [("couch",7,0),("trash_can",5.5,1.5),("table",8,-1),("bookshelf",6,-2)]),
    ("FT4_95",  "Mixed far 2",     [("chair",7,-1),("cardboard_box",5.5,0.5),("couch",8.5,1),("table",7,2)]),
    ("FT4_96",  "Mixed far 3",     [("bookshelf",7,1),("couch",8,-0.5),("trash_can",6,2),("chair",8.5,-2)]),
    ("FT4_97",  "Mixed far 4",     [("table",7,0),("chair",6.5,2),("trash_can",8,-1.5),("bookshelf",7,-2)]),
    ("FT4_98",  "Mixed far 5 obj", [("couch",7.5,1),("table",8.5,-0.5),("chair",6,0),("trash_can",7,-2),("cardboard_box",5.5,1.5)]),
    ("FT4_99",  "Mixed far 6",     [("bookshelf",8,0),("couch",7,2.5),("chair",9,-1),("cardboard_box",6,0.5)]),
    ("FT4_100", "Mixed far 7",     [("trash_can",6.5,1),("table",8,-2),("couch",7.5,0),("bookshelf",8.5,2)]),
]

def gen_tier4():
    count = 0
    all_groups = T4_ALL_OBJ + T4_LIVING + T4_STORAGE + T4_OFFICE + T4_MIXED
    for sid, desc, obj_list in all_groups:
        xml = t4_make(sid, desc, obj_list)
        write_xml(sid, xml)
        count += 1
    print(f"Tier 4: {count} scenarios")
    return count


# ════════════════════════════════════════════════════════════════════════════
#  TIER 5 — Edge Cases (80 scenarios)
# ════════════════════════════════════════════════════════════════════════════

def t5_make(sid, desc, tgt_xml, tx, ty, teuler, extras=None):
    """extras: list of (xml, x, y) distractors."""
    bodies = [(f"{tgt_xml}_target", tgt_xml, tx, ty, teuler, f"TARGET: {tgt_xml}")]
    if extras:
        counts = {}
        for xml, x, y in extras:
            counts[xml] = counts.get(xml, 0) + 1
            bodies.append((f"{xml}_distractor{counts[xml]}", xml, x, y, deuler(xml), f"DISTRACTOR: {xml}"))
    return make_xml(sid, desc, bodies)

# §9.1 Extreme Off-Center (16 scenarios, FT5_01–16)
TIER5_OFFCENTER = [
    ("FT5_01", "chair", 3.0,  2.8, F,  None,                           "Extreme left ~43°"),
    ("FT5_02", "chair", 3.0, -2.8, F,  None,                           "Extreme right ~43°"),
    ("FT5_03", "couch", 3.5,  3.2, F,  None,                           "Wide object near left edge"),
    ("FT5_04", "couch", 3.5, -3.2, F,  None,                           "Wide object near right edge"),
    ("FT5_05", "bookshelf", 3.0, 2.8, R, None,                         "Narrow side near left edge"),
    ("FT5_06", "trash_can", 2.5,-2.5, F, None,                         "Small object near right edge"),
    ("FT5_07", "cardboard_box", 2.5, 2.5, F, None,                     "Small object near left edge"),
    ("FT5_08", "table", 4.0,  3.5, R,  None,                           "Long side near left edge"),
    ("FT5_09", "chair", 2.0, -2.5, BK, None,                           "Back view, extreme right"),
    ("FT5_10", "bookshelf", 4.0, -3.5, F, None,                        "Tall narrow object, far right"),
    ("FT5_11", "couch", 2.5,  3.0, R,  None,                           "Side view, extreme left"),
    ("FT5_12", "table", 3.5, -3.0, FR, None,                           "Diagonal view, near right edge"),
    ("FT5_13", "trash_can", 2.0, 2.4, F, None,                         "Very small object at extreme left"),
    ("FT5_14", "cardboard_box", 3.0,-2.7, L, None,                     "Low flat object at right edge"),
    ("FT5_15", "chair", 5.0,  4.0, F,  None,                           "Far object at extreme left"),
    ("FT5_16", "bookshelf", 5.0, -4.0, BK, None,                       "Far back view, extreme right"),
]

# §9.2 Same-Class Multiples (12 scenarios, FT5_17–28)
# Each entry: (sid, target_xml, positions_list, desc)
TIER5_MULTICLASS = [
    ("FT5_17", "chair",        [(2,0),(3,1.5),(3,-1.5)],  "Three chairs at varying depths"),
    ("FT5_18", "chair",        [(2.5,1.5),(2.5,-1.5)],    "Symmetric pair left/right"),
    ("FT5_19", "bookshelf",    [(3,0),(5,-1)],             "Front and far"),
    ("FT5_20", "trash_can",    [(2,0),(2,1.5),(2,-1.5)],   "Row of cans"),
    ("FT5_21", "cardboard_box",[(2,0.5),(2,-0.5),(3,1),(3,-1)], "Box cluster"),
    ("FT5_22", "couch",        [(4,2),(4,-2)],             "Symmetric pair, wide objects"),
    ("FT5_23", "chair",        [(2,0),(5,0)],              "Depth-stacked same class"),
    ("FT5_24", "table",        [(3,1.5),(4,-1.0)],         "Two tables, crossed lateral"),
    ("FT5_25", "bookshelf",    [(2.5,0),(4,1.5),(4,-1.5)], "Three, near + flanking far"),
    ("FT5_26", "trash_can",    [(2,1),(3,-1)],             "Asymmetric pair"),
    ("FT5_27", "cardboard_box",[(1.5,0),(3.5,1.5)],        "Near tiny and medium"),
    ("FT5_28", "couch",        [(4,0),(5,2.5),(5,-2.5)],   "Center + flanking couches"),
]

# §9.3 Close-Quarters with Clutter (12 scenarios, FT5_29–40)
# (sid, target_xml, tx, ty, teuler, clutter_list, desc)
TIER5_CLUTTER = [
    ("FT5_29", "trash_can",    1.5, 0.0, F,  [("cardboard_box",1.2,0.4),("cardboard_box",1.2,-0.4)], "Boxes flank can"),
    ("FT5_30", "cardboard_box",1.2, 0.0, F,  [("trash_can",1.2,0.5),("trash_can",1.2,-0.5)],         "Cans flank box"),
    ("FT5_31", "chair",        1.8, 0.0, F,  [("cardboard_box",1.5,0.4),("trash_can",1.5,-0.3)],      "Small clutter in front"),
    ("FT5_32", "table",        2.0, 0.0, F,  [("cardboard_box",1.5,0.3),("trash_can",1.5,-0.3),("chair",2.5,1.0)], "Clutter in front + chair"),
    ("FT5_33", "couch",        2.0, 0.0, F,  [("chair",1.5,0.5),("chair",1.5,-0.5)],                 "Chairs in foreground"),
    ("FT5_34", "bookshelf",    2.0, 0.0, F,  [("cardboard_box",1.5,0.3),("cardboard_box",1.5,-0.3)],  "Boxes partially in front"),
    ("FT5_35", "chair",        1.5, 0.0, BK, [("couch",2.5,0.8),("table",2.5,-0.8)],                 "Large clutter flanking"),
    ("FT5_36", "trash_can",    1.5, 0.5, F,  [("cardboard_box",1.2,-0.2),("chair",2.5,1.5)],          "Off-center with near box"),
    ("FT5_37", "table",        1.8, 0.0, R,  [("bookshelf",2.5,1.0),("trash_can",1.2,-0.3)],         "Narrow side, clutter"),
    ("FT5_38", "couch",        2.0, 0.0, L,  [("bookshelf",2.5,-1.0),("cardboard_box",1.5,0.5)],      "Side view, tall clutter"),
    ("FT5_39", "bookshelf",    1.8, 0.0, R,  [("couch",2.5,1.5),("chair",2.5,-0.8)],                 "Narrow side, large flankers"),
    ("FT5_40", "cardboard_box",1.5, 0.0, F,  [("chair",2.0,0.8),("bookshelf",3.0,0)],                "Small box dwarfed by tall objects"),
]

# §9.4 Very Far (20 scenarios, FT5_41–60)
# (sid, target_xml, tx, ty, teuler, distractor_list, desc)
TIER5_VERYFAR = [
    ("FT5_41", "couch",        10, 0,    F,  [("chair",7,0.8)],                           "Couch at 10m"),
    ("FT5_42", "table",        10, 0,    F,  [("bookshelf",7,-0.8)],                      "Table at 10m"),
    ("FT5_43", "chair",        10, 0,    F,  [("couch",7,1.5)],                           "Chair at 10m"),
    ("FT5_44", "bookshelf",    10, 0,    F,  [("table",7,-1.0)],                          "Bookshelf at 10m"),
    ("FT5_45", "cardboard_box", 9, 0,    F,  [("table",6,0.5)],                           "Box tiny at 9m"),
    ("FT5_46", "trash_can",     9, 0,    F,  [("bookshelf",6,0.5)],                       "Can tiny at 9m"),
    ("FT5_47", "chair",        10, 2.5,  F,  [("table",7,0)],                             "Far, off-center left"),
    ("FT5_48", "couch",        10,-2.5,  F,  [("chair",7,-1.0)],                          "Far, off-center right"),
    ("FT5_49", "bookshelf",    10, 0,    F,  [("bookshelf",7,1.5)],                       "Two bookshelves, far is GT"),
    ("FT5_50", "trash_can",     9, 2.0,  F,  [("cardboard_box",6,0.5),("cardboard_box",8,2.5)], "Far can, two box distractors"),
    ("FT5_51", "couch",        11, 0,    F,  [("table",7,0)],                             "Couch at 11m"),
    ("FT5_52", "chair",         9,-2.0,  F,  [("couch",6,-0.5)],                          "Far chair, off-center right"),
    ("FT5_53", "table",        10, 1.5,  F,  [("chair",7,0.5)],                           "Far table left, chair nearer"),
    ("FT5_54", "cardboard_box", 8, 0,    F,  [("trash_can",5,0.5),("table",7,-1.0)],      "Box at 8m, two distractors"),
    ("FT5_55", "trash_can",    10, 0,    F,  [("couch",7,1.0)],                           "Can at 10m"),
    ("FT5_56", "bookshelf",     9, 2.5,  F,  [("couch",6,0)],                             "Bookshelf far left, couch nearer"),
    ("FT5_57", "chair",        11, 0,    F,  [("bookshelf",8,-1.5)],                      "Chair at 11m"),
    ("FT5_58", "table",         9,-2.5,  F,  [("trash_can",5,0)],                         "Far table right edge"),
    ("FT5_59", "couch",         8, 0,   BK,  [("chair",6,1.0)],                           "Far + back view (double hard)"),
    ("FT5_60", "bookshelf",     8, 0,    R,  [("table",5,-1.0)],                          "Far + narrow side (double hard)"),
]

# §9.5 Extra edge cases to reach 80 total (FT5_61–80)
TIER5_EXTRA = [
    ("FT5_61", "couch",        3.0, 2.5, FR, [("table",5.0,0)],                           "FR diagonal extreme left"),
    ("FT5_62", "chair",        3.0,-2.5, BL, [("bookshelf",4.5,0)],                       "BL diagonal extreme right"),
    ("FT5_63", "bookshelf",    2.5, 0,   BR, [("couch",4,1.5)],                           "BR diagonal, narrow"),
    ("FT5_64", "table",        2.5,-1.5,  R, [("chair",3.5,-2.5)],                        "Side view, both offset right"),
    ("FT5_65", "trash_can",    1.5, 0,    F, [("trash_can",1.5,0.7),("trash_can",1.5,-0.7)], "3 cans in a row near"),
    ("FT5_66", "cardboard_box",1.2, 0,    F, [("cardboard_box",1.2,0.5),("cardboard_box",1.2,-0.5),("cardboard_box",2.5,0)], "4 boxes very close"),
    ("FT5_67", "chair",        4.0, 0,   FL, [("couch",3.0,1.5),("bookshelf",5.5,-1)],    "FL diagonal medium"),
    ("FT5_68", "table",        6.0, 3.0,  L, [("couch",4.5,1.5)],                         "Far off-center side view"),
    ("FT5_69", "bookshelf",    5.0,-3.5, BK, None,                                         "Far back, edge clip right"),
    ("FT5_70", "couch",        2.0, 0,   BL, [("chair",3.5,1.5),("table",3,-0.5)],        "Near BL diagonal confusion"),
    ("FT5_71", "chair",        1.8, 0,   FR, [("couch",2.5,1),("table",2.5,-1)],          "Near FR confusion flanking"),
    ("FT5_72", "trash_can",    1.5, 2.0,  F, [("cardboard_box",1.5,1.2),("table",3,2)],   "Small objects clustered left"),
    ("FT5_73", "cardboard_box",1.5,-2.0,  F, [("trash_can",1.5,-1.3),("bookshelf",3,-2)], "Small objects clustered right"),
    ("FT5_74", "couch",        6.5, 0,   BK, [("chair",5,1.5),("bookshelf",5.5,-1)],      "Medium-far back, confusion"),
    ("FT5_75", "table",        3.5, 0,   BL, [("bookshelf",5,-1.5),("chair",2.5,1)],      "BL diagonal medium"),
    ("FT5_76", "bookshelf",    3.0, 2.0, FL, [("table",4.5,0.5)],                         "FL diagonal left of center"),
    ("FT5_77", "chair",        2.0, 1.5, BK, [("couch",3.5,2.5)],                         "Near back off-center"),
    ("FT5_78", "couch",        4.5,-2.5,  L, [("bookshelf",3,-1.5),("trash_can",5.5,-2)], "Side view, objects flanking"),
    ("FT5_79", "table",        5.5, 2.5, FR, [("couch",4,1)],                             "Far-medium, FR, left of center"),
    ("FT5_80", "trash_can",    4.0,-3.0,  F, [("cardboard_box",3,-2.2),("chair",5,-2.5)], "Medium far right, small objects"),
]

def gen_tier5():
    count = 0

    # §9.1 Off-center
    for sid, tgt_xml, tx, ty, teuler, extras, desc in TIER5_OFFCENTER:
        extra_list = extras if extras else []
        xml = t5_make(sid, desc, tgt_xml, tx, ty, teuler, extra_list)
        write_xml(sid, xml)
        count += 1

    # §9.2 Same-class multiples
    for sid, tgt_xml, positions, desc in TIER5_MULTICLASS:
        bodies = []
        for i, (x, y) in enumerate(positions):
            bname = f"{tgt_xml}_{i+1}"
            bodies.append((bname, tgt_xml, x, y, deuler(tgt_xml), f"TARGET instance {i+1}: {tgt_xml}"))
        xml = make_xml(sid, desc, bodies)
        write_xml(sid, xml)
        count += 1

    # §9.3 Close-quarters clutter
    for sid, tgt_xml, tx, ty, teuler, clutter, desc in TIER5_CLUTTER:
        xml = t5_make(sid, desc, tgt_xml, tx, ty, teuler, clutter)
        write_xml(sid, xml)
        count += 1

    # §9.4 Very far
    for sid, tgt_xml, tx, ty, teuler, extras, desc in TIER5_VERYFAR:
        xml = t5_make(sid, desc, tgt_xml, tx, ty, teuler, extras)
        write_xml(sid, xml)
        count += 1

    # §9.5 Extra
    for sid, tgt_xml, tx, ty, teuler, extras, desc in TIER5_EXTRA:
        extra_list = extras if extras else []
        xml = t5_make(sid, desc, tgt_xml, tx, ty, teuler, extra_list)
        write_xml(sid, xml)
        count += 1

    print(f"Tier 5: {count} scenarios")
    return count


# ════════════════════════════════════════════════════════════════════════════
#  TIER 6 — Partial Occlusion (90 scenarios)
# ════════════════════════════════════════════════════════════════════════════
# Each scenario: target + occluder(s) + optional extra distractor
# Format: (sid, tgt_xml, tx, ty, teuler, occluders, occlusion_type, notes)
# occluders: list of (xml, x, y)

def t6_make(sid, desc, tgt_xml, tx, ty, teuler, occluders):
    bodies = [(f"{tgt_xml}_target", tgt_xml, tx, ty, teuler, f"TARGET: {tgt_xml}")]
    counts = {}
    for xml, x, y in occluders:
        counts[xml] = counts.get(xml, 0) + 1
        bname = f"{xml}_distractor{counts[xml]}"
        bodies.append((bname, xml, x, y, deuler(xml), f"OCCLUDER: {xml}"))
    return make_xml(sid, desc, bodies)

# §10.2 Couch target (FT6_C_01–15) — fully specified in plan
T6_COUCH = [
    ("FT6_C_01", "couch", 4.0, 0, F,  [("bookshelf",2.5,-0.3)],                          "O1: bookshelf in front ~35%"),
    ("FT6_C_02", "couch", 4.0, 0, F,  [("chair",2.5,0.2)],                               "O1: chair in front ~25%"),
    ("FT6_C_03", "couch", 4.0, 0, F,  [("table",2.5,-0.2)],                              "O1: table in front ~40%"),
    ("FT6_C_04", "couch", 4.0, 0, F,  [("couch",2.5,0.4)],                               "O2: same-class in front ~30%"),
    ("FT6_C_05", "couch", 5.0, 1.5, R,[("couch",3.5,0.8)],                               "O2: same-class angled ~45%"),
    ("FT6_C_06", "couch", 3.5, 0, F,  [("couch",2.5,-0.3)],                              "O2: same-class frontal ~30%"),
    ("FT6_C_07", "couch", 3.5, 3.5, F, [],                                                "O3: target at 46° lateral ~20% left"),
    ("FT6_C_08", "couch", 3.5,-3.5, L, [],                                                "O3: side + right edge ~25%"),
    ("FT6_C_09", "couch", 4.5, 4.0, F, [],                                                "O3: wide couch, far left ~30%"),
    ("FT6_C_10", "couch", 5.0, 0, F,  [("bookshelf",2.5,0.3),("cardboard_box",3.0,-0.2)],"O4: two-object cluster ~40%"),
    ("FT6_C_11", "couch", 5.0, 0, F,  [("chair",2.8,0.5),("trash_can",3.2,-0.3)],        "O4: chair+can cluster ~35%"),
    ("FT6_C_12", "couch", 4.5, 0.8, F,  [("table",2.5,0),("bookshelf",3.5,0.4)],         "O4: table+shelf stagger ~45%"),
    ("FT6_C_13", "couch", 4.5, 3.5, F,[("bookshelf",2.5,1.5)],                           "O5: occluder+edge clip ~25%+~20%"),
    ("FT6_C_14", "couch", 4.0,-3.2, R,[("chair",2.5,-1.5)],                              "O5: side+right edge ~30%+~25%"),
    ("FT6_C_15", "couch", 5.0, 3.8, FL,[("table",3.0,2.0)],                              "O5: diagonal+table+edge ~35%"),
]

# §10.3 Table target (FT6_T_01–15) — fully specified in plan
T6_TABLE = [
    ("FT6_T_01", "table", 4.0, 0, F,  [("bookshelf",2.5,-0.2)],                          "O1: bookshelf in front ~35%"),
    ("FT6_T_02", "table", 4.0, 0, F,  [("chair",2.5,0.3)],                               "O1: chair in front ~25%"),
    ("FT6_T_03", "table", 4.0, 0.8, F,  [("couch",2.5,0)],                               "O1: couch in front ~45%"),
    ("FT6_T_04", "table", 4.0, 0, F,  [("table",2.5,0.3)],                               "O2: same-class in front ~30%"),
    ("FT6_T_05", "table", 5.0, 1.0, R,[("table",3.5,0.4)],                               "O2: same-class angled ~40%"),
    ("FT6_T_06", "table", 4.5, 0, L,  [("table",3.0,-0.2)],                              "O2: same-class side ~35%"),
    ("FT6_T_07", "table", 3.5, 3.5, F, [],                                                "O3: edge clip left ~20%"),
    ("FT6_T_08", "table", 3.5,-3.5, R, [],                                                "O3: narrow side+right edge ~25%"),
    ("FT6_T_09", "table", 4.5,-4.0, F, [],                                                "O3: right edge, long table ~30%"),
    ("FT6_T_10", "table", 5.0, 0, F,  [("chair",2.8,0.4),("cardboard_box",3.2,-0.3)],    "O4 ~35%"),
    ("FT6_T_11", "table", 5.0, 1.0, F,  [("bookshelf",2.5,-0.2),("trash_can",3.0,0.4)],  "O4 ~40%"),
    ("FT6_T_12", "table", 4.5, 0, F,  [("couch",2.5,0.3),("cardboard_box",3.5,-0.4)],    "O4 ~45%"),
    ("FT6_T_13", "table", 4.5, 3.5, F,[("chair",2.5,1.5)],                               "O5 ~30%"),
    ("FT6_T_14", "table", 4.0,-3.2, R,[("bookshelf",2.5,-1.5)],                          "O5 ~35%"),
    ("FT6_T_15", "table", 5.0, 4.0, FL,[("trash_can",3.0,2.0)],                          "O5 ~25%"),
]

# §10.4 Chair target (FT6_CH_01–15) — generated following O1-O5 pattern
T6_CHAIR = [
    ("FT6_CH_01", "chair", 4.0, 0, F,  [("bookshelf",2.5,-0.2)],                          "O1: bookshelf in front ~35%"),
    ("FT6_CH_02", "chair", 4.0, 0, F,  [("couch",2.5,0.3)],                               "O1: couch in front ~40%"),
    ("FT6_CH_03", "chair", 4.0, 0, F,  [("table",2.5,-0.2)],                              "O1: table in front ~30%"),
    ("FT6_CH_04", "chair", 4.0, 0, F,  [("chair",2.5,0.3)],                               "O2: same-class in front ~30%"),
    ("FT6_CH_05", "chair", 5.0, 1.0, R,[("chair",3.5,0.4)],                               "O2: same-class angled ~40%"),
    ("FT6_CH_06", "chair", 4.5, 0, L,  [("chair",3.0,-0.2)],                              "O2: same-class side ~35%"),
    ("FT6_CH_07", "chair", 3.5, 3.5, F, [],                                                "O3: edge clip left ~20%"),
    ("FT6_CH_08", "chair", 3.5,-3.5, R, [],                                                "O3: right edge ~25%"),
    ("FT6_CH_09", "chair", 4.5,-4.0, F, [],                                                "O3: far right edge ~30%"),
    ("FT6_CH_10", "chair", 5.0, 0, F,  [("bookshelf",2.8,0.3),("cardboard_box",3.2,-0.3)],"O4: two-object cluster ~35%"),
    ("FT6_CH_11", "chair", 5.0, 0, F,  [("couch",2.5,0.3),("trash_can",3.2,-0.3)],        "O4: couch+can cluster ~40%"),
    ("FT6_CH_12", "chair", 4.5, 1.0, F,  [("table",2.5,0),("bookshelf",3.5,0.3)],         "O4: table+shelf stagger ~45%"),
    ("FT6_CH_13", "chair", 4.5, 3.5, F,[("couch",2.5,1.5)],                               "O5: occluder+edge clip ~30%"),
    ("FT6_CH_14", "chair", 4.0,-3.2, R,[("table",2.5,-1.5)],                              "O5: side+right edge ~35%"),
    ("FT6_CH_15", "chair", 5.0, 3.8, FL,[("bookshelf",3.0,2.0)],                          "O5: diagonal+shelf+edge ~30%"),
]

# §10.5 Bookshelf target (FT6_B_01–15) — narrow depth makes occlusion significant
T6_BOOKSHELF = [
    ("FT6_B_01", "bookshelf", 4.0, 0, F,  [("chair",2.5,-0.2)],                           "O1: chair in front ~35%"),
    ("FT6_B_02", "bookshelf", 4.0, 0, F,  [("table",2.5,0.3)],                            "O1: table in front ~40%"),
    ("FT6_B_03", "bookshelf", 4.0, 0.6, F,  [("couch",2.5,-0.1)],                         "O1: couch in front ~50%"),
    ("FT6_B_04", "bookshelf", 4.0, 0, F,  [("bookshelf",2.5,0.2)],                        "O2: same-class in front ~30%"),
    ("FT6_B_05", "bookshelf", 5.0, 1.0, R,[("bookshelf",3.5,0.3)],                        "O2: same-class angled ~40%"),
    ("FT6_B_06", "bookshelf", 4.5, 0, L,  [("bookshelf",3.0,-0.15)],                      "O2: same-class side ~35%"),
    ("FT6_B_07", "bookshelf", 3.5, 3.5, F, [],                                             "O3: edge clip left ~20%"),
    ("FT6_B_08", "bookshelf", 3.5,-3.5, R, [],                                             "O3: narrow side + right edge ~25%"),
    ("FT6_B_09", "bookshelf", 4.5,-4.2, F, [],                                             "O3: very narrow at edge ~30%"),
    ("FT6_B_10", "bookshelf", 5.0, 0, F,  [("chair",2.8,0.3),("cardboard_box",3.2,-0.3)], "O4: two-object cluster ~35%"),
    ("FT6_B_11", "bookshelf", 5.0, 1.2, F,  [("couch",2.5,0.2),("trash_can",3.2,-0.3)],   "O4: couch+can cluster ~40%"),
    ("FT6_B_12", "bookshelf", 4.5, 1.5, F,  [("table",2.5,0.1),("chair",3.5,0.3)],        "O4: table+chair stagger ~45%"),
    ("FT6_B_13", "bookshelf", 4.5, 3.5, F,[("chair",2.5,1.5)],                            "O5: occluder+edge clip ~30%"),
    ("FT6_B_14", "bookshelf", 4.0,-3.2, R,[("couch",2.5,-1.5)],                           "O5: side+right edge ~35%"),
    ("FT6_B_15", "bookshelf", 5.0, 3.8, FL,[("table",3.0,2.0)],                           "O5: diagonal+table+edge ~30%"),
]

# §10.6 Cardboard Box target (FT6_BX_01–15) — small/low object, focus on O1 and O4
T6_BOX = [
    ("FT6_BX_01", "cardboard_box", 3.0, 0, F, [("chair",2.0,0.2)],                         "O1: chair in front ~40%"),
    ("FT6_BX_02", "cardboard_box", 3.0, 0, F, [("table",2.0,-0.2)],                        "O1: table in front ~50%"),
    ("FT6_BX_03", "cardboard_box", 3.0, 0, F, [("trash_can",2.0,0.1)],                     "O1: trash_can in front ~30%"),
    ("FT6_BX_04", "cardboard_box", 3.0, 0, F, [("cardboard_box",2.0,0.2)],                  "O2: same-class in front ~35%"),
    ("FT6_BX_05", "cardboard_box", 4.0, 1.0, R,[("cardboard_box",2.8,0.3)],                 "O2: same-class angled ~40%"),
    ("FT6_BX_06", "cardboard_box", 3.5, 0, L, [("cardboard_box",2.5,-0.2)],                 "O2: same-class side ~35%"),
    ("FT6_BX_07", "cardboard_box", 2.5, 3.2, F, [],                                         "O3: edge clip left ~20%"),
    ("FT6_BX_08", "cardboard_box", 2.5,-3.2, F, [],                                         "O3: right edge ~20%"),
    ("FT6_BX_09", "cardboard_box", 3.5,-3.8, F, [],                                         "O3: far right edge ~25%"),
    ("FT6_BX_10", "cardboard_box", 4.0, 0, F, [("chair",2.5,0.3),("trash_can",2.8,-0.3)],   "O4: chair+can cluster ~50%"),
    ("FT6_BX_11", "cardboard_box", 4.0, 1.5, F, [("table",2.5,0.2),("bookshelf",3.0,-0.3)], "O4: table+shelf ~60%"),
    ("FT6_BX_12", "cardboard_box", 3.5, 1.3, F, [("couch",2.0,0.1),("chair",2.8,0.4)],      "O4: couch+chair ~55%"),
    ("FT6_BX_13", "cardboard_box", 3.5, 3.0, F,[("chair",2.0,1.5)],                         "O5: occluder+edge clip ~30%"),
    ("FT6_BX_14", "cardboard_box", 3.0,-3.0, R,[("trash_can",2.0,-1.5)],                    "O5: side+edge clip ~25%"),
    ("FT6_BX_15", "cardboard_box", 4.5, 3.5, FL,[("table",3.0,2.0)],                        "O5: diagonal+table+edge ~35%"),
]

# §10.7 Trash Can target (FT6_TC_01–15) — cylindrical, focus on O1 and O4
T6_TRASHCAN = [
    ("FT6_TC_01", "trash_can", 3.0, 0, F, [("cardboard_box",2.0,0.2)],                      "O1: box in front ~30%"),
    ("FT6_TC_02", "trash_can", 3.0, 0, F, [("chair",2.0,0.1)],                              "O1: chair in front ~40%"),
    ("FT6_TC_03", "trash_can", 3.0, 0.6, F, [("table",2.0,-0.2)],                           "O1: table in front ~50%"),
    ("FT6_TC_04", "trash_can", 3.0, 0, F, [("trash_can",2.0,0.1)],                          "O2: same-class in front ~30%"),
    ("FT6_TC_05", "trash_can", 4.0, 1.0, R,[("trash_can",2.8,0.3)],                         "O2: same-class angled ~35%"),
    ("FT6_TC_06", "trash_can", 3.5, 0, L, [("trash_can",2.5,-0.1)],                         "O2: same-class side ~30%"),
    ("FT6_TC_07", "trash_can", 2.5, 3.0, F, [],                                              "O3: edge clip left, round sliver ~20%"),
    ("FT6_TC_08", "trash_can", 2.5,-3.0, F, [],                                              "O3: right edge ~20%"),
    ("FT6_TC_09", "trash_can", 3.5,-3.8, F, [],                                              "O3: far right edge ~25%"),
    ("FT6_TC_10", "trash_can", 4.0, 0, F, [("cardboard_box",2.5,0.2),("chair",2.8,-0.3)],   "O4: box+chair cluster ~45%"),
    ("FT6_TC_11", "trash_can", 4.0, -0.7, F, [("table",2.5,0.1),("couch",2.0,0.5)],         "O4: table+couch ~55%"),
    ("FT6_TC_12", "trash_can", 3.5, 1.0, F, [("bookshelf",2.0,-0.1),("cardboard_box",2.5,0.4)],"O4: shelf+box ~50%"),
    ("FT6_TC_13", "trash_can", 3.5, 3.0, F,[("cardboard_box",2.0,1.5)],                     "O5: occluder+edge clip ~30%"),
    ("FT6_TC_14", "trash_can", 3.0,-3.0, R,[("cardboard_box",2.0,-1.5)],                    "O5: side+edge clip ~25%"),
    ("FT6_TC_15", "trash_can", 4.5, 3.5, FL,[("chair",3.0,2.0)],                            "O5: diagonal+chair+edge ~30%"),
]

def gen_tier6():
    count = 0
    for group in [T6_COUCH, T6_TABLE, T6_CHAIR, T6_BOOKSHELF, T6_BOX, T6_TRASHCAN]:
        for row in group:
            sid, tgt_xml, tx, ty, teuler, occluders, desc = row
            xml = t6_make(sid, desc, tgt_xml, tx, ty, teuler, occluders)
            write_xml(sid, xml)
            count += 1
    print(f"Tier 6: {count} scenarios")
    return count


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)
    total = 0
    total += gen_tier1()
    total += gen_tier2()
    total += gen_tier3()
    total += gen_tier4()
    total += gen_tier5()
    total += gen_tier6()
    print(f"\nTotal scenarios generated: {total}")
    print(f"Output directory: {BASE_DIR}")
