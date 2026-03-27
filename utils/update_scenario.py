#!/usr/bin/env python3
"""
Utility to swap the active scenario <include> line in go2.xml.
Uses regex substitution to preserve MuJoCo-specific XML extensions.
"""
import re
import os
import argparse

# Matches active (non-commented) scenario include lines only.
# Does NOT match commented-out lines (e.g. <!-- <include file="./room/..."> -->)
# because comments include surrounding whitespace/text that this pattern won't appear in.
INCLUDE_PATTERN = re.compile(
    r'<include\s+file="\.\/scenarios\/[^"]+\/scene_room\.xml"\s*\/>'
)

def update_scenario_include(xml_path: str, scenario_id: str) -> str:
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    target_path = os.path.join(xml_dir, "scenarios", scenario_id, "scene_room.xml")
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Scenario XML not found: {target_path}"
        )
    
    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    match = INCLUDE_PATTERN.search(content)
    if match is None:
        raise ValueError(
            f"No active scenario include line found in {xml_path}. "
            "Expected pattern: <include file=\"./scenarios/.../scene_room.xml\" />"
        )
    original_include = match.group(0)

    new_include = f'<include file="./scenarios/{scenario_id}/scene_room.xml" />'

    new_content = INCLUDE_PATTERN.sub(new_include, content, count=1)

    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return original_include

def restore_scenario_include(xml_path: str, original_include: str) -> None:
    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = INCLUDE_PATTERN.sub(original_include, content, count=1)

    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

def main():
    parser = argparse.ArgumentParser(
        description='Swap the active scenario include line in go2.xml'
    )
    parser.add_argument('--xml_path', type=str, required=True,
                        help='Path to go2.xml')
    parser.add_argument('--scenario_id', type=str, required=True,
                        help='Scenario ID to activate, e.g. C01, X07')
    args = parser.parse_args()

    if not os.path.exists(args.xml_path):
        raise FileNotFoundError(f"XML file not found: {args.xml_path}")

    old_include = update_scenario_include(args.xml_path, args.scenario_id)
    xml_dir = os.path.dirname(os.path.abspath(args.xml_path))
    print(f"Updated scenario include in: {args.xml_path}")
    print(f"  Old: {old_include}")
    print(f"  New: <include file=\"./scenarios/{args.scenario_id}/scene_room.xml\" />")


if __name__ == '__main__':
    main()