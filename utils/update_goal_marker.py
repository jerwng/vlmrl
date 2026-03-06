#!/usr/bin/env python3
"""
Script to update the red_disk_marker position in the Go2 XML file.
This visualizes the goal position in the MuJoCo simulation.
"""
import argparse
import xml.etree.ElementTree as ET
import os


def update_marker_position(xml_path, x, y, z):
    """
    Update the red disk marker position in the XML file.
    
    Args:
        xml_path: Path to the XML file
        x: X coordinate (forward/backward)
        y: Y coordinate (left/right)
        z: Z coordinate (height)
    """
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find the red_disk_marker body element
    marker_body = None
    for body in root.iter('body'):
        if body.get('name') == 'red_disk_marker':
            marker_body = body
            break
    
    if marker_body is None:
        raise ValueError("Could not find 'red_disk_marker' body in XML file")
    
    # Update the position
    old_pos = marker_body.get('pos')
    new_pos = f"{x} {y} {z}"
    marker_body.set('pos', new_pos)
    
    # Write back to file
    tree.write(xml_path, encoding='unicode', xml_declaration=True)
    
    print(f"Updated red_disk_marker position:")
    print(f"  Old: {old_pos}")
    print(f"  New: {new_pos}")
    print(f"  File: {xml_path}")


def main():
    parser = argparse.ArgumentParser(description='Update goal marker position in Go2 XML file')
    parser.add_argument('--xml_path', type=str, required=True,
                        help='Path to the Go2 XML file')
    parser.add_argument('--x', type=float, default=20.0,
                        help='X position of marker (forward/backward), default=20.0')
    parser.add_argument('--y', type=float, default=0.0,
                        help='Y position of marker (left/right), default=0.0')
    parser.add_argument('--z', type=float, default=0.485,
                        help='Z position of marker (height), default=0.485')
    
    args = parser.parse_args()
    
    # Check if XML file exists
    if not os.path.exists(args.xml_path):
        raise FileNotFoundError(f"XML file not found: {args.xml_path}")
    
    # Update the XML file
    update_marker_position(args.xml_path, args.x, args.y, args.z)


if __name__ == '__main__':
    main()
