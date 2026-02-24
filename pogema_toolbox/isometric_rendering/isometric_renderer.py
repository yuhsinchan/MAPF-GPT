import re
from dataclasses import dataclass

import svgwrite
import random
import math
import base64


@dataclass
class IsometricConfig:
    tile_width: int = 128
    tile_height: int = 32
    wall_height: int = 16
    view_angle: float = math.radians(70)
    stroke_color: str = "black"
    box_class_name: str = "cls-6"

    colors: tuple = (
        "#494C60",
        "#7C8398",
        "#99A2B3",
        "#737A5D",
        "#849E97",
        "#B0C2B0",
        "#B57168",
        "#D78A76",
        "#E8B39A",
        "#DBBDA1",
        "#DECAC1",
        "#E2DBDF",
    )


pattern = """
  ...#...#...#.........
  ...#.#.#...#####.#.#.
  ...#.#.....#.....#.#.
  ...###.....#.###.#.#.
  ...............#.#.#.
  ####.#####.###.#.#.#.
  .....#.......#.....#.
  .#.#######.#.###.#.#.
  ...#.....#.#.#...#.#.
  ####...#.#.#.#.#.#.#.
  ####...#.#.....#.#...
"""

pattern = [q.replace(" ", "") for q in pattern.split('\n') if q]
rows = len(pattern)
cols = len(pattern[0])

dwg = svgwrite.Drawing('01-warehouse.svg', profile='tiny')

min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')


def update_bounds(points):
    global min_x, max_x, min_y, max_y
    for x, y in points:
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)


def create_iso_tile(x, y, color, cfg):
    cos_angle = math.cos(cfg.view_angle)
    sin_angle = math.sin(cfg.view_angle)
    points = [
        (x, y),
        (x + cfg.tile_width / 2 * cos_angle, y + cfg.tile_height / 2 * sin_angle),
        (x, y + cfg.tile_height * sin_angle),
        (x - cfg.tile_width / 2 * cos_angle, y + cfg.tile_height / 2 * sin_angle)
    ]
    update_bounds(points)
    return dwg.polygon(points, fill=color, stroke='black', stroke_width=1)


def create_iso_wall(x, y, cfg, wall_height=32, wall_color="#84A1AE", stroke_color='#333333'):
    cos_angle = math.cos(cfg.view_angle)
    sin_angle = math.sin(cfg.view_angle)
    half_width = cfg.tile_width / 2

    left = [
        (x, y + cfg.tile_height * sin_angle - wall_height * sin_angle),
        (x, y + cfg.tile_height * sin_angle),
        (x - half_width * cos_angle, y + cfg.tile_height / 2 * sin_angle),
        (x - half_width * cos_angle, y - wall_height * sin_angle + cfg.tile_height / 2 * sin_angle),
    ]
    update_bounds(left)
    left_wall = dwg.polygon(left, fill=wall_color, stroke=stroke_color, stroke_width=1)
    dwg.add(left_wall)

    top = [
        (x, y + cfg.tile_height * sin_angle - wall_height * sin_angle),
        (x + cfg.tile_width / 2 * cos_angle, y + cfg.tile_height / 2 * sin_angle - wall_height * sin_angle),
        (x, y - wall_height * sin_angle),
        (x - cfg.tile_width / 2 * cos_angle, y + cfg.tile_height / 2 * sin_angle - wall_height * sin_angle)
    ]
    update_bounds(top)
    top_wall = dwg.polygon(top, fill=wall_color, stroke=stroke_color, stroke_width=1)
    dwg.add(top_wall)

    front = [
        (x + half_width * cos_angle, y - wall_height * sin_angle + cfg.tile_height / 2 * sin_angle),
        (x, y - wall_height * sin_angle + cfg.tile_height * sin_angle),
        (x, y + cfg.tile_height * sin_angle),
        (x + half_width * cos_angle, y + cfg.tile_height / 2 * sin_angle),
    ]
    update_bounds(front)
    front_wall = dwg.polygon(front, fill=wall_color, stroke=stroke_color, stroke_width=1)
    dwg.add(front_wall)


def create_iso_flag(row, col, cfg, flag_height=25, flag_color="brown", ):
    cos_angle = math.cos(cfg.view_angle)
    sin_angle = math.sin(cfg.view_angle)

    base_x = (col - row) * cfg.tile_width / 2 * cos_angle + cols * cfg.tile_width / 2
    base_y = (col + row) * cfg.tile_height / 2 * sin_angle + cfg.tile_height / 2

    pole_top_x = base_x
    pole_top_y = base_y - flag_height * sin_angle

    flag_points = [
        (base_x, base_y),
        (pole_top_x, pole_top_y),
        (pole_top_x + 10, pole_top_y + 5),
        (pole_top_x, pole_top_y + 10)
    ]
    update_bounds(flag_points)
    flag_pole = dwg.line(start=(base_x, base_y), end=(pole_top_x, pole_top_y), stroke=cfg.stroke_color, stroke_width=2)
    flag = dwg.polygon(flag_points, fill=flag_color, stroke=cfg.stroke_color, stroke_width=1)

    small_tile_width = cfg.tile_width / 3
    small_tile_height = cfg.tile_height / 3
    offset_y = small_tile_height / 2
    small_tile_points = [
        (base_x, base_y - offset_y * sin_angle),
        (base_x + small_tile_width / 2 * cos_angle, base_y - offset_y * sin_angle + small_tile_height / 2 * sin_angle),
        (base_x, base_y - offset_y * sin_angle + small_tile_height * sin_angle),
        (base_x - small_tile_width / 2 * cos_angle, base_y - offset_y * sin_angle + small_tile_height / 2 * sin_angle)
    ]
    update_bounds(small_tile_points)
    small_tile = dwg.polygon(small_tile_points, fill=flag_color, stroke=cfg.stroke_color, stroke_width=1)

    return small_tile, flag_pole, flag


def read_robot_svg_as_data_url(file_path):
    with open(file_path, 'rb') as file:
        data = file.read()
    encoded_data = base64.b64encode(data).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded_data}"


def modify_robot_color(svg_content, cfg, new_color='#FFFFFF'):
    class_pattern = re.compile(r'(\.' + re.escape(cfg.box_class_name) + r'\s*\{\s*fill\s*:\s*)(#[0-9a-fA-F]{6})(\s*;?)\s*\}')
    modified_svg_content = class_pattern.sub(r'\1' + new_color + r'\3}', svg_content)
    return modified_svg_content


def insert_robot(dwg, robot_data_url, x, y, scale=0.75, offset=3):
    # Assuming the original size of the robot image is 50x50 pixels
    original_width = 50
    original_height = 50

    # Calculate the offsets to center the image
    offset_x = -original_width * scale / 2
    offset_y = -original_height * scale / 2

    # Create a group with the necessary transformations
    robot_group = dwg.g(transform=f"translate({x - offset}, {y + offset}) scale({scale})")
    robot_image = dwg.image(href=robot_data_url, insert=(offset_x, offset_y), size=(f"{original_width}px", f"{original_height}px"))
    robot_group.add(robot_image)
    dwg.add(robot_group)


def main():
    cfg = IsometricConfig()
    cos_angle = math.cos(cfg.view_angle)
    sin_angle = math.sin(cfg.view_angle)

    traversable_cells = [(row, col) for row in range(rows) for col in range(cols) if pattern[row][col] == '.']
    random.shuffle(traversable_cells)

    agent_positions = traversable_cells[:12]
    goal_positions = traversable_cells[16:16 + 12]
    agent_id = 0
    goal_id = 0
    robot_svg_path = 'Robot_v4.svg'

    with open(robot_svg_path, 'r') as robot_file:
        robot_svg_content = robot_file.read()

    for slice in range(rows + cols - 1):
        for row in range(rows):
            col = slice - row
            if col >= 0 and col < cols:
                iso_x = (col - row) * cfg.tile_width / 2 * cos_angle + cols * cfg.tile_width / 2
                iso_y = (col + row) * cfg.tile_height / 2 * sin_angle
                floor_tile = create_iso_tile(iso_x, iso_y, '#cccccc' if pattern[row][col] == '.' else '#666666', cfg)
                dwg.add(floor_tile)

    for slice in range(rows + cols - 1):
        for row in range(rows):
            col = slice - row
            if col >= 0 and col < cols:
                iso_x = (col - row) * cfg.tile_width / 2 * cos_angle + cols * cfg.tile_width / 2
                iso_y = (col + row) * cfg.tile_height / 2 * sin_angle
                if pattern[row][col] == '#':
                    create_iso_wall(iso_x, iso_y, cfg, wall_height=cfg.wall_height)
                if (row, col) in agent_positions:
                    modified_robot_svg_content = modify_robot_color(robot_svg_content, cfg, new_color=cfg.colors[agent_id])
                    robot_data_url = f"data:image/svg+xml;base64,{base64.b64encode(modified_robot_svg_content.encode()).decode()}"
                    insert_robot(dwg, robot_data_url, iso_x, iso_y)
                    agent_id += 1
                if (col, row) in goal_positions:
                    parts = create_iso_flag(col, row, cfg, flag_color=cfg.colors[-goal_id], )
                    goal_id += 1
                    for part in parts:
                        dwg.add(part)

    dwg['width'] = max_x - min_x
    dwg['height'] = max_y - min_y
    dwg['viewBox'] = f"{min_x} {min_y} {max_x - min_x} {max_y - min_y}"

    dwg.save()


if __name__ == '__main__':
    main()
