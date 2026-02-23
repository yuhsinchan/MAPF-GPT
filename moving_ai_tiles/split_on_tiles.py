from pathlib import Path

import yaml


def save_maps_to_single_yaml(filename, maps):
    with open(filename, 'w') as file:
        yaml.add_representer(str,
                             lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|'))
        yaml.dump(maps, file)


def read_map_from_string(map_string):
    map_data = map_string.split('\n')
    return map_data


def split_map_into_tiles(map_data, tile_size):
    map_height = len(map_data)
    map_width = len(map_data[0])
    tiles = []

    for row in range(0, map_height, tile_size):
        for col in range(0, map_width, tile_size):
            tile = []
            for tile_row in range(tile_size):
                tile.append(map_data[row + tile_row][col:col + tile_size])
            tiles.append(tile)
    return ['\n'.join(tile) for tile in tiles]


def main():
    maps = {}
    for filename in Path("moving-ai-maps").glob("*.map"):

        with open(filename, 'r') as f:
            data = f.readlines()[4:]
            symbols = set()
            for idx, x in enumerate(data):
                data[idx] = x.strip()
                data[idx] = data[idx].replace("@", "#")

            for tile_idx, tile in enumerate(split_map_into_tiles(data, 64)):
                maps[filename.stem + '_' + str(tile_idx).zfill(2)] = tile

    save_maps_to_single_yaml("moving-ai-tiles.yaml", maps)


if __name__ == '__main__':
    main()
