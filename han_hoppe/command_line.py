import sys
import json
import os
import han_hoppe.tile_utils as tu


def parse_args():
    print('parsing args')

    print(sys.argv)

    if len(sys.argv) == 1:
        task_file = os.path.join(os.getcwd(),
            'data/tasks/download_and_tesselate.json')
    else:
        task_file = sys.argv[1]

    print('using task file ' + task_file)

    with open(task_file) as task_file:
        json_data = json.load(task_file)

    print(json_data)
    tu.download_and_tesselate(
        json_data['tile_url'],
        json_data['sat_qk_nw'],
        json_data['sat_w_tiles'],
        json_data['sat_h_tiles'],
        json_data['aerial_lod'],
    )
