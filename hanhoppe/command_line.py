import sys
import json


def parse_args():
    print('parsing args')
    print(sys.argv)

    with open(sys.argv[1]) as task_file:
        json_data = json.load(task_file)

    print(json_data)
    print(json_data['foo'])
