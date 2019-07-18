#!/usr/bin/env python3

import os.path
import sys
import json

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} <annotations.json>'.format(sys.argv[0]))
        exit(1)
    with open(sys.argv[1], 'r') as f:
        annotations = json.load(f)

    fps = 29.973810
    for fname, timestamps in annotations.items():
        with open(fname, 'r') as f:
            data = json.load(f)
        base_name = os.path.splitext(fname)[0]
        for i, (start, end) in enumerate(timestamps):
            new_fname = '{}_{}.json'.format(base_name, i)
            start = int(fps * start)
            end = int(fps * end)
            split = data[start:end]
            with open(new_fname, 'w') as f:
                json.dump(split, f)
            print('Written: {}'.format(new_fname))
