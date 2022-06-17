import argparse
from pathlib import Path

def parse_opts():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root_path',
                        default=None,
                        type=Path,
                        help='Root directory path')
    parser.add_argument('--data_path',
                            default=None,
                        type=Path,
                        help='Directory path of all data')
    parser.add_argument('--result_path',
                        default=None,
                        type=Path,
                        help='Result directory path')
