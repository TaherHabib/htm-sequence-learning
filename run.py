import os
import argparse
import sys
import logging
import json
from utils import settings

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
logger.addHandler(handler)

project_root = settings.get_project_root()

parser = argparse.ArgumentParser(description='Train the given HTM model on generated Reber Grammar Strings')
parser.add_argument('-cf', '--config', dest='config_json', action='store', default=None, help='')
parser.add_argument('-df', '--default', dest='run_default', action='store_true', help='')
parser.add_argument('-dt', '--dataset', dest='reber_strings_dataset', action='store', default=None, help='')
default_config = 'default_config.json'


if __name__ == '__main__':

    args = parser.parse_args()

    if args.run_default:
        logger.info('Building HTM network with default configurations from: {}'.format(default_config))
        with open(os.path.join(project_root, 'configs', default_config, 'r')) as config:
            experiment_params = json.load(config)

    if args.config_json is not None:
        logger.info('Building HTM network with configurations from: {}'.format(args.config_json))
        with open(os.path.join(project_root, 'configs', args.config_json, 'r')) as config:
            experiment_params = json.load(config)
