import yaml

from preprocess.preprocess import run_preprocessing
from gnnome_decoding.gnnome_decoding import run_gnnome_decoding
from postprocess.postprocess import run_postprocessing

if __name__ == "__main__":
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
        run_config = config['run']

    if run_config['preprocessing']:
        run_preprocessing(run_config['preprocessing'])

    if run_config['gnnome_decoding']:
        run_gnnome_decoding(run_config['gnnome_decoding'])

    if run_config['postprocessing']:
        run_postprocessing(run_config['postprocessing'])