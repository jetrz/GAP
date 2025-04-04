import copy, yaml

from preprocess.preprocess import run_preprocessing
from generate_baseline.generate_baseline import run_generate_baseline
# from postprocess.postprocess_old import run_postprocessing
from postprocess.postprocess import run_postprocessing
from misc.utils import print_ascii

if __name__ == "__main__":
    print_ascii()
    
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
        run_config = config['run']

    if run_config['preprocessing']['genomes']: run_preprocessing(copy.deepcopy(config))

    if run_config['generate_baseline']['genomes']: run_generate_baseline(copy.deepcopy(config))

    if run_config['postprocessing']['genomes']: run_postprocessing(copy.deepcopy(config))