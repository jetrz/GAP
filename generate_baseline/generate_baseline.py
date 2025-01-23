from .gnnome_decoding import gnnome_decoding
from .hifiasm_decoding import hifiasm_decoding

def run_generate_baseline(config):
    source, genomes = config['run']['generate_baseline']['source'], config['run']['generate_baseline']['genomes']
    for genome in genomes:
        print(f"\n===== Generating baseline for {genome}. Source: {source} =====")
        paths = config['genome_info'][genome]['paths']
        paths.update(config['misc']['paths'])
        motif = config['genome_info'][genome]['telo_motifs'][0]

        if source == "GNNome":
            gnnome_decoding(genome, config['gnnome'], paths, motif)
        elif source == "hifiasm":
            hifiasm_decoding(paths, motif)
        else:
            raise ValueError("Invalid source!")