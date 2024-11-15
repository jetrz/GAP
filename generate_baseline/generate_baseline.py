from preprocess.gfa_util import preprocess_gfa
from preprocess.fasta_util import parse_ec_fasta
from misc.utils import pyg_to_dgl

from .gnnome_decoding import gnnome_decoding
from .hifiasm_decoding import hifiasm_decoding

def run_generate_baseline(config):
    source, genomes = config['run']['generate_baseline']['source'], config['run']['generate_baseline']['genomes']
    for genome in genomes:
        print(f"\n===== Generating baseline for {genome}. Source: {source} =====")
        paths = config['genome_info'][genome]['paths']
        paths.update(config['misc']['paths'])

        r2s = parse_ec_fasta(paths['ec_reads'])
        g, aux = preprocess_gfa(paths['gfa'], {'r2s':r2s}, source)
        dgl_g = pyg_to_dgl(g, aux['node_attrs'], aux['edge_attrs'])

        if source == "GNNome":
            gnnome_decoding(genome, config['gnnome'], paths, dgl_g, aux['n2s'])
        elif source == "hifiasm":
            hifiasm_decoding(paths, aux['r2s'], aux['r2n'])
        else:
            raise ValueError("Invalid source!")