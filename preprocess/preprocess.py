from datetime import datetime
import dgl, os, pickle, torch

from misc.utils import pyg_to_dgl, timedelta_to_str
from .gfa_util import preprocess_gfa
from .fasta_util import parse_ec_fasta
from .paf_util import parse_paf

def run_preprocessing(config):
    source, genomes = config['run']['preprocessing']['source'], config['run']['preprocessing']['genomes']
    for genome in genomes:
        print(f"\n===== Preprocessing {genome}. Source: {source} =====")
        time_start = datetime.now()
        aux = {}
    
        genome_info = config['genome_info'][genome]
        gfa_path = genome_info['paths']['gfa']
        assert (source == "GNNome" and gfa_path.endswith(".bp.raw.r_utg.gfa")) or (source == "hifiasm" and gfa_path.endswith(".p_ctg.gfa")), "Invalid GFA file!"
        
        print(f"Processing Error Corrected Reads FASTA... (Time: {timedelta_to_str(datetime.now() - time_start)})")
        if os.path.isfile(genome_info['paths']['r2s']) and os.path.getsize(genome_info['paths']['r2s']) > 0:
            print("Existing r2s file found! Reading...")
            with open(genome_info['paths']['r2s'], 'rb') as f:
                r2s = pickle.load(f)
        else:
            r2s = parse_ec_fasta(genome_info['paths']['ec_reads'])
            with open(genome_info['paths']['r2s'], "wb") as p:
                pickle.dump(r2s, p)
        aux['r2s'] = r2s

        print(f"Processing GFA... (Time: {timedelta_to_str(datetime.now() - time_start)})")
        g, aux = preprocess_gfa(gfa_path, aux, source)
        with open(genome_info['paths']['n2s'], "wb") as p:
            pickle.dump(aux['n2s'], p)
        with open(genome_info['paths']['r2n'], "wb") as p:
            pickle.dump(aux['r2n'], p)
        torch.save(g, genome_info['paths']['graph']+f'{genome}.pt')
        dgl_g = pyg_to_dgl(g, aux['node_attrs'], aux['edge_attrs'])
        dgl.save_graphs(genome_info['paths']['graph']+f'{genome}.dgl', [dgl_g])

        print(f"Processing PAF... (Time: {timedelta_to_str(datetime.now() - time_start)})")
        paf_data = parse_paf(genome_info['paths']['paf'], aux)
        with open(genome_info['paths']['paf_processed'], "wb") as p:
            pickle.dump(paf_data, p)

        print(f"Run finished! (Time: {timedelta_to_str(datetime.now() - time_start)})")







    