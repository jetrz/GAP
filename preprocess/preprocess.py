from datetime import datetime
import dgl, gc, pickle, subprocess, torch
from pyfaidx import Fasta

from misc.utils import pyg_to_dgl, timedelta_to_str
from .gfa_util import preprocess_gfa
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

        print(f"Processing FASTAs... (Time: {timedelta_to_str(datetime.now() - time_start)})")
        aux = {
            'hifi_r2s' : Fasta(genome_info['paths']['ec_reads']),
            'ul_r2s' : Fasta(genome_info['paths']['ul_reads']) if genome_info['paths']['ul_reads'] else None
        }

        print(f"Processing GFA... (Time: {timedelta_to_str(datetime.now() - time_start)})")
        g, aux = preprocess_gfa(genome_info['paths']['gfa'], aux, source)
        with open(genome_info['paths']['n2s'], "wb") as p:
            pickle.dump(aux['n2s'], p)
        with open(genome_info['paths']['r2n'], "wb") as p:
            pickle.dump(aux['r2n'], p)
        torch.save(g, genome_info['paths']['graph']+f'{genome}.pt')
        dgl_g = pyg_to_dgl(g, aux['node_attrs'], aux['edge_attrs'])
        dgl.save_graphs(genome_info['paths']['graph']+f'{genome}.dgl', [dgl_g])
        del aux['n2s']
        gc.collect()

        print(f"Processing PAF... (Time: {timedelta_to_str(datetime.now() - time_start)})")
        paf_data = parse_paf(genome_info['paths'], aux)
        with open(genome_info['paths']['paf_processed'], "wb") as p:
            pickle.dump(paf_data, p)
        del aux, paf_data, g, dgl_g
        gc.collect()

        print(f"Generating k-mer counts using Jellyfish... (Time: {timedelta_to_str(datetime.now() - time_start)})")
        k = config['misc']['kmers']['k']
        command = f"jellyfish count -m {k} -s 100M -t 10 -o {k}mers.jf -C {config['genome_info'][genome]['paths']['ec_reads']}"
        subprocess.run(command, shell=True, cwd=config['genome_info'][genome]['paths']['graph'])

        print(f"Run finished! (Time: {timedelta_to_str(datetime.now() - time_start)})")







    