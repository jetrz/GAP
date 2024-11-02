from datetime import datetime
import dgl, pickle, yaml

from ..misc.utils import pyg_to_dgl, timedelta_to_str
from .gfa_util import preprocess_gfa
from .fasta_util import parse_fasta
from .paf_util import parse_paf

def preprocess(genomes):
    with open("../config.yaml") as file:
        config = yaml.safe_load(file)

    for genome in genomes:
        genome_info = config['genome_info'][genome]
        gfa_path = genome_info['paths']['gfa']
        if gfa_path.endswith(".bp.raw.r_utg.gfa"):
            source = 'gnnome'
        elif gfa_path.endswith(".bp.p_ctg.gfa"):
            source = 'hifiasm'
        else:
            raise ValueError("Invalid GFA file!")
        
        print(f"\n===== Preprocessing {genome}. Source: {source} =====")
        time_start = datetime.now()
        
        print(f"Processing GFA... (Time: {timedelta_to_str(datetime.now() - time_start)})")
        g, aux = preprocess_gfa(gfa_path, source)
        with open(genome_info['paths']['n2s'], "wb") as p:
            pickle.dump(aux['n2s'], p)
        with open(genome_info['paths']['r2n'], "wb") as p:
            pickle.dump(aux['r2n'], p)
        dgl_g = pyg_to_dgl(g, aux['node_attrs', aux['edge_attrs']])
        dgl.save_graphs(genome_info['paths']['graph'], [dgl_g])

        print(f"Processing Reads FASTA... (Time: {timedelta_to_str(datetime.now() - time_start)})")
        fasta_data = parse_fasta(genome_info['paths']['reads'])
        aux['annotated_fasta_data'] = fasta_data
        with open(genome_info['paths']['reads_processed'], "wb") as p:
            pickle.dump(fasta_data, p)

        print(f"Processing PAF... (Time: {timedelta_to_str(datetime.now() - time_start)})")
        paf_data = parse_paf(genome_info['paths']['paf'], aux)
        with open(genome_info['paths']['paf_processed'], "wb") as p:
            pickle.dump(paf_data, p)







    