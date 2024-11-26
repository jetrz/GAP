from Bio import Seq, SeqIO
from collections import defaultdict
from datetime import datetime
import pickle
from pyfaidx import Fasta

from misc.utils import asm_metrics, get_seqs, timedelta_to_str

def hifiasm_decoding(paths):
    time_start = datetime.now()

    print(f"Loading files... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    with open(paths["gfa"]) as f:
        rows = f.readlines()
        c2r = defaultdict(list)
        for row in rows:
            row = row.strip().split()
            if row[0] != "A": continue
            c2r[row[1]].append(row)

    with open(paths['r2n'], 'rb') as f:
        r2n = pickle.load(f)
    hifi_r2s = Fasta(paths['ec_reads'], as_raw=True)
    ul_r2s = Fasta(paths['ul_reads'], as_raw=True) if paths['ul_reads'] else None

    print(f"Generating contigs... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    contigs, walks = [], []
    for c_id, reads in c2r.items():
        reads = sorted(reads, key=lambda x:int(x[2]))

        # Remove "Ns" from the start and end of a contig
        while True:
            curr_row = reads[-1]
            if curr_row[4] != "Ns" and curr_row[4] != "scaf": break
            reads.pop()
        while True:
            curr_row = reads[0]
            if curr_row[4] != "Ns" and curr_row[4] != "scaf": break
            reads.pop(0)

        c_seq, c_walk = "", []
        for i in range(len(reads)-1):
            curr_row, next_row = reads[i], reads[i+1]
            curr_read = curr_row[4]
            
            # Handling of scaffolded regions
            if curr_read == "Ns" or curr_read == "scaf":
                curr_n_len = int(next_row[2])-int(curr_row[2])
                src_seq = "N"*int(curr_n_len)
                curr_read = f"custom_n_{curr_n_len}"
            else:
                if curr_row[3] == "+":
                    src_seq, _ = get_seqs(curr_read, hifi_r2s, ul_r2s)
                else:
                    _, src_seq = get_seqs(curr_read, hifi_r2s, ul_r2s)

            src_seq = src_seq[int(curr_row[5]):int(curr_row[6])]
            curr_prefix = int(next_row[2])-int(curr_row[2])
            c_seq += src_seq[:curr_prefix]

            curr_node = r2n[curr_read][0] if curr_row[3] == "+" else r2n[curr_read][1]
            c_walk.append(curr_node)

        curr_row = reads[-1]
        if curr_row[3] == "+":
            src_seq, _ = get_seqs(curr_row[4], hifi_r2s, ul_r2s)
        else:
            _, src_seq = get_seqs(curr_row[4], hifi_r2s, ul_r2s)
        c_seq += src_seq[int(curr_row[5]):int(curr_row[6])]
        c_seq = Seq.Seq(c_seq)
        c_seq = SeqIO.SeqRecord(c_seq)
        c_seq.id = c_id
        c_seq.description = f'length={len(c_seq)}'
        contigs.append(c_seq)

        curr_node = r2n[curr_row[4]][0] if curr_row[3] == "+" else r2n[curr_row[4]][1]
        c_walk.append(curr_node)
        walks.append(c_walk)

    pickle.dump(walks, open(f"{paths['baseline']}walks.pkl", 'wb'))

    print(f"Calculating assembly metrics... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    asm_metrics(contigs, paths['baseline'], paths['ref'], paths['minigraph'], paths['paftools'])

    print(f"Run finished! (Time: {timedelta_to_str(datetime.now() - time_start)})")
    return