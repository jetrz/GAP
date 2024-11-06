from collections import defaultdict
from datetime import datetime
import pickle

from misc.utils import asm_metrics, timedelta_to_str

def hifiasm_decoding(paths):
    time_start = datetime.now()

    print(f"Loading files... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    with open(paths["gfa"]) as f:
        rows = f.readlines()
        contigs = defaultdict(list)
        for row in rows:
            row = row.strip().split()
            if row[0] != "A": continue
            contigs[row[1]].append(row)

    with open(paths['r2s'], 'rb') as f:
        r2s = pickle.load(f)

    print(f"Generating contigs... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    contigs = []
    for reads in contigs.values():
        reads = sorted(reads, key=lambda x:int(x[2]))
        c_seq = ""
        for i in range(len(reads)-1):
            curr_row, next_row = reads[i], reads[i+1]
            src_seq = r2s[curr_row[4]][0] if curr_row[3] == "+" else r2s[curr_row[4]][1]

            src_seq = src_seq[int(curr_row[5]):int(curr_row[6])]
            curr_prefix = int(next_row[2])-int(curr_row[2])
            c_seq += src_seq[:curr_prefix]

        curr_row = reads[-1]
        src_seq = r2s[curr_row[4]][0] if curr_row[3] == "+" else r2s[curr_row[4]][1]
        c_seq += src_seq[int(curr_row[5]):int(curr_row[6])]
        contigs.append(c_seq)

    print(f"Calculating assembly metrics... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    asm_metrics(contigs, paths['baseline'], paths['ref'], paths['minigraph'], paths['paftools'])

    print(f"Run finished! (Time: {timedelta_to_str(datetime.now() - time_start)})")
    return