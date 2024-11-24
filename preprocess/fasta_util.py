from Bio import SeqIO
from Bio.Seq import Seq
import gzip
from multiprocessing import Pool
from tqdm import tqdm

def parse_read(read):
    seqs = (str(read.seq), str(Seq(read.seq).reverse_complement()))
    return read.id, seqs

def parse_fasta(path):
    print(f"Parsing {path}...")
    if path.endswith('gz'):
        if path.endswith('fasta.gz') or path.endswith('fna.gz') or path.endswith('fa.gz'):
            filetype = 'fasta'
        elif path.endswith('fastq.gz') or path.endswith('fnq.gz') or path.endswith('fq.gz'):
            filetype = 'fastq'
    else:
        if path.endswith('fasta') or path.endswith('fna') or path.endswith('fa'):
            filetype = 'fasta'
        elif path.endswith('fastq') or path.endswith('fnq') or path.endswith('fq'):
            filetype = 'fastq'

    data = {}
    open_func = gzip.open if path.endswith('.gz') else open
    with open_func(path, 'rt') as handle:
        rows = SeqIO.parse(handle, filetype)

        with Pool(15) as pool:
            results = pool.imap_unordered(parse_read, rows, chunksize=50)
            for id, seqs in tqdm(results, ncols=120):
                data[id] = seqs

    return data