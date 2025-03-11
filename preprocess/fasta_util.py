from Bio import bgzf, SeqIO
from Bio.Seq import Seq
import gzip
from multiprocessing import Pool
from tqdm import tqdm

def parse_read(read):
    seqs = (str(read.seq), str(Seq(read.seq).reverse_complement()))
    return read.id, seqs

def parse_fasta(path):
    print(f"Parsing {path}...")

    if path.endswith('bgz'):
        if path.endswith('fasta.bgz') or path.endswith('fna.bgz') or path.endswith('fa.bgz'):
            filetype = 'fasta'
        elif path.endswith('fastq.bgz') or path.endswith('fnq.bgz') or path.endswith('fq.bgz'):
            filetype = 'fastq'
        open_func = bgzf.open
    elif path.endswith('gz'):
        if path.endswith('fasta.gz') or path.endswith('fna.gz') or path.endswith('fa.gz'):
            filetype = 'fasta'
        elif path.endswith('fastq.gz') or path.endswith('fnq.gz') or path.endswith('fq.gz'):
            filetype = 'fastq'
        open_func = gzip.open
    else:
        if path.endswith('fasta') or path.endswith('fna') or path.endswith('fa'):
            filetype = 'fasta'
        elif path.endswith('fastq') or path.endswith('fnq') or path.endswith('fq'):
            filetype = 'fastq'
        open_func = open

    data = {}
    with open_func(path, 'rt') as handle:
        rows = SeqIO.parse(handle, filetype)
        with Pool(40) as pool:
            results = pool.imap_unordered(parse_read, rows, chunksize=50)
            for id, seqs in tqdm(results, ncols=120):
                data[id] = seqs

    return data

def parse_kmer(read):
    return str(read.seq), int(read.id)

def parse_kmer_fasta(path):
    print("Parsing kmer fasta...")
    data = {}
    with open(path, 'rt') as f:
        rows = SeqIO.parse(f, 'fasta')
        with Pool(40) as pool:
            results = pool.imap_unordered(parse_kmer, rows, chunksize=50)
            for kmer, freq in tqdm(results, ncols=120):
                data[kmer] = freq

    return data