import os, pickle, subprocess
from Bio import SeqIO
from collections import Counter
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from scipy.signal import argrelextrema
from tqdm import tqdm

BASE, MOD = 4, 2**61-1

def char_to_int(c):
    return {'A': 0, 'C': 1, 'G': 2, 'T': 3}[c]

def hash_kmer(kmer):
    h = 0
    for c in kmer:
        h = (h * BASE + char_to_int(c)) % MOD
    return h

def parse_kmer(read):
    rev_kmer = read.seq.reverse_complement()
    return hash_kmer(str(read.seq)), hash_kmer(str(rev_kmer)), int(read.id)

def parse_kmer_fasta(path):
    print("Parsing kmer fasta...")
    data = {}
    with open(path, 'rt') as f:
        rows = SeqIO.parse(f, 'fasta')
        with Pool(40) as pool:
            results = pool.imap_unordered(parse_kmer, rows, chunksize=50)
            for hash, rev_hash, freq in tqdm(results, ncols=120):
                data[hash] = freq
                data[rev_hash] = freq

    return data

class KmerManager():
    """
    This class is used both in preprocessing and postprocessing portions. 
    - In preprocessing, gen_jf() and gen_hashed_kmers() are called.
    - In postprocessing, get_seq_cov() is called. get_seq_cov() will not work if gen_jf() and gen_hashed_kmers() has not been called for that respective dataset. 
    """

    def __init__(self, k, save_path):
        self.k = k
        self.seq_memo = {}
        self.base = BASE
        self.mod = MOD
        self.freqs = None
        self.save_path = save_path

        hashed_path = f"{save_path}{k}mers_hashed.pkl"
        if os.path.isfile(hashed_path):
            with open(hashed_path, 'rb') as f:
                self.freqs = pickle.load(f)
        else:
            print("Hashed kmers pickle not found!")

        return
    
    def get_seq_cov(self, seq):
        if seq in self.seq_memo: 
            avg_cov, missed, total = self.seq_memo[seq]
            return avg_cov, missed, total
        
        h = hash_kmer(seq[:self.k])
        hashes = [h]
        power = pow(self.base, self.k-1, self.mod)

        for i in range(1, len(seq)-self.k+1):
            h = (h - char_to_int(seq[i-1]) * power) % self.mod
            h = (h * self.base + char_to_int(seq[i+self.k-1])) % self.mod
            hashes.append(h)

        total_cov, missed, total = 0, 0, len(hashes)
        for hash in hashes:
            if hash not in self.freqs:
                missed += 1
            else:
                total_cov += self.freqs[hash]

        avg_cov = total_cov/(total-missed) if total > missed else None
        self.seq_memo[seq] = (avg_cov, missed, total)
        return avg_cov, missed, total
    
    def gen_jf(self, ec_reads_path):
        jf_path = self.save_path+f"{self.k}mers.jf"

        if os.path.isfile(jf_path):
            print("Jellyfish has already been generated!")
            return
        
        command = f"jellyfish count -m {self.k} -s 100M -t 10 -o {jf_path} -C {ec_reads_path}"
        subprocess.run(command, shell=True)
        return

    def gen_hashed_kmers(self):
        jf_path = self.save_path+f"{self.k}mers.jf"
        png_path = self.save_path+f"{self.k}mers.png"
        fa_path = self.save_path+f"{self.k}mers.fa"
        hashed_path = self.save_path+f"{self.k}mers_hashed.pkl"

        if os.path.isfile(hashed_path):
            print("Hashed kmers pickle has already been generated!")
            return
        
        # Get lower and upper bounds, and plot the graph
        cmd = f"jellyfish histo {jf_path}"
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        res = res.stdout.split("\n")
        kmer_freqs = []
        for s in res[:-1]:
            split = [int(x) for x in s.split()]
            kmer_freqs.extend([split[0]]*split[1])
        cutoff = np.percentile(kmer_freqs, 99.5)
        kmer_freqs = [i for i in kmer_freqs if i <= cutoff]
        unique_kmer_freqs = np.array(list(set(kmer_freqs)))

        freqs = Counter(kmer_freqs)
        max_freq = np.max(unique_kmer_freqs)
        values = np.array([freqs.get(i,0) for i in range(1, max_freq+1)])
        minima_inds = argrelextrema(values, np.less)[0]

        lower, upper = minima_inds[0]+1, None
        kmer_freqs = [i for i in kmer_freqs if i > lower]
        average = np.mean(kmer_freqs)
        nearest_average = unique_kmer_freqs[(np.abs(unique_kmer_freqs - average)).argmin()]
        for m in minima_inds:
            if m > nearest_average-1:
                upper = m+1
                break
        if upper is None: upper = len(values)

        plt.figure(figsize=(10, 5))
        x_indices = range(1, len(values) + 1)
        plt.plot(x_indices, values)
        plt.axvline(x=lower, color='r', linestyle='--', label=f'Lower Bound at {lower}')
        plt.axvline(x=nearest_average, color='g', linestyle='--', label=f'Average at {nearest_average}')
        plt.axvline(x=upper, color='b', linestyle='--', label=f'Upper Bound at {upper}')

        plt.xlabel('Kmer Frequency')
        plt.ylabel('# Kmers')
        plt.savefig(png_path)
        plt.clf()

        cmd = f"jellyfish dump {jf_path} -L {lower} -U {upper} -o {fa_path}"
        subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        data = parse_kmer_fasta(fa_path)
        with open(hashed_path, "wb") as p:
            pickle.dump(data, p)
        os.remove(fa_path)

        return