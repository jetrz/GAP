from Bio import Seq
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
import itertools, pickle, subprocess

def process_chunk(pairs):
    results = []
    for line1, line2 in pairs:
        freq = int(line1.strip()[1:])
        kmer = line2.strip()
        rev_kmer = str(Seq.Seq(kmer).reverse_complement())
        results.append((kmer, rev_kmer, freq))

    return results

class KmerManager():
    def __init__(self, k, save_path, mode):
        if mode not in ("all", "query", "pickle"): raise ValueError("Invalid Kmer mode!")

        self.k = k
        self.mode = mode
        self.seq_memo = {}
        self.kmer_memo = {}
        self.base = 4
        self.mod = 2**61-1
        self.freqs = None
        save_path += f"{self.k}mers"
        self.jf_path = save_path+".jf"

        if mode == "pickle":
            with open(f"{save_path}_hashed.pkl", 'rb') as f:
                self.freqs = pickle.load(f)
            return

        # Get lower and upper bounds, and plot the graph
        cmd = f"jellyfish histo {save_path}.jf"
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        res = res.stdout.split("\n")
        kmer_freqs = []
        for s in res[:-1]:
            split = [int(x) for x in s.split()]
            kmer_freqs.extend([split[0]]*split[1])
        cutoff = np.percentile(kmer_freqs, 99.5)
        kmer_freqs = [i for i in kmer_freqs if i <= cutoff]

        average = np.mean(kmer_freqs)
        unique_kmer_freqs = np.array(list(set(kmer_freqs)))
        nearest_average = unique_kmer_freqs[(np.abs(unique_kmer_freqs - average)).argmin()]

        freqs = Counter(kmer_freqs)
        max_freq = np.max(unique_kmer_freqs)
        values = np.array([freqs.get(i,0) for i in range(1, max_freq+1)])
        minima_inds = argrelextrema(values, np.less)[0]

        lower, upper = minima_inds[0]+1, None
        for m in minima_inds:
            if m > nearest_average-1:
                upper = m+1
                break
        if upper is None: upper = len(values)

        self.upper = int(upper); self.lower = int(lower)

        plt.figure(figsize=(10, 5))
        x_indices = range(1, len(values) + 1)
        plt.plot(x_indices, values)
        plt.axvline(x=lower, color='r', linestyle='--', label=f'Lower Bound at {lower}')
        plt.axvline(x=nearest_average, color='g', linestyle='--', label=f'Average at {nearest_average}')
        plt.axvline(x=upper, color='b', linestyle='--', label=f'Upper Bound at {upper}')

        plt.xlabel('Kmer Frequency')
        plt.ylabel('# Kmers')
        plt.savefig(save_path+".png")
        plt.clf()

        if mode == "all":
            # Generate frequency dict for all kmers
            freqs = {}
            cmd = ["jellyfish", "dump", self.jf_path, "-L", str(self.lower), "-U", str(self.upper)]

            chunk_size = 5000
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=8192) as proc:
                lines = iter(proc.stdout)
                futures = []

                with ProcessPoolExecutor(max_workers=40) as executor:
                    while True:
                        chunk = list(itertools.islice(lines, chunk_size*2))
                        if not chunk: break
                        if len(chunk) % 2 != 0: raise ValueError("Unmatched line in Jellyfish output.")
                        pairs = [(chunk[i], chunk[i+1]) for i in range(0, len(chunk), 2)]
                        futures.append(executor.submit(process_chunk, pairs))

                    for future in as_completed(futures):
                        for kmer, kmer_rev, freq in future.result():
                            freqs[self.hash_kmer(kmer)] = freq
                            freqs[self.hash_kmer(kmer_rev)] = freq
            
            self.freqs = freqs

        return
    
    def get_seq_cov(self, seq):
        if seq in self.seq_memo: 
            avg_cov, missed, total = self.seq_memo[seq]
            return avg_cov, missed, total
        
        if self.mode in ["all", "pickle"]:
            h = self.hash_kmer(seq[:self.k])
            hashes = [h]
            power = pow(self.base, self.k-1, self.mod)

            for i in range(1, len(seq)-self.k+1):
                h = (h - self.char_to_int(seq[i-1]) * power) % self.mod
                h = (h * self.base + self.char_to_int(seq[i+self.k-1])) % self.mod
                hashes.append(h)

            total_cov, missed, total = 0, 0, len(hashes)
            for hash in hashes:
                if hash not in self.freqs:
                    missed += 1
                else:
                    total_cov += self.freqs[hash]
        elif self.mode == "query":
            kmer_list = [seq[i:i+self.k] for i in range(len(seq)-self.k+1)]
            new_kmer_list = [kmer for kmer in kmer_list if kmer not in self.kmer_memo]

            batch_size = 1000
            for i in range(0, len(new_kmer_list), batch_size):
                batch = new_kmer_list[i:i+batch_size]
                res = subprocess.run(["jellyfish", "query", self.jf_path] + batch, capture_output=True, text=True)
                counts = res.stdout.splitlines()
                for l in counts:
                    split = l.split()
                    self.kmer_memo[split[0]] = int(split[1])
                    self.kmer_memo[str(Seq.Seq(split[0]).reverse_complement())] = int(split[1])

            total_cov, missed, total = 0, 0, len(kmer_list)
            for kmer in kmer_list:
                c_freq = self.kmer_memo[kmer]
                if c_freq <= self.lower or c_freq >= self.upper:
                    missed += 1
                else:
                    total_cov += c_freq

        avg_cov = total_cov/(total-missed) if total > missed else None
        self.seq_memo[seq] = (avg_cov, missed, total)
        return avg_cov, missed, total

    def hash_kmer(self, kmer):
        h = 0
        for c in kmer:
            h = (h * self.base + self.char_to_int(c)) % self.mod
        return h
    
    def char_to_int(self, c):
        return {'A': 0, 'C': 1, 'G': 2, 'T': 3}[c]