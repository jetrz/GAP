import dgl, os, pickle, random, subprocess
from Bio import Seq, SeqIO
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool
from pyfaidx import Fasta
from tqdm import tqdm

from .custom_graph import AdjList, Edge
from .kmer_manager import KmerManager
from misc.utils import analyse_graph, asm_metrics, get_seqs, timedelta_to_str, yak_metrics, t2t_metrics


def chop_walks_seqtk(old_walks, n2s, graph, edges_full, rep1, rep2, seqtk_path):
    """
    Generates telomere information, then chops the walks. 
    1. Contigs are regenerated from the walk nodes.
    2. seqtk is used to detect telomeres, then I manually count the motifs in each region to determine if it is a '+' or '-' motif.
    3. The walks are then chopped. When a telomere is found:
        a. If there is no telomere in the current walk, and the walk is already >= twice the length of the found telomere, the telomere is added to the walk and then chopped.
        b. If there is an opposite telomere in the current walk, the telomere is added to the walk and then chopped.
        c. If there is an identical telomere in the current walk, the walk is chopped and a new walk begins with the found telomere.
    """
    # Regenerate old contigs
    old_contigs, pos_to_node = [], defaultdict(dict)
    for walk_id, walk in enumerate(old_walks):
        seq, curr_pos = "", 0
        for idx, node in enumerate(walk):
            # Preprocess the sequence
            c_seq = str(n2s[node])
            if idx != len(walk)-1:
                c_prefix = graph.edata['prefix_length'][edges_full[node,walk[idx+1]]]
                c_seq = c_seq[:c_prefix]

            seq += c_seq
            c_len_seq = len(c_seq)
            for i in range(curr_pos, curr_pos+c_len_seq):
                pos_to_node[walk_id][i] = node
            curr_pos += c_len_seq
        old_contigs.append(seq)
    
    temp_fasta_name = f'temp_{random.randint(1,9999999)}.fasta'
    with open(temp_fasta_name, 'w') as f:
        for i, contig in enumerate(old_contigs):
            f.write(f'>{i}\n')  # Using index as ID
            f.write(f'{contig}\n')

    # Use seqtk to get telomeric regions
    seqtk_cmd_rep1 = f"{seqtk_path} telo -m {rep1} {temp_fasta_name}"
    seqtk_cmd_rep2 = f"{seqtk_path} telo -m {rep2} {temp_fasta_name}"
    seqtk_res_rep1 = subprocess.run(seqtk_cmd_rep1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    seqtk_res_rep2 = subprocess.run(seqtk_cmd_rep2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if seqtk_res_rep1.returncode != 0: raise RuntimeError(seqtk_res_rep1.stderr.strip())
    if seqtk_res_rep2.returncode != 0: raise RuntimeError(seqtk_res_rep2.stderr.strip())
    seqtk_res_rep1 = seqtk_res_rep1.stdout.split("\n"); seqtk_res_rep1.pop()
    seqtk_res_rep2 = seqtk_res_rep2.stdout.split("\n"); seqtk_res_rep2.pop()

    telo_info = defaultdict(dict)
    for row in seqtk_res_rep1:
        row_split = row.split("\t")
        walk_id, start, end = int(row_split[0]), int(row_split[1]), int(row_split[2])-1
        c_seq = old_contigs[walk_id][start:end]
        rep1_count, rep2_count = c_seq.count(rep1), c_seq.count(rep2)
        c_rep = rep1 if rep1_count > rep2_count else rep2
        start_node, end_node = pos_to_node[walk_id][start], pos_to_node[walk_id][end]
        if start_node in telo_info[walk_id]:
            print("Duplicate telomere region found 1!")
        else:
            telo_info[walk_id][start_node] = (end_node, c_rep)
    for row in seqtk_res_rep2:
        row_split = row.split("\t")
        walk_id, start, end = int(row_split[0]), int(row_split[1]), int(row_split[2])-1
        c_seq = old_contigs[walk_id][start:end]
        rep1_count, rep2_count = c_seq.count(rep1), c_seq.count(rep2)
        c_rep = rep1 if rep1_count > rep2_count else rep2
        start_node, end_node = pos_to_node[walk_id][start], pos_to_node[walk_id][end]
        if start_node in telo_info[walk_id]:
            print("Duplicate telomere region found 2!")
        else:
            telo_info[walk_id][start_node] = (end_node, c_rep)
    os.remove(temp_fasta_name)

    # Chop walks
    print("Chopping walks based on telomeres...")
    new_walks, telo_ref = [], {}
    for walk_id, walk in tqdm(enumerate(old_walks), ncols=120):
        curr_ind, curr_walk, curr_telo = 0, [], None
        while curr_ind < len(walk):
            curr_node = walk[curr_ind]
            if curr_node in telo_info[walk_id]:
                end_node, telo_type = telo_info[walk_id][curr_node]
                if curr_telo is None: # There is currently no telo type in the walk. 
                    curr_telo = telo_type
                    init_walk_len = len(curr_walk)
                    while True:
                        curr_node = walk[curr_ind]
                        curr_walk.append(curr_node)
                        curr_ind += 1
                        if curr_node == end_node: break
                    if init_walk_len != 0: # if there was anything before the telomeric region, include the region and chop the walk
                        new_walks.append(curr_walk.copy())
                        telo_ref[len(new_walks)-1] = {
                            'start' : None,
                            'end' : '+' if curr_telo == rep1 else '-'
                        }
                        curr_walk, curr_telo = [], None
                elif curr_telo == telo_type: # The newly found telo type matches the current walk's telo type. Should be chopped immediately.
                    new_walks.append(curr_walk.copy())
                    telo_ref[len(new_walks)-1] = {
                        'start' : '+' if curr_telo == rep1 else '-',
                        'end' : None
                    }
                    curr_walk, curr_telo = [], telo_type
                    while True:
                        curr_node = walk[curr_ind]
                        curr_walk.append(curr_node)
                        curr_ind += 1
                        if curr_node == end_node: break
                else: # The newly found telo type does not match the current walk's telo type. Add the telomeric region, then chop the walk.
                    while True:
                        curr_node = walk[curr_ind]
                        curr_walk.append(curr_node)
                        curr_ind += 1
                        if curr_node == end_node: 
                            new_walks.append(curr_walk.copy())
                            telo_ref[len(new_walks)-1] = {
                                'start' : '+' if curr_telo == rep1 else '-',
                                'end' : '+' if telo_type == rep1 else '-'
                            }
                            curr_walk, curr_telo = [], None
                            break
            else:
                curr_walk.append(curr_node)
                curr_ind += 1

        if curr_walk: 
            new_walks.append(curr_walk.copy())
            if curr_telo == rep1:
                start_telo = '+'
            elif curr_telo == rep2:
                start_telo = '-'
            else:
                start_telo = None
            telo_ref[len(new_walks)-1] = {
                'start' : start_telo,
                'end' : None
            }

    # Sanity Check
    assert [item for inner in new_walks for item in inner] == [item for inner in old_walks for item in inner], "Not all nodes accounted for when chopping old walks!"

    rep1_count, rep2_count = 0, 0
    for v in telo_ref.values():
        if v['start'] == '+': rep1_count += 1
        if v['end'] == '+': rep1_count += 1
        if v['start'] == '-': rep2_count += 1
        if v['end'] == '-': rep2_count += 1
    print(f"Telomere chopping complete! n Old Walks: {len(old_walks)}, n New Walks: {len(new_walks)}, n +ve telomeric regions: {rep1_count}, n -ve telomeric regions: {rep2_count}")

    return new_walks, telo_ref

def add_ghosts(old_walks, paf_data, r2n, hifi_r2s, ul_r2s, walk_valid_p, hop):
    """
    Adds nodes and edges from the PAF and graph.

    1. Stores all nodes in the walks that are available for connection in n2n_start and n2n_end (based on walk_valid_p). 
    This is split into nodes at the start and end of walks bc incoming edges can only connect to nodes at the start of walks, and outgoing edges can only come from nodes at the end of walks.
    2. I add edges between existing walk nodes using information from PAF (although in all experiments no such edges have been found).
    3. I add nodes using information from PAF.
    4. I add nodes using information from the graph (and by proxy the GFA).
    5. I calculate the probability scores for all these new edges using GNNome's model and save them in e2s. This info is only used if decoding = 'gnnome_score'.
    """
    n_id = 0
    adj_list = AdjList()

    # Only the first and last walk_valid_p% of nodes in a walk can be connected. Also initialises nodes from walks
    n2n_start, n2n_end = {}, {} # n2n maps old n_id to new n_id, for the start and ends of the walks respectively
    nodes_in_old_walks = set()
    for walk in old_walks:
        nodes_in_old_walks.update(walk)

        if len(walk) == 1:
            n2n_start[walk[0]] = n_id
            n2n_end[walk[0]] = n_id
        else:
            cutoff = int(max(1, len(walk) // (1/walk_valid_p)))
            first_part, last_part = walk[:cutoff], walk[-cutoff:]
            for n in first_part:
                n2n_start[n] = n_id
            for n in last_part:
                n2n_end[n] = n_id

        n_id += 1

    ghost_data = paf_data['ghost_nodes']
    print("Adding ghost nodes for Hop 1...")
    n2s_ghost, r2n_ghost = {}, {}
    added_nodes_h1_count, added_nodes_h2_count = 0, 0
    c_ghost_data = ghost_data['hop_1']
    
    r2s = {}
    for read_id in tqdm(c_ghost_data['+'].keys(), ncols=120, desc="Parsing sequences"):
        pos_seq, neg_seq = get_seqs(read_id, hifi_r2s, ul_r2s)
        r2s[read_id] = { '+':pos_seq, '-':neg_seq }

    for orient in ['+', '-']:
        for read_id, data in tqdm(c_ghost_data[orient].items(), ncols=120, desc=f"Orient: {orient}"):
            curr_out_neighbours, curr_in_neighbours = set(), set()

            for i, out_read_id in enumerate(data['outs']):
                out_n_id = r2n[out_read_id[0]][0] if out_read_id[1] == '+' else r2n[out_read_id[0]][1]
                if out_n_id not in n2n_start: continue
                curr_out_neighbours.add((out_n_id, data['prefix_len_outs'][i], data['ol_len_outs'][i], data['ol_similarity_outs'][i]))

            for i, in_read_id in enumerate(data['ins']):
                in_n_id = r2n[in_read_id[0]][0] if in_read_id[1] == '+' else r2n[in_read_id[0]][1] 
                if in_n_id not in n2n_end: continue
                curr_in_neighbours.add((in_n_id, data['prefix_len_ins'][i], data['ol_len_ins'][i], data['ol_similarity_ins'][i]))

            for n in curr_out_neighbours:
                adj_list.add_edge(Edge(
                    new_src_nid=n_id,
                    new_dst_nid=n2n_start[n[0]],
                    old_src_nid=None,
                    old_dst_nid=n[0],
                    prefix_len=n[1],
                    ol_len=n[2],
                    ol_sim=n[3]
                ))
            for n in curr_in_neighbours:
                adj_list.add_edge(Edge(
                    new_src_nid=n2n_end[n[0]],
                    new_dst_nid=n_id,
                    old_src_nid=n[0],
                    old_dst_nid=None,
                    prefix_len=n[1],
                    ol_len=n[2],
                    ol_sim=n[3]
                ))

            n2s_ghost[n_id] = r2s[read_id][orient]
            r2n_ghost[read_id] = n_id
            n_id += 1; added_nodes_h1_count += 1

    print("Adding ghost nodes for Hop 2...")
    c_ghost_data = ghost_data['hop_2'] if hop >= 2 and 'hop_2' in ghost_data else defaultdict(dict)
    for orient in ['+', '-']:
        for read_id, data in tqdm(c_ghost_data[orient].items(), ncols=120, desc=f"Orient: {orient}"):
            curr_out_neighbours, curr_in_neighbours = set(), set()

            for i, out_read_id in enumerate(data['outs']):
                c_read_id = out_read_id[0]
                if c_read_id in r2n_ghost: # This is a ghost node
                    out_n_id = r2n_ghost[c_read_id]
                    curr_out_neighbours.add((out_n_id, data['prefix_len_outs'][i], data['ol_len_outs'][i], data['ol_similarity_outs'][i]))

            for i, in_read_id in enumerate(data['ins']):
                c_read_id = in_read_id[0]
                if c_read_id in r2n_ghost:
                    in_n_id = r2n_ghost[c_read_id]
                    curr_in_neighbours.add((in_n_id, data['prefix_len_ins'][i], data['ol_len_ins'][i], data['ol_similarity_ins'][i]))

            # ghost nodes in outermost hop are only useful if they have both at least one outgoing and one incoming edge
            if not curr_out_neighbours or not curr_in_neighbours: continue

            for n in curr_out_neighbours:
                adj_list.add_edge(Edge(
                    new_src_nid=n_id,
                    new_dst_nid=n[0],
                    old_src_nid=None,
                    old_dst_nid=None,
                    prefix_len=n[1],
                    ol_len=n[2],
                    ol_sim=n[3]
                ))
            for n in curr_in_neighbours:
                adj_list.add_edge(Edge(
                    new_src_nid=n[0],
                    new_dst_nid=n_id,
                    old_src_nid=None,
                    old_dst_nid=None,
                    prefix_len=n[1],
                    ol_len=n[2],
                    ol_sim=n[3]
                ))

            if orient == '+':
                seq, _ = get_seqs(read_id, hifi_r2s, ul_r2s)
            else:
                _, seq = get_seqs(read_id, hifi_r2s, ul_r2s)
            n2s_ghost[n_id] = seq
            r2n_ghost[read_id] = n_id
            n_id += 1; added_nodes_h2_count += 1

    print(f"Number of ghost nodes added - Hop 1: {added_nodes_h1_count}, Hop 2: {added_nodes_h2_count}")

    print(f"Adding edges between existing nodes...")
    valid_src, valid_dst, prefix_lens, ol_lens, ol_sims = paf_data['ghost_edges']['valid_src'], paf_data['ghost_edges']['valid_dst'], paf_data['ghost_edges']['prefix_len'], paf_data['ghost_edges']['ol_len'], paf_data['ghost_edges']['ol_similarity']
    added_edges_h1_count, added_edges_h2_count = 0, 0
    for i in tqdm(range(len(valid_src)), ncols=120):
        src, dst, prefix_len, ol_len, ol_sim = valid_src[i], valid_dst[i], prefix_lens[i], ol_lens[i], ol_sims[i]
        src_read_id, dst_read_id = src[0], dst[0]
        if src_read_id in r2n and dst_read_id in r2n: # Edge is from 1-hop neighbourhood
            src_n_id = r2n[src_read_id][0] if src[1] == '+' else r2n[src_read_id][1]
            dst_n_id = r2n[dst_read_id][0] if dst[1] == '+' else r2n[dst_read_id][1]
            if src_n_id not in n2n_end or dst_n_id not in n2n_start: continue
            if n2n_end[src_n_id] == n2n_start[dst_n_id]: continue # ignore self-edges
            adj_list.add_edge(Edge(
                new_src_nid=n2n_end[src_n_id], 
                new_dst_nid=n2n_start[dst_n_id], 
                old_src_nid=src_n_id, 
                old_dst_nid=dst_n_id, 
                prefix_len=prefix_len, 
                ol_len=ol_len, 
                ol_sim=ol_sim
            ))
            added_edges_h1_count += 1
        elif hop >= 2 and src_read_id in r2n_ghost and dst_read_id in r2n_ghost: # Edge is from 2-hop neighbourhood
            src_n_id, dst_n_id = r2n_ghost[src_read_id], r2n_ghost[dst_read_id]
            adj_list.add_edge(Edge(
                new_src_nid=src_n_id, 
                new_dst_nid=dst_n_id, 
                old_src_nid=None, 
                old_dst_nid=None, 
                prefix_len=prefix_len, 
                ol_len=ol_len, 
                ol_sim=ol_sim
            ))
            added_edges_h2_count += 1

    print(f"Number of edges between existing nodes added - Hop 1: {added_edges_h1_count}, Hop 2: {added_edges_h2_count}")

    print("Removing ghost nodes with no in or out neighbours...")
    in_out_degs = { i:[0,0] for i in range(len(old_walks), n_id) }
    for src_nid, neighbours in adj_list.adj_list.items():
        for n in neighbours:
            if n.new_dst_nid in in_out_degs: in_out_degs[n.new_dst_nid][0] += 1
        if src_nid in in_out_degs: in_out_degs[src_nid][1] += len(neighbours)
    to_remove = set(i for i in range(len(old_walks), n_id) if in_out_degs[i][0] <= 0 or in_out_degs[i][1] <= 0)
    adj_list.remove_nodes(to_remove)
    for n in to_remove:
        del n2s_ghost[n]
    removed = len(to_remove)
    print("Final number of nodes:", n_id-removed)

    if added_edges_h1_count or added_edges_h2_count or (added_nodes_h1_count+added_nodes_h2_count > removed):
        return adj_list, n2s_ghost
    else:
        return None, None

def deduplicate(adj_list, old_walks):
    """
    De-duplicates edges. Duplicates are possible because a node can connect to multiple nodes in a single walk/key node.

    1. For all duplicates, I choose the edge that causes less bases to be discarded. 
    For edges between key nodes and ghost nodes this is simple, but for Key Node -> Key Node the counting is slightly more complicated. 
    """
    n_old_walks = len(old_walks)

    for new_src_nid, connected in adj_list.adj_list.items():
        dup_checker = {}
        for neigh in connected:
            new_dst_nid = neigh.new_dst_nid
            if new_dst_nid not in dup_checker:
                dup_checker[new_dst_nid] = neigh
            else:
                # duplicate is found
                og = dup_checker[new_dst_nid]
                if new_src_nid < n_old_walks and new_dst_nid < n_old_walks: # both are walks
                    walk_src, walk_dst = old_walks[new_src_nid], old_walks[new_dst_nid]
                    start_counting = None
                    score = 0
                    for i in reversed(walk_src):
                        if i == og.old_src_nid:
                            if start_counting: break # both old and new have been found, and score updated
                            start_counting = '+'
                        elif i == neigh.old_src_nid:
                            if start_counting: break # both old and new have been found, and score updated
                            start_counting = '-'

                        if start_counting == '+':
                            score += 1
                        elif start_counting == '-':
                            score -= 1

                    start_counting = None
                    for i in walk_dst:
                        if i == og.old_dst_nid:
                            if start_counting: break # both old and new have been found, and score updated
                            start_counting = '+'
                        elif i == neigh.old_dst_nid:
                            if start_counting: break # both old and new have been found, and score updated
                            start_counting = '-'

                        if start_counting == '+':
                            score += 1
                        elif start_counting == '-':
                            score -= 1

                    if score < 0: # if score is < 0, new is better
                        dup_checker[new_dst_nid] = neigh
                elif new_src_nid < n_old_walks:
                    walk = old_walks[new_src_nid]
                    for i in reversed(walk):
                        if i == neigh.old_src_nid: # new one is better, update dupchecker and remove old one from adj list
                            dup_checker[new_dst_nid] = neigh
                            break
                        elif i == og.old_src_nid:
                            break
                elif new_dst_nid < n_old_walks:
                    walk = old_walks[new_dst_nid]
                    for i in walk:
                        if i == neigh.old_dst_nid: # new one is better, update dupchecker and remove old one from adj list
                            dup_checker[new_dst_nid] = neigh
                            break
                        elif i == og.old_dst_nid:
                            break
                else:
                    raise ValueError("Duplicate edge between two non-walks found!")
                
        adj_list.adj_list[new_src_nid] = set(n for n in dup_checker.values())
    
    print("Final number of edges:", sum(len(x) for x in adj_list.adj_list.values()))
    return adj_list

def check_connection_cov(s1, s2, kmers, kmers_config):
    """
    Validates an edge based on relative coverage, calculated using k-mer frequency. 
    If the difference in coverage between two sequences is too great, the edge is rejected.
    """
    def get_avg_cov(seq):
        avg_cov, missed, total = kmers.get_seq_cov(seq)        
        if missed >= kmers_config['rep_threshold']*total: # the sequence only contained invalid kmers
            return -99999
        else:
            return avg_cov
    
    cov1, cov2 = get_avg_cov(s1), get_avg_cov(s2)
    is_invalid = cov1 == -99999 or cov2 == -99999
    cov_diff = abs(cov1-cov2)
    check = (cov1 == -99999 and cov2 == -99999) or cov_diff <= kmers_config['diff']*max(cov1,cov2)
    return cov_diff, check, is_invalid



def parse_ghost_for_repetitive_wrapper(args):
    return parse_ghost_for_repetitive(*args)

def parse_ghost_for_repetitive(nid, seq, kmers, threshold):
    _, missed, total = kmers.get_seq_cov(seq)
    return nid, missed >= threshold*total

def remove_repetitive_ghosts(adj_list, n2s_ghost, kmers, threshold):
    """
    Removes ghost nodes that are flagged as repetitive. (Threshold set by rep_threshold hyperparam). Uses multiprocessing.
    """
    full_args = [(nid, seq, kmers, threshold) for nid, seq in n2s_ghost.items()]
    to_remove = set()
    with Pool(40) as pool:
        results = pool.imap_unordered(parse_ghost_for_repetitive_wrapper, full_args)
        for nid, is_repetitive in tqdm(results, ncols=120, total=len(n2s_ghost)):
            if is_repetitive: to_remove.add(nid)
            
    adj_list.remove_nodes(to_remove)
    for n in to_remove:
        del n2s_ghost[n]

    print(f"Repetitive ghosts removed: {len(to_remove)}/{len(to_remove)+len(n2s_ghost)}")
    return adj_list, n2s_ghost



# def remove_repetitive_ghosts(adj_list, n2s_ghost, kmers, threshold):
#     to_remove = set()
#     for nid, seq in tqdm(n2s_ghost.items(), ncols=120):
#         _, missed, total = kmers.get_seq_cov(seq)
#         if missed >= threshold*total: to_remove.add(nid)

#     adj_list.remove_nodes(to_remove)
#     for n in to_remove:
#         del n2s_ghost[n]

#     print(f"Repetitive ghosts removed: {len(to_remove)}/{len(to_remove)+len(n2s_ghost)}")
#     return adj_list, n2s_ghost

def get_best_walk_coverage(adj_list, start_node, n_old_walks, telo_ref, n2s, n2s_ghost, kmers, kmers_config, penalty=None, visited_init=None):
    """
    Given a start node, recursively and greedily chooses the next node which has the lowest coverage difference.
    Note: dfs penalty is currently not being used, but leaving it here for possible future extension. the 0 returned by this function represents the penalty of the best walk.
    """
    def get_telo_info(node):
        if node >= n_old_walks: return None

        if telo_ref[node]['start']:
            return ('start', telo_ref[node]['start'])
        elif telo_ref[node]['end']:
            return ('end', telo_ref[node]['end'])
        else:
            return None

    def check_telo_compatibility(t1, t2):
        if t2 is None:
            return 0
        elif t1 is None and t2 is not None:
            return 1
        elif t1[0] != t2[0] and t1[1] != t2[1]: # The position and motif var must be different.
            return 1
        else:
            return -1

    if visited_init is None: visited_init = set()
    walk, n_key_nodes, visited, terminate = [start_node], 1, visited_init, False
    visited.add(start_node)
    c_node = start_node
    walk_telo = get_telo_info(start_node)

    while True:
        neighs = adj_list.get_neighbours(c_node)
        c_neighs, c_neighs_terminate = [], []

        for n in neighs:
            if n.new_dst_nid in visited: continue
            curr_telo = get_telo_info(n.new_dst_nid)
            telo_compatibility = check_telo_compatibility(walk_telo, curr_telo)
            if telo_compatibility < 0: 
                continue
            elif telo_compatibility > 0:
                c_neighs_terminate.append(n)
            else:
                c_neighs.append(n)

        if not c_neighs and not c_neighs_terminate: break

        best_diff, best_neigh = float('inf'), None
        # if difference is not being checked, and there is only one neighbour, select it immediately
        if kmers_config['diff'] >= 1 and len(c_neighs)+len(c_neighs_terminate) == 1:
            if c_neighs_terminate:
                best_neigh = c_neighs_terminate[0].new_dst_nid
                terminate = True
            else:
                best_neigh = c_neighs[0].new_dst_nid
                terminate = False
        else:
            for n in c_neighs_terminate:
                terminate = True
                s1 = n2s_ghost[c_node] if c_node >= n_old_walks else n2s[n.old_src_nid]
                s2 = n2s_ghost[n.new_dst_nid] if n.new_dst_nid >= n_old_walks else n2s[n.old_dst_nid]
                if n.ol_len + 100 + kmers_config['k'] > len(s1) or n.ol_len + 100 + kmers_config['k'] > len(s2): continue # there must be at least 100 kmers to calculate the coverage
                cov_diff, cov_check, is_invalid = check_connection_cov(s1[:n.ol_len], s2[n.ol_len:], kmers, kmers_config)
                if not is_invalid and not cov_check: continue
                if cov_diff < best_diff:
                    best_diff = cov_diff
                    best_neigh = n.new_dst_nid

            if best_neigh is None:
                for n in c_neighs:
                    terminate = False
                    s1 = n2s_ghost[c_node] if c_node >= n_old_walks else n2s[n.old_src_nid]
                    s2 = n2s_ghost[n.new_dst_nid] if n.new_dst_nid >= n_old_walks else n2s[n.old_dst_nid]
                    if n.ol_len + 100 + kmers_config['k'] > len(s1) or n.ol_len + 100 + kmers_config['k'] > len(s2): continue # there must be at least 100 kmers to calculate the coverage
                    cov_diff, cov_check, is_invalid = check_connection_cov(s1[:n.ol_len], s2[n.ol_len:], kmers, kmers_config)
                    if not is_invalid and not cov_check: continue
                    if cov_diff < best_diff:
                        best_diff = cov_diff
                        best_neigh = n.new_dst_nid

        if best_neigh is None: break
        walk.append(best_neigh)
        visited.add(best_neigh)
        if best_neigh < n_old_walks: n_key_nodes += 1
        c_node = best_neigh
        if terminate: break

    while walk[-1] >= n_old_walks:
        walk.pop()
    is_t2t = walk_telo and walk[-1] < n_old_walks and ((telo_ref[walk[-1]]['start'] and telo_ref[walk[-1]]['start'] != walk_telo[1]) or (telo_ref[walk[-1]]['end'] and telo_ref[walk[-1]]['end'] != walk_telo[1]))

    return walk, n_key_nodes, 0, is_t2t

def get_walks(adj_list, telo_ref, dfs_penalty, n2s, n2s_ghost, kmers, kmers_config, old_graph, old_walks, edges_full, decoding='default'):
    """
    Creates the new walks, priotising key nodes with telomeres.

    1. Key nodes with start and end telomeres in its sequence are removed beforehand.
    2. We separate out all key nodes that have telomeres. For each of these key nodes, we find the best walk starting from that node. The best walk out of all is then saved, and the process is repeated until all key nodes are used.
        i. Depending on whether the telomere is in the start or end of the sequence, we search forwards or in reverse. We create a reversed version of the adj_list for this.
    3. We then repeat the above step for all key nodes without telomere information that are still unused.
    """ 

    n_old_walks = len(old_walks)
    def get_best_walk(adj_list, start_node, visited_init=None):
        if decoding == 'coverage':
            return get_best_walk_coverage(adj_list, start_node, n_old_walks, telo_ref, n2s, n2s_ghost, kmers, kmers_config, visited_init=visited_init)
        else:
            raise ValueError("Invalid decoding param!")

    # Initialise reverse adj list
    new_walks = []
    walk_ids, temp_adj_list = list(range(len(old_walks))), deepcopy(adj_list)
    rev_adj_list = AdjList()
    for edges in temp_adj_list.adj_list.values():
        for e in edges:
            rev_adj_list.add_edge(Edge(
                new_src_nid=e.new_dst_nid,
                new_dst_nid=e.new_src_nid,
                old_src_nid=e.old_dst_nid,
                old_dst_nid=e.old_src_nid,
                prefix_len=e.prefix_len,
                ol_len=e.ol_len,
                ol_sim=e.ol_sim
            ))

    # Remove all old walks that have both start and end telo regions
    for walk_id, v in telo_ref.items():
        if v['start'] and v['end']:
            new_walks.append([walk_id])
            temp_adj_list.remove_node(walk_id)
            rev_adj_list.remove_node(walk_id)
            walk_ids.remove(walk_id)

    # Split walks into those with telomeric regions and those without
    telo_walk_ids, non_telo_walk_ids = [], []
    for i in walk_ids:
        if telo_ref[i]['start'] or telo_ref[i]['end']:
            telo_walk_ids.append(i)
        else:
            non_telo_walk_ids.append(i)
        
    # Generate walks for walks with telomeric regions first
    while telo_walk_ids:
        print(f"Number of telo walk ids left: {len(telo_walk_ids)}", end='\r')
        best_walk, best_key_nodes, best_penalty, is_best_t2t = [], 0, 0, False
        for walk_id in telo_walk_ids: # the node_id is also the index        
            if telo_ref[walk_id]['start']:
                curr_walk, curr_key_nodes, curr_penalty, is_curr_t2t = get_best_walk(temp_adj_list, walk_id)
            else:
                curr_walk, curr_key_nodes, curr_penalty, is_curr_t2t = get_best_walk(rev_adj_list, walk_id)
                curr_walk.reverse()

            if is_best_t2t and not is_curr_t2t: continue
            if curr_key_nodes > best_key_nodes or (curr_key_nodes == best_key_nodes and curr_penalty < best_penalty) or (is_curr_t2t and not is_best_t2t):
                is_best_t2t = is_curr_t2t
                best_key_nodes = curr_key_nodes
                best_walk = curr_walk
                best_penalty = curr_penalty

        for w in best_walk:
            temp_adj_list.remove_node(w)
            rev_adj_list.remove_node(w)
            if w < n_old_walks: 
                if w in telo_walk_ids:
                    telo_walk_ids.remove(w)
                else:
                    non_telo_walk_ids.remove(w)

        new_walks.append(best_walk)

    assert len(telo_walk_ids) == 0, "Telomeric walks not all used!"

    # Generate walks for the rest
    while non_telo_walk_ids:
        print(f"Number of non telo walk ids left: {len(non_telo_walk_ids)}", end='\r')
        best_walk, best_key_nodes, best_penalty = [], 0, 0
        for walk_id in non_telo_walk_ids: # the node_id is also the index
            curr_walk, curr_key_nodes, curr_penalty, _ = get_best_walk(temp_adj_list, walk_id)
            visited_init = set(curr_walk[1:]) if len(curr_walk) > 1 else set()
            curr_walk_rev, curr_key_nodes_rev, curr_penalty_rev, _ = get_best_walk(rev_adj_list, walk_id, visited_init=visited_init)
            curr_walk_rev.reverse(); curr_walk_rev = curr_walk_rev[:-1]; curr_walk_rev.extend(curr_walk); curr_walk = curr_walk_rev
            curr_key_nodes += (curr_key_nodes_rev-1)
            curr_penalty += curr_penalty_rev
            if curr_key_nodes > best_key_nodes or (curr_key_nodes == best_key_nodes and curr_penalty < best_penalty):
                best_key_nodes = curr_key_nodes
                best_walk = curr_walk
                best_penalty = curr_penalty

        for w in best_walk:
            temp_adj_list.remove_node(w)
            rev_adj_list.remove_node(w)
            if w < n_old_walks: non_telo_walk_ids.remove(w)

        new_walks.append(best_walk)

    print(f"New walks generated! n new walks: {len(new_walks)}")
    return new_walks

def get_contigs(old_walks, new_walks, adj_list, n2s, n2s_ghost, g, edges_full):
    """
    Recreates the contigs given the new walks. 
    
    1. Pre-processes the new walks to break down key nodes into the original nodes based on their connections.
    2. Converts the nodes into the contigs. This is done in the same way as in the GNNome pipeline.
    """
    n_old_walks = len(old_walks)
    walk_seqs, walk_prefix_lens = [], []
    for i, walk in enumerate(new_walks):
        c_seqs, c_prefix_lens = [], []

        for j, node in enumerate(walk):
            if node >= n_old_walks: # Node is a ghost node, it can never be the first or last node in a walk
                c_seqs.append(str(n2s_ghost[node]))
                c_prefix_lens.append(adj_list.get_edge(node, walk[j+1]).prefix_len)
            else: # Node is an original walk
                old_walk = old_walks[node]
                if j == 0:
                    start = 0
                else:
                    curr_edge = adj_list.get_edge(walk[j-1], node)
                    start = old_walk.index(curr_edge.old_dst_nid)
                
                if j+1 == len(walk):
                    end = len(old_walk)-1
                    last_prefix_len = None
                else:
                    curr_edge = adj_list.get_edge(node, walk[j+1])
                    end = old_walk.index(curr_edge.old_src_nid)
                    last_prefix_len = curr_edge.prefix_len

                for k in range(start, end+1):
                    c_seqs.append(str(n2s[old_walk[k]]))
                    if k != end:
                        c_prefix_lens.append(g.edata['prefix_length'][edges_full[(old_walk[k], old_walk[k+1])]])

                if last_prefix_len: c_prefix_lens.append(last_prefix_len)

        walk_seqs.append(c_seqs)
        walk_prefix_lens.append(c_prefix_lens)

    contigs = []
    for i, seqs in enumerate(walk_seqs):
        prefix_lens = walk_prefix_lens[i]
        assert len(seqs) == len(prefix_lens)+1, "Error in generating contigs. Please report this, thank you!"
        c_contig = []
        for j, seq in enumerate(seqs[:-1]):
            c_contig.append(seq[:prefix_lens[j]])
        c_contig.append(seqs[-1])

        c_contig = Seq.Seq(''.join(c_contig))
        c_contig = SeqIO.SeqRecord(c_contig)
        c_contig.id = f'contig_{str(i+1).zfill(10)}'
        c_contig.description = f'length={str(len(c_contig)).zfill(15)}'
        contigs.append(c_contig)

    return contigs

def postprocess(name, hyperparams, paths, aux, time_start):
    """
    (\(\        \|/        /)/)
    (  ^.^)     -o-     (^.^  )
    o_(")(")    /|\    (")(")_o

    Performs scaffolding on GNNome's walks using information from PAF, GFA, and telomeres.
    Currently, only uses info from 1-hop neighbourhood of original graph. Any two walks are at most connected by a single ghost node. Also, all added ghost nodes must have at least one incoming and one outgoing edge to a walk.
    
    Summary of the pipeline (details can be found in the respective functions):
    1. Generates telomere information, then chops walks accordingly.
    2. Compresses each GNNome walk into a single node, then adds 'ghost' nodes and edges using information from PAF and GFA.
    3. Decodes the new sequences using DFS and telomere information.
    4. Regenerates contigs and calculates metrics.
    """
    print(f"\n===== Postprocessing {name} =====")
    hyperparams_str = ""
    for k, v in hyperparams.items():
        hyperparams_str += f"{k}: {v}, "
    print(hyperparams_str[:-2]+"\n")
    walks, n2s, r2n, paf_data, old_graph, edges_full, hifi_r2s, ul_r2s, kmers = aux['walks'], aux['n2s'], aux['r2n'], aux['paf_data'], aux['old_graph'], aux['edges_full'], aux['hifi_r2s'], aux['ul_r2s'], aux['kmers']

    print(f"Chopping old walks... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    if hyperparams['use_telomere_info']:
        rep1, rep2 = hyperparams['telo_motif'][0], hyperparams['telo_motif'][1]
        walks, telo_ref = chop_walks_seqtk(walks, n2s, old_graph, edges_full, rep1, rep2, paths['seqtk'])
    else:
        telo_ref = { i:{'start':None, 'end':None} for i in range(len(walks)) }

    print(f"Adding ghost nodes and edges... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    adj_list, n2s_ghost = add_ghosts(
        old_walks=walks,
        paf_data=paf_data,
        r2n=r2n,
        hifi_r2s=hifi_r2s,
        ul_r2s=ul_r2s,
        walk_valid_p=hyperparams['walk_valid_p'][0],
        hop=hyperparams['hop']
    )
    if adj_list is None and n2s_ghost is None:
        print("No suitable nodes and edges found to add to these walks. Returning...")
        return
    
    print(f"Removing repetitive ghosts... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    adj_list, n2s_ghost = remove_repetitive_ghosts(adj_list, n2s_ghost, kmers, hyperparams['kmers']['rep_threshold'])

    # Remove duplicate edges between nodes. If there are multiple connections between a walk and another node/walk, we choose the best one.
    # This could probably have been done while adding the edges in. However, to avoid confusion, i'm doing this separately.
    print(f"De-duplicating edges... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    adj_list = deduplicate(adj_list, walks)

    print(f"Generating new walks with {hyperparams['decoding']} decoding... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    new_walks = get_walks(
        adj_list=adj_list,
        telo_ref=telo_ref,
        dfs_penalty=hyperparams['dfs_penalty'],
        n2s=n2s,
        n2s_ghost=n2s_ghost,
        kmers=kmers,
        kmers_config=hyperparams['kmers'],
        old_graph=old_graph,
        old_walks=walks,
        edges_full=edges_full,
        decoding=hyperparams['decoding']
    )
    analyse_graph(adj_list, telo_ref, new_walks, paths['save'], 0)

    print(f"Generating contigs... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    contigs = get_contigs(walks, new_walks, adj_list, n2s, n2s_ghost, old_graph, edges_full)

    print(f"Calculating final assembly metrics... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    asm_metrics(contigs, paths['save'], paths['ref'], paths['minigraph'], paths['paftools'])
    t2t_metrics(paths['save'], paths['t2t_chr'], paths['ref'], hyperparams['telo_motif'][0])
    if paths['yak1'] and paths['yak2']: yak_metrics(paths['save'], paths['yak1'], paths['yak2'], paths['yak'])

    print(f"Run finished! (Time: {timedelta_to_str(datetime.now() - time_start)})")
    return

def run_postprocessing(config):
    time_start = datetime.now()

    postprocessing_config = config['postprocessing']
    postprocessing_config['kmers'] = config['misc']['kmers']
    genomes = config['run']['postprocessing']['genomes']
    for genome in genomes:
        postprocessing_config['telo_motif'] = config['genome_info'][genome]['telo_motifs']
        paths = config['genome_info'][genome]['paths']
        paths.update(config['misc']['paths'])

        print("Loading files...")
        aux = {}
        with open(paths['walks'], 'rb') as f:
            aux['walks'] = pickle.load(f)
            print(f"Walks loaded... (Time: {timedelta_to_str(datetime.now() - time_start)})")
        with open(paths['n2s'], 'rb') as f:
            aux['n2s'] = pickle.load(f)
            print(f"N2S loaded... (Time: {timedelta_to_str(datetime.now() - time_start)})")
        with open(paths['r2n'], 'rb') as f:
            aux['r2n'] = pickle.load(f)
            print(f"R2N loaded... (Time: {timedelta_to_str(datetime.now() - time_start)})")
        with open(paths['paf_processed'], 'rb') as f:
            aux['paf_data'] = pickle.load(f)
            print(f"PAF data loaded... (Time: {timedelta_to_str(datetime.now() - time_start)})")

        kmers = KmerManager(k=postprocessing_config['kmers']['k'])
        kmers.generate_freqs(paths['hifiasm'])
        aux['kmers'] = kmers
        print(f"Kmers data loaded... (Time: {timedelta_to_str(datetime.now() - time_start)})")

        old_graph = dgl.load_graphs(paths['graph']+f'{genome}.dgl')[0][0]
        # Create a list of all edges
        edges_full = {}  # I dont know why this is necessary. but when cut transitives some edges are wrong otherwise. (This is from Martin's script)
        for idx, (src, dst) in enumerate(zip(old_graph.edges()[0], old_graph.edges()[1])):
            src, dst = src.item(), dst.item()
            edges_full[(src, dst)] = idx
        aux['old_graph'] = old_graph
        aux['edges_full'] = edges_full
        print(f"Graph data loaded... (Time: {timedelta_to_str(datetime.now() - time_start)})")

        aux['hifi_r2s'] = Fasta(paths['ec_reads'])
        aux['ul_r2s'] = Fasta(paths['ul_reads']) if paths['ul_reads'] else None
        print(f'R2S data loaded... (Time: {timedelta_to_str(datetime.now() - time_start)})')

        postprocess(genome, hyperparams=postprocessing_config, paths=paths, aux=aux, time_start=time_start)
        # for diff in [0.75, 1]:
        #     postprocessing_config['kmers']['diff'] = diff
        #     postprocess(genome, hyperparams=postprocessing_config, paths=paths, aux=aux, time_start=time_start)