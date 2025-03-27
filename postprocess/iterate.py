from Bio import Seq
from copy import deepcopy
from tqdm import tqdm

from misc.utils import analyse_graph, get_seqs
from .custom_graph import AdjList, Edge

########################### COPIED FROM postprocess.py TO PREVENT CIRCULAR IMPORT #############################
###############################################################################################################

COV_MEMO = {} # memoises the coverage differences between two seqs

def check_connection_cov(s1, s2, kmers, k, diff, memoize=True):
    """
    Validates an edge based on relative coverage, calculated using k-mer frequency. 
    If the difference in coverage between two sequences is too great, the edge is rejected.
    """
    if (s1, s2) in COV_MEMO:
        cov_diff, check, is_invalid = COV_MEMO[(s1, s2)]
        return cov_diff, check, is_invalid
    if (s2, s1) in COV_MEMO:
        cov_diff, check, is_invalid = COV_MEMO[(s2, s1)]
        return cov_diff, check, is_invalid

    def get_avg_cov(seq):
        kmer_list = [seq[i:i+k] for i in range(len(seq)-k+1)]

        total_cov, missed = 0, 0
        for c_kmer in kmer_list:
            if c_kmer not in kmers: c_kmer = str(Seq.Seq(c_kmer).reverse_complement())
            if c_kmer not in kmers: # if it is still not in kmers, that means it was filtered out due to missing solid threshold
                missed += 1
            else:
                total_cov += kmers[c_kmer]
            
        if missed > 0.8*len(kmer_list): # the sequence only contained invalid kmers
            return -99999
        else:
            return total_cov/(len(kmer_list)-missed)
    
    cov1, cov2 = get_avg_cov(s1), get_avg_cov(s2)
    is_invalid = cov1 == -99999 or cov2 == -99999
    cov_diff = abs(cov1-cov2)
    check = (cov1 == -99999 and cov2 == -99999) or cov_diff <= diff*max(cov1,cov2)
    if memoize: COV_MEMO[(s1, s2)] = (cov_diff, check, is_invalid)
    return cov_diff, check, is_invalid

def rename_ghosts(iteration, new_walks, n2s_ghost, n_old_walks):
    new_n2s_ghost = {}
    for nw in new_walks:
        for i, n in enumerate(nw):
            if n < n_old_walks: continue
            new_name = f"{iteration}-{n}"
            new_n2s_ghost[new_name] = n2s_ghost[n]
            nw[i] = new_name

    return new_walks, new_n2s_ghost

###############################################################################################################
###############################################################################################################

def get_new_telo_ref(new_walks, telo_ref):
    new_telo_ref = {}

    for i, walk in enumerate(new_walks):
        if len(walk) == 1:
            new_telo_ref[i] = telo_ref[walk[0]]
        else:
            first, last = telo_ref[walk[0]], telo_ref[walk[-1]]
            # we can assume that the start node of walks will only have telo info in its 'start'. same for the end node
            new_telo_ref[i] = {
                'start' : first['start'],
                'end' : last['end']
            }

    return new_telo_ref

def get_new_graph(new_walks, old_walks, paf_data, r2n, hifi_r2s, ul_r2s, walk_valid_p):
    n_old_walks = len(old_walks)
    n2n_start, n2n_end = {}, {}

    def add_to_n2n_start_end(old_walk_id, new_nid, start=False, end=False):
        walk = old_walks[old_walk_id]
        if len(walk) == 1:
            if start: n2n_start[walk[0]] = new_nid
            if end: n2n_end[walk[0]] = new_nid
        else:
            cutoff = int(max(1, len(walk) // (1/walk_valid_p)))
            first_part, last_part = walk[:cutoff], walk[-cutoff:]
            if start:
                for n in first_part:
                    n2n_start[n] = new_nid
            if end:
                for n in last_part:
                    n2n_end[n] = new_nid
        return
    
    for new_nid, new_walk in enumerate(new_walks):
        if len(new_walk) == 1:
            add_to_n2n_start_end(new_walk[0], new_nid, True, True)
        else:
            assert new_walk[0] < n_old_walks and new_walk[-1] < n_old_walks, "New walk does not begin and end with seq nodes!"
            add_to_n2n_start_end(new_walk[0], new_nid, start=True)
            add_to_n2n_start_end(new_walk[-1], new_nid, end=True)

    new_adj_list = AdjList()
    
    print(f"Adding edges between existing nodes...")
    valid_src, valid_dst, prefix_lens, ol_lens, ol_sims, ghost_data = paf_data['ghost_edges']['valid_src'], paf_data['ghost_edges']['valid_dst'], paf_data['ghost_edges']['prefix_len'], paf_data['ghost_edges']['ol_len'], paf_data['ghost_edges']['ol_similarity'], paf_data['ghost_nodes']['hop_1']
    added_edges_count = 0
    for i in range(len(valid_src)):
        src, dst, prefix_len, ol_len, ol_sim = valid_src[i], valid_dst[i], prefix_lens[i], ol_lens[i], ol_sims[i]
        if src in n2n_end and dst in n2n_start:
            if n2n_end[src] == n2n_start[dst]: continue # ignore self-edges
            added_edges_count += 1
            new_adj_list.add_edge(Edge(
                new_src_nid=n2n_end[src], 
                new_dst_nid=n2n_start[dst], 
                old_src_nid=src, 
                old_dst_nid=dst, 
                prefix_len=prefix_len, 
                ol_len=ol_len, 
                ol_sim=ol_sim
            ))
    print("Added edges:", added_edges_count)

    print(f"Adding ghost nodes...")
    new_nid = len(new_walks) # ghost nodes start getting ids from this value 
    n2s_ghost = {}
    added_nodes_count = 0
    for orient in ['+', '-']:
        print("Orient:", orient)
        for read_id, data in tqdm(ghost_data[orient].items(), ncols=120):
            curr_out_neighbours, curr_in_neighbours = set(), set()

            for i, out_read_id in enumerate(data['outs']):
                out_n_id = r2n[out_read_id[0]][0] if out_read_id[1] == '+' else r2n[out_read_id[0]][1]
                if out_n_id not in n2n_start: continue
                curr_out_neighbours.add((out_n_id, data['prefix_len_outs'][i], data['ol_len_outs'][i], data['ol_similarity_outs'][i]))

            for i, in_read_id in enumerate(data['ins']):
                in_n_id = r2n[in_read_id[0]][0] if in_read_id[1] == '+' else r2n[in_read_id[0]][1] 
                if in_n_id not in n2n_end: continue
                curr_in_neighbours.add((in_n_id, data['prefix_len_ins'][i], data['ol_len_ins'][i], data['ol_similarity_ins'][i]))

            # ghost nodes are only useful if they have both at least one outgoing and one incoming edge
            if not curr_out_neighbours or not curr_in_neighbours: continue
            if all(x==n2n_start[next(iter(curr_out_neighbours))[0]] for x in [n2n_start[n[0]] for n in curr_out_neighbours]+[n2n_end[n[0]] for n in curr_in_neighbours]): continue

            for n in curr_out_neighbours:
                new_adj_list.add_edge(Edge(
                    new_src_nid=new_nid,
                    new_dst_nid=n2n_start[n[0]],
                    old_src_nid=None,
                    old_dst_nid=n[0],
                    prefix_len=n[1],
                    ol_len=n[2],
                    ol_sim=n[3]
                ))
            for n in curr_in_neighbours:
                new_adj_list.add_edge(Edge(
                    new_src_nid=n2n_end[n[0]],
                    new_dst_nid=new_nid,
                    old_src_nid=n[0],
                    old_dst_nid=None,
                    prefix_len=n[1],
                    ol_len=n[2],
                    ol_sim=n[3]
                ))

            if orient == '+':
                seq, _ = get_seqs(read_id, hifi_r2s, ul_r2s)
            else:
                _, seq = get_seqs(read_id, hifi_r2s, ul_r2s)
            n2s_ghost[new_nid] = seq
            new_nid += 1; added_nodes_count += 1
    print("Number of nodes added from PAF:", added_nodes_count)

    print("Final number of nodes:", new_nid)
    if added_edges_count or added_nodes_count:
        return new_adj_list, n2s_ghost
    else:
        return None, None
    
def deduplicate(adj_list, new_walks, old_walks):
    """
    De-duplicates edges. Duplicates are possible because a node can connect to multiple nodes in a single walk/key node.

    1. For all duplicates, I choose the edge that causes less bases to be discarded. 
    For edges between key nodes and ghost nodes this is simple, but for Key Node -> Key Node the counting is slightly more complicated. 
    """
    n_new_walks = len(new_walks)

    for new_src_nid, connected in adj_list.adj_list.items():
        dup_checker = {}
        for neigh in connected:
            new_dst_nid = neigh.new_dst_nid
            if new_dst_nid not in dup_checker:
                dup_checker[new_dst_nid] = neigh
            else:
                # duplicate is found
                og = dup_checker[new_dst_nid]
                if new_src_nid < n_new_walks and new_dst_nid < n_new_walks: # both are walks
                    walk_src, walk_dst = old_walks[new_walks[new_src_nid][-1]], old_walks[new_walks[new_dst_nid][0]]
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
                elif new_src_nid < n_new_walks:
                    walk = old_walks[new_walks[new_src_nid][-1]]
                    for i in reversed(walk):
                        if i == neigh.old_src_nid: # new one is better, update dupchecker and remove old one from adj list
                            dup_checker[new_dst_nid] = neigh
                            break
                        elif i == og.old_src_nid:
                            break
                elif new_dst_nid < n_new_walks:
                    walk = old_walks[new_walks[new_dst_nid][0]]
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

def filter_edges(adj_list, ol_len_cutoff, ol_sim_cutoff):
    new_adj_list = AdjList()
    n_removed = 0
    for edges in adj_list.adj_list.values():
        for e in edges:
            if e.ol_len < ol_len_cutoff or e.ol_sim < ol_sim_cutoff: 
                n_removed += 1
                continue
            new_adj_list.add_edge(e)

    print("Number of edges removed:", n_removed)
    return new_adj_list

def get_best_walk_coverage(adj_list, start_node, telo_ref, n2s, kmers, kmers_config, graph, old_walks, new_walks, edges_full, reverse, visited_init=None):
    """
    Given a start node, recursively and greedily chooses the next sequence node (performs 1 step lookahead to skip over the ghost) which has the lowest coverage difference.
    """
    
    n_old_walks = len(old_walks)

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

    def get_surrounding_seq(walk_id, old_node_id):
        seq = n2s[old_node_id]

        old_walk = old_walks[walk_id]
        ind = old_walk.index(old_node_id)
        if ind > 0:
            prev_node_id = old_walk[ind-1]
            prefix_len = graph.edata['prefix_length'][edges_full[(prev_node_id, old_node_id)]]
            seq = n2s[prev_node_id][:prefix_len] + seq
        if ind+1 < len(old_walk):
            next_node_id = old_walk[ind+1]
            prefix_len = graph.edata['prefix_length'][edges_full[(old_node_id, next_node_id)]]
            seq = seq[:prefix_len] + n2s[next_node_id]

        return seq
        
    if visited_init is None: visited_init = set()
    walk, n_key_nodes, visited = [start_node], 1, visited_init
    visited.add(start_node)
    c_node = start_node
    walk_telo = get_telo_info(start_node)
    while True:
        assert c_node < n_old_walks, "Current node is not a sequence node. Please report this bug, thanks!"
        ghost_neighs = adj_list.get_neighbours(c_node)

        # performs 1-hop lookahead
        best_diff, best_g_neigh, best_s_neigh = float('inf'), None, None
        for g in ghost_neighs:
            if g.new_dst_nid in visited: continue
            seq_neighs = adj_list.get_neighbours(g.new_dst_nid) # get all neighbours of n, which is a ghost
            new_walk = new_walks[g.new_src_nid][0] if reverse else new_walks[g.new_src_nid][-1]
            s1 = get_surrounding_seq(new_walk, g.old_src_nid)    
            for s in seq_neighs:
                if s.new_dst_nid in visited: continue
                curr_telo = get_telo_info(s.new_dst_nid)
                if check_telo_compatibility(walk_telo, curr_telo) < 0: continue
                new_walk = new_walks[s.new_dst_nid][-1] if reverse else new_walks[s.new_dst_nid][0]
                s2 = get_surrounding_seq(new_walk, s.old_dst_nid)
                diff, cov_check, is_invalid = check_connection_cov(s1, s2, kmers, kmers_config['k'], kmers_config['diff'])
                if not is_invalid and not cov_check: continue
                if diff < best_diff:
                    best_diff = diff
                    best_g_neigh = g.new_dst_nid
                    best_s_neigh = s.new_dst_nid

        if best_g_neigh is None: break
        walk.append(best_g_neigh); walk.append(best_s_neigh)
        visited.add(best_g_neigh); visited.add(best_s_neigh)
        n_key_nodes += 1
        c_node = best_s_neigh

        curr_telo = get_telo_info(best_s_neigh)
        if check_telo_compatibility(walk_telo, curr_telo) > 0: break

    is_t2t = walk_telo and walk[-1] < n_old_walks and ((telo_ref[walk[-1]]['start'] and telo_ref[walk[-1]]['start'] != walk_telo[1]) or (telo_ref[walk[-1]]['end'] and telo_ref[walk[-1]]['end'] != walk_telo[1]))

    return walk, n_key_nodes, is_t2t

def get_walks(adj_list, telo_ref, old_walks, new_walks, n2s, kmers, kmers_config, old_graph, edges_full):

    def get_best_walk(adj_list, start_node, reverse, visited_init=None):
        return get_best_walk_coverage(adj_list, start_node, telo_ref, n2s, kmers, kmers_config, old_graph, old_walks, new_walks, edges_full, reverse, visited_init=visited_init)
        
    # Initialise reverse adj list
    new_new_walks = []
    temp_walk_ids, temp_adj_list = list(telo_ref.keys()), deepcopy(adj_list)
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
            new_new_walks.append([walk_id])
            temp_adj_list.remove_node(walk_id)
            rev_adj_list.remove_node(walk_id)
            temp_walk_ids.remove(walk_id)

    # Split walks into those with telomeric regions and those without
    telo_walk_ids, non_telo_walk_ids = [], []
    for i in temp_walk_ids:
        if telo_ref[i]['start'] or telo_ref[i]['end']:
            telo_walk_ids.append(i)
        else:
            non_telo_walk_ids.append(i)

    # Generate walks for walks with telomeric regions first
    n_new_walks = len(new_walks)
    while telo_walk_ids:
        print(f"Number of telo walk ids left: {len(telo_walk_ids)}", end='\r')
        best_walk, best_key_nodes, is_best_t2t = [], 0, False
        for walk_id in telo_walk_ids: # the node_id is also the index        
            if telo_ref[walk_id]['start']:
                curr_walk, curr_key_nodes, is_curr_t2t = get_best_walk(temp_adj_list, walk_id, False)
            else:
                curr_walk, curr_key_nodes, is_curr_t2t = get_best_walk(rev_adj_list, walk_id, True)
                curr_walk.reverse()

            if is_best_t2t and not is_curr_t2t: continue
            if curr_key_nodes > best_key_nodes or (is_curr_t2t and not is_best_t2t):
                is_best_t2t = is_curr_t2t
                best_key_nodes = curr_key_nodes
                best_walk = curr_walk

        for w in best_walk:
            temp_adj_list.remove_node(w)
            rev_adj_list.remove_node(w)
            if w < n_new_walks: 
                if w in telo_walk_ids:
                    telo_walk_ids.remove(w)
                else:
                    non_telo_walk_ids.remove(w)

        new_new_walks.append(best_walk)

    assert len(telo_walk_ids) == 0, "Telomeric walks not all used!"

    # Generate walks for the rest
    while non_telo_walk_ids:
        print(f"Number of non telo walk ids left: {len(non_telo_walk_ids)}", end='\r')
        best_walk, best_key_nodes = [], 0
        for walk_id in non_telo_walk_ids: # the node_id is also the index
            curr_walk, curr_key_nodes, _ = get_best_walk(temp_adj_list, walk_id, False)
            visited_init = set(curr_walk[1:]) if len(curr_walk) > 1 else set()
            curr_walk_rev, curr_key_nodes_rev, _ = get_best_walk(rev_adj_list, walk_id, True, visited_init=visited_init)
            curr_walk_rev.reverse(); curr_walk_rev = curr_walk_rev[:-1]; curr_walk_rev.extend(curr_walk); curr_walk = curr_walk_rev
            curr_key_nodes += (curr_key_nodes_rev-1)
            if curr_key_nodes > best_key_nodes:
                best_key_nodes = curr_key_nodes
                best_walk = curr_walk

        for w in best_walk:
            temp_adj_list.remove_node(w)
            rev_adj_list.remove_node(w)
            if w < n_new_walks: non_telo_walk_ids.remove(w)

        new_new_walks.append(best_walk)

    print(f"New walks generated! n new walks: {len(new_new_walks)}")
    return new_new_walks

def decompress_walks(new_new_walks, new_walks):
    res = []
    n_new_walks = len(new_walks)
    for nnw in new_new_walks:
        curr_walk = []
        for n in nnw:
            if "-" in str(n): # is a ghost
                curr_walk.append(n)
            else:
                assert int(n) < n_new_walks, "Invalid new sequence node id found!"
                curr_walk.extend(new_walks[n])

        res.append(curr_walk)

    return res
                
def iterate_postprocessing(aux, hyperparams, paths, new_walks, telo_ref, n2s_ghost, edges_full, filtering_config):
    old_walks, n2s, r2n, paf_data, hifi_r2s, ul_r2s, kmers, old_graph = aux['walks'], aux['n2s'], aux['r2n'], aux['paf_data'], aux['hifi_r2s'], aux['ul_r2s'], aux['kmers'], aux['old_graph']
    adj_lists, n2nns = [], []

    print("Iteration 0 walks:", [w for w in new_walks if len(w)>1])

    for iteration in range(1, hyperparams['iterations']):
        print(f"Starting iteration {iteration}...")
        n2nn = {}
        for i, nw in enumerate(new_walks):
            n2nn[nw[0]] = i
            n2nn[nw[-1]] = i
        n2nns.append(n2nn)

        new_telo_ref = get_new_telo_ref(new_walks, telo_ref)

        new_adj_list, curr_n2s_ghost = get_new_graph(
            new_walks=new_walks,
            old_walks=old_walks,
            paf_data=paf_data,
            r2n=r2n,
            hifi_r2s=hifi_r2s,
            ul_r2s=ul_r2s,
            walk_valid_p=hyperparams['walk_valid_p'][iteration]
        )
        if new_adj_list == None and curr_n2s_ghost == None:
            print("No suitable nodes and edges found to add to these walks. Skipping iteration...")
            adj_lists.append(None)
            continue

        new_adj_list = filter_edges(new_adj_list, filtering_config['ol_len_cutoff'], filtering_config['ol_sim_cutoff'])
        new_adj_list = deduplicate(new_adj_list, new_walks, old_walks)
        adj_lists.append(new_adj_list)

        new_new_walks = get_walks(
            adj_list=new_adj_list,
            telo_ref=new_telo_ref,
            old_walks=old_walks,
            new_walks=new_walks,
            n2s=n2s,
            kmers=kmers,
            kmers_config=hyperparams['kmers'],
            old_graph=old_graph,
            edges_full=edges_full
        )
        analyse_graph(new_adj_list, new_telo_ref, new_new_walks, paths['save'], iteration)

        # Convert all ghost node ids in new_new_walks and n2s_ghost in this iteration from .e.g 35 -> 1-35 for iteration 1.
        new_new_walks, curr_n2s_ghost = rename_ghosts(iteration, new_new_walks, curr_n2s_ghost, len(new_walks))
        assert not (set(n2s_ghost) & set(curr_n2s_ghost)), "Duplicate keys in n2s_ghost found!"
        n2s_ghost.update(curr_n2s_ghost)

        # Replace new_walks with the decompressed version of our new new walks. It should follow the same format so the process is repeatable
        new_walks = decompress_walks(new_new_walks, new_walks)

        print(f"Iteration {iteration} walks:", [w for w in new_walks if len(w)>1])
    
    return new_walks, n2s_ghost, adj_lists, n2nns