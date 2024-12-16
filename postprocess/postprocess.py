import dgl, os, pickle, random, subprocess, torch
from Bio import Seq, SeqIO
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import numpy as np
from pyfaidx import Fasta

from generate_baseline.gnnome_decoding import preprocess_graph
from generate_baseline.SymGatedGCN import SymGatedGCNModel
from misc.utils import asm_metrics, get_seqs, timedelta_to_str, yak_metrics

SCORE_CUTOFF = float('-inf')

class Edge():
    def __init__(self, new_src_nid, new_dst_nid, old_src_nid, old_dst_nid, prefix_len, ol_len, ol_sim):
        self.new_src_nid = new_src_nid
        self.new_dst_nid = new_dst_nid
        self.old_src_nid = old_src_nid
        self.old_dst_nid = old_dst_nid
        self.prefix_len = prefix_len
        self.ol_len = ol_len
        self.ol_sim = ol_sim

class AdjList():
    """
    Maps new_src_nid to edges.
    """

    def __init__(self):
        self.adj_list = defaultdict(set)

    def add_edge(self, edge):
        self.adj_list[edge.new_src_nid].add(edge)

    def remove_edge(self, edge):
        neighbours = self.adj_list[edge.new_src_nid]
        if edge not in neighbours:
            print("WARNING: Removing an edge that does not exist!")
        self.adj_list[edge.new_src_nid].discard(edge)
        if not self.adj_list[edge.new_src_nid]: del self.adj_list[edge.new_src_nid]

    def get_edge(self, new_src_nid, new_dst_nid):
        for e in self.adj_list[new_src_nid]:
            if e.new_dst_nid == new_dst_nid: 
                return e
            
    def remove_node(self, n_id):
        if n_id in self.adj_list: del self.adj_list[n_id]

        new_adj_list = defaultdict(set)
        for new_src_nid, neighbours in self.adj_list.items():
            new_neighbours = set(e for e in neighbours if e.new_dst_nid != n_id)
            if new_neighbours: new_adj_list[new_src_nid] = new_neighbours
        self.adj_list = new_adj_list

    def get_neighbours(self, n_id):
        return self.adj_list.get(n_id, [])
    
    def __str__(self):
        n_nodes, n_edges = len(self.adj_list), sum(len(v) for v in self.adj_list.values())
        text = f"Number of nodes: {n_nodes}, Number of edges: {n_edges}\n"
        for k, v in self.adj_list.items():
            c_text = f"Node: {k}, Neighbours: "
            for e in v:
                c_text += f"{e.new_dst_nid}, "
            text += c_text[:-2]
            text += "\n"
        return text

def chop_walks_seqtk(old_walks, n2s, graph, rep1, rep2, seqtk_path):
    """
    Generates telomere information, then chops the walks. 
    1. I regenerate the contigs from the walk nodes. I'm not sure why but when regenerating it this way it differs slightly from the assembly fasta, so i'm doing it this way just to be safe.
    2. seqtk is used to detect telomeres, then I manually count the motifs in each region to determine if it is a '+' or '-' motif.
    3. The walks are then chopped. When a telomere is found:
        a. If there is no telomere in the current walk, and the walk is already >= twice the length of the found telomere, the telomere is added to the walk and then chopped.
        b. If there is an opposite telomere in the current walk, the telomere is added to the walk and then chopped.
        c. If there is an identical telomere in the current walk, the walk is chopped and a new walk begins with the found telomere.
    """

    # Create a list of all edges
    edges_full = {}  ## I dont know why this is necessary. but when cut transitives some edges are wrong otherwise. (This comment is from Martin's script)
    for idx, (src, dst) in enumerate(zip(graph.edges()[0], graph.edges()[1])):
        src, dst = src.item(), dst.item()
        edges_full[(src, dst)] = idx

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
    new_walks, telo_ref = [], {}
    for walk_id, walk in enumerate(old_walks):
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
                    if len(curr_walk) > 2*init_walk_len: # if the telomeric region is as long as the walk preceding it, chop it off
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
    print(f"Chopping complete! n Old Walks: {len(old_walks)}, n New Walks: {len(new_walks)}, n +ve telomeric regions: {rep1_count}, n -ve telomeric regions: {rep2_count}")
    return new_walks, telo_ref

def add_ghosts(old_walks, paf_data, r2n, hifi_r2s, ul_r2s, n2s, old_graph, walk_valid_p, gnnome_config, model_path):
    """
    Adds nodes and edges from the PAF and graph.

    1. Stores all nodes in the walks that are available for connection in n2n_start and n2n_end (based on walk_valid_p). 
    This is split into nodes at the start and end of walks bc incoming edges can only connect to nodes at the start of walks, and outgoing edges can only come from nodes at the end of walks.
    2. I add edges between existing walk nodes using information from PAF (although in all experiments no such edges have been found).
    3. I add nodes using information from PAF.
    4. I add nodes using information from the graph (and by proxy the GFA). 
    """
    n_id = 0
    adj_list = AdjList()

    n2n_start, n2n_end = {}, {} # n2n maps old n_id to new n_id, for the start and ends of the walks respectively
    walk_ids = [] # all n_ids that belong to walks
    nodes_in_old_walks = set()
    for walk in old_walks:
        # Only the first and last walk_valid_p% of nodes in a walk can be connected. Also initialises nodes from walks
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

        walk_ids.append(n_id)
        n_id += 1

    print("Recreating new graph...")
    # Create a list of all edges
    edges_full = {}  ## I dont know why this is necessary. but when cut transitives some edges are wrong otherwise. (This comment is from Martin's script)
    for idx, (src, dst) in enumerate(zip(old_graph.edges()[0], old_graph.edges()[1])):
        src, dst = src.item(), dst.item()
        edges_full[(src, dst)] = idx

    new_graph = dgl.DGLGraph()
    new_old_nids = {}
    ngrl = [] # new graph read length
    for i, node in enumerate(nodes_in_old_walks):
        new_old_nids[node] = i
        ngrl.append(old_graph.ndata['read_length'][node])
    new_graph.add_nodes(
        len(nodes_in_old_walks),
        data = {
            'read_length' : torch.tensor(ngrl)
        }
    )

    ngeid, ngneid, ngos, ngol, ngpl = [[],[]], [[],[]], [], [], [] # new graph edge index, new graph new edge index, new graph ol sim, new graph ol len, new graph prefix len
    for wid, walk in enumerate(old_walks):
        for i, n in enumerate(walk[:-1]):
            next_node = walk[i+1]
            ngeid[0].append(new_old_nids[n]); ngeid[1].append(new_old_nids[next_node])
            ngneid[0].append(wid); ngneid[1].append(wid)
            old_eid = edges_full[(n, next_node)]
            ngos.append(old_graph.edata['overlap_similarity'][old_eid])
            ngol.append(old_graph.edata['overlap_length'][old_eid])
            ngpl.append(old_graph.edata['prefix_length'][old_eid])
    new_graph.add_edges(ngeid[0], ngeid[1])
    new_graph.edata['new_edge_index_src'] = torch.tensor(ngneid[0], dtype=torch.int64)
    new_graph.edata['new_edge_index_dst'] = torch.tensor(ngneid[1], dtype=torch.int64)
    new_graph.edata['overlap_similarity'] = torch.tensor(ngos, dtype=torch.float32)
    new_graph.edata['overlap_length'] = torch.tensor(ngol, dtype=torch.float32)
    new_graph.edata['prefix_length'] = torch.tensor(ngpl, dtype=torch.float32)

    print(f"Adding edges between existing nodes...")
    valid_src, valid_dst, prefix_lens, ol_lens, ol_sims, ghost_data = paf_data['ghost_edges']['valid_src'], paf_data['ghost_edges']['valid_dst'], paf_data['ghost_edges']['prefix_len'], paf_data['ghost_edges']['ol_len'], paf_data['ghost_edges']['ol_similarity'], paf_data['ghost_nodes']
    added_edges_count = 0
    ngeid, ngneid, ngos, ngol, ngpl = [[],[]], [[],[]], [], [], []
    for i in range(len(valid_src)):
        src, dst, prefix_len, ol_len, ol_sim = valid_src[i], valid_dst[i], prefix_lens[i], ol_lens[i], ol_sims[i]
        if src in n2n_end and dst in n2n_start:
            if n2n_end[src] == n2n_start[dst]: continue # ignore self-edges
            added_edges_count += 1
            adj_list.add_edge(Edge(
                new_src_nid=n2n_end[src], 
                new_dst_nid=n2n_start[dst], 
                old_src_nid=src, 
                old_dst_nid=dst, 
                prefix_len=prefix_len, 
                ol_len=ol_len, 
                ol_sim=ol_sim
            ))
            ngeid[0].append(new_old_nids[src]); ngeid[1].append(new_old_nids[dst])
            ngneid[0].append(n2n_end[src]); ngneid[1].append(n2n_start[dst])
            ngos.append(ol_sim); ngol.append(ol_len); ngpl.append(prefix_len)
    print("Added edges:", added_edges_count)

    print(f"Adding ghost nodes...")
    ngrl = []
    dgl_nid = new_graph.num_nodes()
    n2s_ghost = {}
    ghost_data = ghost_data['hop_1'] # WE ONLY DO FOR 1-HOP FOR NOW
    added_nodes_count = 0
    for orient in ['+', '-']:
        for read_id, data in ghost_data[orient].items():
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
                ngeid[0].append(dgl_nid); ngeid[1].append(new_old_nids[n[0]])
                ngneid[0].append(n_id); ngneid[1].append(n2n_start[n[0]]) 
                ngos.append(n[3]); ngol.append(n[2]); ngpl.append(n[1])
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
                ngeid[0].append(new_old_nids[n[0]]); ngeid[1].append(dgl_nid)
                ngneid[0].append(n2n_end[n[0]]); ngneid[1].append(n_id)
                ngos.append(n[3]); ngol.append(n[2]); ngpl.append(n[1])

            if orient == '+':
                seq, _ = get_seqs(read_id, hifi_r2s, ul_r2s)
            else:
                _, seq = get_seqs(read_id, hifi_r2s, ul_r2s)
            n2s_ghost[n_id] = seq
            n_id += 1
            ngrl.append(data['read_len'])
            dgl_nid += 1
            added_nodes_count += 1
    print("Number of nodes added from PAF:", added_nodes_count)

    print(f"Adding nodes from old graph...")
    edges, edge_features = old_graph.edges(), old_graph.edata
    graph_data = defaultdict(lambda: defaultdict(list))
    for i in range(edges[0].shape[0]):
        src_node = edges[0][i].item()  
        dst_node = edges[1][i].item()  
        ol_len = edge_features['overlap_length'][i].item()  
        ol_sim = edge_features['overlap_similarity'][i].item()
        prefix_len = edge_features['prefix_length'][i].item() 

        if src_node not in nodes_in_old_walks:
            graph_data[src_node]['read_len'] = old_graph.ndata['read_length'][src_node]
            graph_data[src_node]['outs'].append(dst_node)
            graph_data[src_node]['ol_len_outs'].append(ol_len)
            graph_data[src_node]['ol_sim_outs'].append(ol_sim)
            graph_data[src_node]['prefix_len_outs'].append(prefix_len)

        if dst_node not in nodes_in_old_walks:
            graph_data[dst_node]['read_len'] = old_graph.ndata['read_length'][dst_node]
            graph_data[dst_node]['ins'].append(src_node)
            graph_data[dst_node]['ol_len_ins'].append(ol_len)
            graph_data[dst_node]['ol_sim_ins'].append(ol_sim)
            graph_data[dst_node]['prefix_len_ins'].append(prefix_len)

    # add to adj list where applicable
    for old_node_id, data in graph_data.items():
        curr_out_neighbours, curr_in_neighbours = set(), set()

        for i, out_n_id in enumerate(data['outs']):
            if out_n_id not in n2n_start: continue
            curr_out_neighbours.add((out_n_id, data['prefix_len_outs'][i], data['ol_len_outs'][i], data['ol_sim_outs'][i]))

        for i, in_n_id in enumerate(data['ins']):
            if in_n_id not in n2n_end: continue
            curr_in_neighbours.add((in_n_id, data['prefix_len_ins'][i], data['ol_len_ins'][i], data['ol_sim_ins'][i]))

        if not curr_out_neighbours or not curr_in_neighbours: continue

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
            ngeid[0].append(dgl_nid); ngeid[1].append(new_old_nids[n[0]])
            ngneid[0].append(n_id); ngneid[1].append(n2n_start[n[0]]) 
            ngos.append(n[3]); ngol.append(n[2]); ngpl.append(n[1])
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
            ngeid[0].append(new_old_nids[n[0]]); ngeid[1].append(dgl_nid)
            ngneid[0].append(n2n_end[n[0]]); ngneid[1].append(n_id)
            ngos.append(n[3]); ngol.append(n[2]); ngpl.append(n[1])

        seq = n2s[old_node_id]
        n2s_ghost[n_id] = seq
        n_id += 1
        ngrl.append(data['read_len'])
        dgl_nid += 1
        added_nodes_count += 1

    new_graph.add_nodes(
        added_nodes_count,
        data = {
            'read_length' : torch.tensor(ngrl)
        }
    )
    new_graph.add_edges(ngeid[0], ngeid[1], data={
        'new_edge_index_src' : torch.tensor(ngneid[0], dtype=torch.int64),
        'new_edge_index_dst' : torch.tensor(ngneid[1], dtype=torch.int64),
        'overlap_similarity' : torch.tensor(ngos, dtype=torch.float32),
        'overlap_length' : torch.tensor(ngol, dtype=torch.float32),
        'prefix_length' : torch.tensor(ngpl, dtype=torch.float32)
    })

    print("Getting ghost edge scores...")
    new_graph, x, e = preprocess_graph(new_graph)
    train_config = gnnome_config['training']
    with torch.no_grad():
        model = SymGatedGCNModel(
            train_config['node_features'],
            train_config['edge_features'],
            train_config['hidden_features'],
            train_config['hidden_edge_features'],
            train_config['num_gnn_layers'],
            train_config['hidden_edge_scores'],
            train_config['batch_norm'],
            dropout=train_config['dropout']
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        edge_predictions, stop_predictions = model(new_graph, x, e)
        new_graph.edata['score'] = edge_predictions.squeeze()
    
    e2s = {}
    for eid in range(new_graph.num_edges()):
        new_src_id, new_dst_id = new_graph.edata['new_edge_index_src'][eid], new_graph.edata['new_edge_index_dst'][eid]
        if new_src_id == new_dst_id: continue # this is not a ghost edge
        e2s[(new_src_id.item(), new_dst_id.item())] = new_graph.edata['score'][eid].item()
        e2s[(new_dst_id.item(), new_src_id.item())] = new_graph.edata['score'][eid].item() # add the reverse as well cuz get_best_walk also searches in reverse

    global SCORE_CUTOFF
    SCORE_CUTOFF = np.percentile(list(e2s.values()), 5)
    print("score cutoff: ", SCORE_CUTOFF)

    print("Final number of nodes:", n_id)
    if added_edges_count or added_nodes_count:
        return adj_list, walk_ids, n2s_ghost, e2s
    else:
        return None, None, None, None

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
                elif new_dst_nid < n_old_walks:
                    walk = old_walks[new_dst_nid]
                    for i in walk:
                        if i == neigh.old_dst_nid: # new one is better, update dupchecker and remove old one from adj list
                            dup_checker[new_dst_nid] = neigh
                            break
                else:
                    raise ValueError("Duplicate edge between two non-walks found!")
                
        adj_list.adj_list[new_src_nid] = set(n for n in dup_checker.values())
    
    print("Final number of edges:", sum(len(x) for x in adj_list.adj_list.values()))
    return adj_list




def get_best_walk(adj_list, start_node, n_old_walks, telo_ref, e2s, penalty=None, memo_chances=50, visited_init=set()):
    visited = visited_init

    walk, n_key_nodes = [start_node], 1
    visited.add(start_node)
    c_node = start_node
    while True:
        c_neighs = adj_list.get_neighbours(c_node)
        c_neighs = [n for n in c_neighs if (n.new_dst_nid not in visited and e2s[c_node, n.new_dst_nid]>SCORE_CUTOFF)]
        if not c_neighs: break

        highest_score = max(c_neighs, key=lambda n: e2s[c_node, n.new_dst_nid])
        walk.append(highest_score.new_dst_nid)
        if highest_score.new_dst_nid < n_old_walks: n_key_nodes += 1
        c_node = highest_score.new_dst_nid
        visited.add(c_node)

    if walk[-1] >= n_old_walks: walk.pop()

    return walk, n_key_nodes, 0




# def get_best_walk(adj_list, start_node, n_old_walks, telo_ref, e2s, penalty=None, memo_chances=50, visited_init=set()):
#     """
#     Given a start node, run the greedy DFS to retrieve the walk with the most key nodes.

#     1. When searching, the number of key nodes in the walk, telomere information, and penalty is tracked.
#         a. Number of key nodes are used to compare and select walks.
#         b. Telomere information is used to terminate walks. If a telomere key node is found, it checks the compatability with the telomere in the current walk (if any). For a telomere to be compatible,
#             i. The motif must be opposite.
#             ii. The position of the telomere in the key node's sequence must be opposite. i.e. If the current walk begins with a key node with a telomere at the start of its sequence, then it will only accept key nodes with telomeres at the end of its sequence, and vice versa.
#             iii. The penalty (either overlap similarity or overlap length, configurable) is also tracked to break ties on number of key nodes. However, we found this to not be of much use.
#     2. Optimal walks from each node are memoised after being visited memo_chances number of times. This is because exhaustively searching is computationally infeasible.
#     """

#     # Dictionary to memoize the longest walk from each node
#     memo, memo_counts = {}, defaultdict(int)

#     def get_telo_info(node):
#         if node >= n_old_walks: return None

#         if telo_ref[node]['start']:
#             return ('start', telo_ref[node]['start'])
#         elif telo_ref[node]['end']:
#             return ('end', telo_ref[node]['end'])
#         else:
#             return None

#     def check_telo_compatibility(t1, t2):
#         if t1 is None or t2 is None:
#             return True
#         elif t1[0] != t2[0] and t1[1] != t2[1]: # The position must be different, and motif var must be different.
#             return True
#         else:
#             return False

#     def dedup(curr_node, curr_walk, curr_key_nodes, curr_penalty):
#         curr_walk_set = set(curr_walk)
#         if curr_node not in curr_walk_set:
#             return curr_walk, curr_key_nodes, curr_penalty
        
#         c_walk, c_key_nodes, c_penalty = [], 0, 0
#         for i, n in enumerate(curr_walk):
#             if n == curr_node: break
#             c_walk.append(n)
#             if n < n_old_walks: c_key_nodes += 1
#             if i != len(curr_walk)-1:
#                 c_penalty -= e2s[(n, curr_walk[i+1])]

#         return c_walk, c_key_nodes, c_penalty

#     def dfs(node, visited, walk_telo):
#         if node < n_old_walks:
#             telo_info = get_telo_info(node)
#             if telo_info is not None:
#                 if walk_telo: print("WARNING: Trying to set walk_telo when it is already set!") 
#                 walk_telo = telo_info

#         # If the longest walk starting from this node is already memoised and telomere is compatible, return it
#         if node in memo: 
#             memo_telo = memo[node][3]
#             if check_telo_compatibility(walk_telo, memo_telo):
#                 return memo[node][0], memo[node][1], memo[node][2]

#         visited.add(node)
#         max_walk, max_key_nodes, min_penalty = [node], 0, 0

#         # Traverse all the neighbors of the current node
#         for neighbor in adj_list.get_neighbours(node):
#             # Check visited
#             dst = neighbor.new_dst_nid
#             if dst in visited: continue
#             # Check telomere compatibility
#             terminate = False
#             if dst < n_old_walks:
#                 curr_telo = get_telo_info(dst)
#                 if curr_telo is not None:
#                     if check_telo_compatibility(walk_telo, curr_telo):
#                         terminate = True
#                     else:
#                         continue 

#             if terminate:
#                 # Terminate search at the next node due to telomere compatibility
#                 current_walk, current_key_nodes, current_penalty = [dst], 1, 0
#             else:
#                 # Perform DFS on the neighbor and check the longest walk from that neighbor
#                 current_walk, current_key_nodes, current_penalty = dfs(dst, visited, walk_telo)
#                 # We have to check if there are duplicates in the returned walk. This is because of memoisation, where a returned memoised result can bypass visited set check.
#                 current_walk, current_key_nodes, current_penalty = dedup(node, current_walk, current_key_nodes, current_penalty)

#             # Add the penalty for selecting that neighbour based on model scores
#             current_penalty -= e2s[(node, dst)]

#             if current_walk[-1] >= n_old_walks: # last node is a ghost node, should not count their penalty
#                 if len(current_walk) > 1:
#                     current_penalty += e2s[(current_walk[-2], current_walk[-1])]

#             # If adding this walk leads to a longer path, or same one with same length but lower penalty, update the max_walk and min_penalty
#             if (current_key_nodes > max_key_nodes) or (current_key_nodes == max_key_nodes and current_penalty < min_penalty):
#                 max_walk = [node] + current_walk
#                 max_key_nodes = current_key_nodes
#                 min_penalty = current_penalty

#         visited.remove(node)
#         if node < n_old_walks: max_key_nodes += 1

#         # Memoize the result for this node if chances are used up
#         memo_counts[node] += 1
#         if memo_counts[node] >= memo_chances:
#             if len(max_walk) == 1 and max_walk[-1] >= n_old_walks:
#                 curr_telo = None
#             elif max_walk[-1] < n_old_walks:
#                 curr_telo = get_telo_info(max_walk[-1])
#             else:
#                 curr_telo = get_telo_info(max_walk[-2])
#             memo[node] = (max_walk, max_key_nodes, min_penalty, curr_telo)

#         return max_walk, max_key_nodes, min_penalty    

#     # Start DFS from the given start node
#     res_walk, res_key_nodes, res_penalty = dfs(start_node, visited_init, None)

#     # If the last node in a walk is a ghost node, remove it from the walk and negate its penalty.
#     # This case should not occur, but I am just double checking
#     if res_walk[-1] >= n_old_walks:
#         res_penalty += e2s[(res_walk[-2], res_walk[-1])]
#         res_walk.pop()

#     return res_walk, res_key_nodes, res_penalty

def get_walks(walk_ids, adj_list, telo_ref, e2s, dfs_penalty):
    """
    Creates the new walks without prioritising nodes with telomeres.

    1. Key nodes with start and end telomeres in its sequence are removed beforehand.
    2. For all key nodes, we find the best walk starting from that node. The best walk out of all is then saved, and the process is repeated until all key nodes are used.
        i. We each node we search forwards and backwards, then append the results together.
    """

    # Generating new walks using greedy DFS
    new_walks = []
    temp_walk_ids, temp_adj_list = deepcopy(walk_ids), deepcopy(adj_list)
    n_old_walks = len(temp_walk_ids)
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
            temp_walk_ids.remove(walk_id)

    # Loop until all walks are connected
    while temp_walk_ids:
        best_walk, best_key_nodes, best_penalty = [], 0, 0
        for walk_id in temp_walk_ids: # the node_id is also the index
            curr_walk, curr_key_nodes, curr_penalty = get_best_walk(temp_adj_list, walk_id, n_old_walks, telo_ref, e2s, dfs_penalty)
            visited_init = set(curr_walk[1:]) if len(curr_walk) > 1 else set()
            curr_walk_rev, curr_key_nodes_rev, curr_penalty_rev = get_best_walk(rev_adj_list, walk_id, n_old_walks, telo_ref, e2s, dfs_penalty, visited_init=visited_init)
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
            if w < n_old_walks: temp_walk_ids.remove(w)

        new_walks.append(best_walk)

    print(f"New walks generated! n new walks: {len(new_walks)}")
    return new_walks

def get_walks_telomere(walk_ids, adj_list, telo_ref, e2s, dfs_penalty):
    """
    Creates the new walks, priotising key nodes with telomeres.

    1. Key nodes with start and end telomeres in its sequence are removed beforehand.
    2. We separate out all key nodes that have telomeres. For each of these key nodes, we find the best walk starting from that node. The best walk out of all is then saved, and the process is repeated until all key nodes are used.
        i. Depending on whether the telomere is in the start or end of the sequence, we search forwards or in reverse. We create a reversed version of the adj_list for this.
    3. We then repeat the above step for all key nodes without telomere information that are still unused.
    """

    # Generating new walks using greedy DFS
    new_walks = []
    temp_walk_ids, temp_adj_list = deepcopy(walk_ids), deepcopy(adj_list)
    n_old_walks = len(temp_walk_ids)
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
            temp_walk_ids.remove(walk_id)

    # Split walks into those with telomeric regions and those without
    telo_walk_ids, non_telo_walk_ids = [], []
    for i in temp_walk_ids:
        if telo_ref[i]['start'] or telo_ref[i]['end']:
            telo_walk_ids.append(i)
        else:
            non_telo_walk_ids.append(i)

    # Generate walks for walks with telomeric regions first
    while telo_walk_ids:
        best_walk, best_key_nodes, best_penalty = [], 0, 0
        for walk_id in telo_walk_ids: # the node_id is also the index
            if telo_ref[walk_id]['start']:
                curr_walk, curr_key_nodes, curr_penalty = get_best_walk(temp_adj_list, walk_id, n_old_walks, telo_ref, e2s, dfs_penalty)
            else:
                curr_walk, curr_key_nodes, curr_penalty = get_best_walk(rev_adj_list, walk_id, n_old_walks, telo_ref, e2s, dfs_penalty)
                curr_walk.reverse()
            if curr_key_nodes > best_key_nodes or (curr_key_nodes == best_key_nodes and curr_penalty < best_penalty):
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
        best_walk, best_key_nodes, best_penalty = [], 0, 0
        for walk_id in non_telo_walk_ids: # the node_id is also the index
            curr_walk, curr_key_nodes, curr_penalty = get_best_walk(temp_adj_list, walk_id, n_old_walks, telo_ref, e2s, dfs_penalty)
            visited_init = set(curr_walk[1:]) if len(curr_walk) > 1 else set()
            curr_walk_rev, curr_key_nodes_rev, curr_penalty_rev = get_best_walk(rev_adj_list, walk_id, n_old_walks, telo_ref, e2s, dfs_penalty, visited_init=visited_init)
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

def get_contigs(old_walks, new_walks, adj_list, n2s, n2s_ghost, g):
    """
    Recreates the contigs given the new walks. 
    
    1. Pre-processes the new walks to break down key nodes into the original nodes based on their connections.
    2. Converts the nodes into the contigs. This is done in the same way as in the GNNome pipeline.
    """

    n_old_walks = len(old_walks)

    # print("Preprocessing walks...")
    # Create a list of all edges
    edges_full = {}  ## I dont know why this is necessary. but when cut transitives some edges are wrong otherwise. (This is from Martin's script)
    for idx, (src, dst) in enumerate(zip(g.edges()[0], g.edges()[1])):
        src, dst = src.item(), dst.item()
        edges_full[(src, dst)] = idx

    walk_nodes, walk_seqs, walk_prefix_lens = [], [], []
    for i, walk in enumerate(new_walks):
        c_nodes, c_seqs, c_prefix_lens = [], [], []
        for j, node in enumerate(walk):
            if node >= n_old_walks: # Node is a new ghost node
                c_nodes.append(node)
                c_seqs.append(str(n2s_ghost[node]))
                curr_edge = adj_list.get_edge(node, walk[j+1])
                c_prefix_lens.append(curr_edge.prefix_len)
            else: # Node is an original walk
                old_walk = old_walks[node]
                if j == 0:
                    start = 0
                else:
                    curr_edge = adj_list.get_edge(walk[j-1], node)
                    start = old_walk.index(curr_edge.old_dst_nid)
                
                if j+1 == len(walk):
                    end = len(old_walk)-1
                    prefix_len = None
                else:
                    curr_edge = adj_list.get_edge(node, walk[j+1])
                    end = old_walk.index(curr_edge.old_src_nid)
                    prefix_len = curr_edge.prefix_len

                for k in range(start, end+1):
                    c_nodes.append(old_walk[k])
                    c_seqs.append(str(n2s[old_walk[k]]))

                    if k != end:
                        c_prefix_lens.append(g.edata['prefix_length'][edges_full[(old_walk[k], old_walk[k+1])]])

                if prefix_len: c_prefix_lens.append(prefix_len)

        walk_nodes.append(c_nodes)
        walk_seqs.append(c_seqs)
        walk_prefix_lens.append(c_prefix_lens)

    # print(f"Generating sequences...")
    contigs = []
    for i, seqs in enumerate(walk_seqs):
        prefix_lens = walk_prefix_lens[i]
        c_contig = []
        for j, seq in enumerate(seqs[:-1]):
            c_contig.append(seq[:prefix_lens[j]])
        c_contig.append(seqs[-1])

        c_contig = Seq.Seq(''.join(c_contig))
        c_contig = SeqIO.SeqRecord(c_contig)
        c_contig.id = f'contig_{i+1}'
        c_contig.description = f'length={len(c_contig)}'
        contigs.append(c_contig)

    return contigs

def postprocess(name, hyperparams, paths, aux, gnnome_config):
    """
    (\(\        \|/        /)/)
    (  ^.^)     -o-     (^.^  )
    o_(")(")    /|\    (")(")_o

    Performs scaffolding on GNNome's walks using information from PAF, GFA, and telomeres.
    Currently, only uses info from 1-hop neighbourhood of original graph. Any two walks are at most connected by a single ghost node. Also, all added ghost nodes must have at least one incoming and one outgoing edge to a walk.
    
    Summary of the pipeline (details can be found in the respective functions):
    1. Loads the relevant files.
    2. Generates telomere information, then chops walks accordingly.
    3. Compresses each GNNome walk into a single node, then adds 'ghost' nodes and edges using information from PAF and GFA.
    4. Decodes the new sequences using DFS and telomere information.
    5. Regenerates contigs and calculates metrics.
    """
    time_start = datetime.now()

    print(f"\n===== Postprocessing {name} =====")
    hyperparams_str = ""
    for k, v in hyperparams.items():
        hyperparams_str += f"{k}: {v}, "
    print(hyperparams_str[:-2]+"\n")
    walks, n2s, r2n, paf_data, old_graph, hifi_r2s, ul_r2s = aux['walks'], aux['n2s'], aux['r2n'], aux['paf_data'], aux['old_graph'], aux['hifi_r2s'], aux['ul_r2s']

    print(f"Chopping old walks... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    if hyperparams['use_telomere_info']:
        rep1, rep2 = hyperparams['telo_motif'][0], hyperparams['telo_motif'][1]
        walks, telo_ref = chop_walks_seqtk(walks, n2s, old_graph, rep1, rep2, paths['seqtk'])
    else:
        telo_ref = { i:{'start':None, 'end':None} for i in range(len(walks)) }

    print(f"Adding ghost nodes and edges... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    adj_list, walk_ids, n2s_ghost, e2s = add_ghosts(
        old_walks=walks,
        paf_data=paf_data,
        r2n=r2n,
        hifi_r2s=hifi_r2s,
        ul_r2s=ul_r2s,
        n2s=n2s,
        old_graph=old_graph,
        walk_valid_p=hyperparams['walk_valid_p'],
        gnnome_config=gnnome_config,
        model_path=paths['model']
    )
    if adj_list is None and walk_ids is None and n2s_ghost is None and e2s is None:
        print("No suitable nodes and edges found to add to these walks. Returning...")
        return

    # Remove duplicate edges between nodes. If there are multiple connections between a walk and another node/walk, we choose the best one.
    # This could probably have been done while adding the edges in. However, to avoid confusion, i'm doing this separately.
    print(f"De-duplicating edges... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    adj_list = deduplicate(adj_list, walks)

    print(f"Generating new walks... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    if hyperparams['walk_var'] == 'default':
        new_walks = get_walks(walk_ids, adj_list, telo_ref, e2s, hyperparams['dfs_penalty'])
    elif hyperparams['walk_var'] == 'telomere':
        new_walks = get_walks_telomere(walk_ids, adj_list, telo_ref, e2s, hyperparams['dfs_penalty'])
    else:
        raise ValueError("Invalid walk_var!")

    print(f"Generating contigs... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    contigs = get_contigs(walks, new_walks, adj_list, n2s, n2s_ghost, old_graph)

    print(f"Calculating assembly metrics... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    asm_metrics(contigs, paths['save'], paths['ref'], paths['minigraph'], paths['paftools'])
    if paths['yak1'] and paths['yak2']: yak_metrics(paths['save'], paths['yak1'], paths['yak2'], paths['yak'])

    print(f"Run finished! (Time: {timedelta_to_str(datetime.now() - time_start)})")
    return

def run_postprocessing(config):
    postprocessing_config = config['postprocessing']
    gnnome_config = config['gnnome']
    genomes = config['run']['postprocessing']['genomes']
    for genome in genomes:
        postprocessing_config['telo_motif'] = config['genome_info'][genome]['telo_motifs']
        paths = config['genome_info'][genome]['paths']
        paths.update(config['misc']['paths'])

        print("Loading files...")
        aux = {}
        with open(paths['walks'], 'rb') as f:
            aux['walks'] = pickle.load(f)
        with open(paths['n2s'], 'rb') as f:
            aux['n2s'] = pickle.load(f)
        with open(paths['r2n'], 'rb') as f:
            aux['r2n'] = pickle.load(f)
        with open(paths['paf_processed'], 'rb') as f:
            aux['paf_data'] = pickle.load(f)
        aux['old_graph'] = dgl.load_graphs(paths['graph']+f'{genome}.dgl')[0][0]
        aux['hifi_r2s'] = Fasta(paths['ec_reads'])
        aux['ul_r2s'] = Fasta(paths['ul_reads']) if paths['ul_reads'] else None

        postprocess(genome, hyperparams=postprocessing_config, paths=paths, aux=aux, gnnome_config=gnnome_config)
        # for w in [0.005, 0.0025, 0.001]:
        #     postprocessing_config['walk_valid_p'] = w
        #     postprocess(genome, hyperparams=postprocessing_config, paths=paths, aux=aux, gnnome_config=gnnome_config)