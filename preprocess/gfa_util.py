from Bio.Seq import Seq
from collections import defaultdict
from torch_geometric.data import Data
from tqdm import tqdm
import edlib, re, torch

def calculate_similarities(edge_ids, n2s, overlap_lengths):
    overlap_similarities = {}
    for src, dst in tqdm(edge_ids.keys(), ncols=120):
        ol_length = overlap_lengths[(src, dst)]
        read_src = n2s[src]
        read_dst = n2s[dst]
        edit_distance = edlib.align(read_src[-ol_length:], read_dst[:ol_length])['editDistance']
        overlap_similarities[(src, dst)] = 1 - edit_distance / ol_length
    return overlap_similarities

def graph_to_successor_dict(g):
    # Ensure the edge_index is in COO format and directed
    edge_index = g.edge_index
    successors_dict = defaultdict(list)

    # edge_index[0] contains source nodes, edge_index[1] contains target nodes
    for src, tgt in zip(edge_index[0], edge_index[1]):
        successors_dict[src.item()].append(tgt.item())

    return successors_dict

def parse_raw_gfa(gfa_path):
    print("Reading GFA...")
    with open(gfa_path) as f:
        rows = f.readlines()

    n_rows = len(rows)    
    n_id, e_id = 0, 0

    r2n, r2n2, n2r = {}, {}, {}
    read_lengths, n2s = {}, {}
    edge_ids, prefix_lengths, overlap_lengths, overlap_similarities = {}, {}, {}, {}
    edge_index = [[],[]]

    r_ind = 0
    while r_ind < n_rows:
        row = rows[r_ind].strip().split()
        tag = row.pop(0)

        if tag == 'S':
            if len(row) == 4: # Hifiasm
                s_id, seq, length, count = row
            else:
                raise Exception("Unknown GFA format!")

            real_id, virt_id = n_id, n_id + 1 # 1+, 1-

            n_id += 2
            r2n[s_id] = (real_id, virt_id)
            n2r[real_id] = s_id; n2r[virt_id] = s_id

            seq = Seq(seq)
            n2s[real_id] = str(seq); n2s[virt_id] = str(seq.reverse_complement())

            length = int(length[5:])
            read_lengths[real_id] = length; read_lengths[virt_id] = length

            if s_id.startswith('utg'):
                # The issue here is that in some cases, one unitig can consist of more than one read
                # So this is the adapted version of the code that supports that
                # The only things of importance here are r2n2 dict (not overly used)
                # And id variable which I use for obtaining positions during training (for the labels)
                # I don't use it for anything else, which is good
                s_ids = []
                while rows[r_ind+1][0] == 'A':
                    r_ind += 1
                    row = rows[r_ind].strip().split()
                    read_orientation, utg_to_read = row[3], row[4]
                    s_ids.append((utg_to_read, read_orientation))
                    r2n2[utg_to_read] = (real_id, virt_id)

                s_id = s_ids
                n2r[real_id] = s_id
                n2r[virt_id] = s_id
        elif tag == 'L':
            if len(row) == 6: # Hifiasm GFA
                s_id1, orient1, s_id2, orient2, cigar, _ = row
                s_id1 = re.findall(r'(.*):\d-\d*', s_id1)[0]
                s_id2 = re.findall(r'(.*):\d-\d*', s_id2)[0]
            elif len(row) == 7: # Hifiasm GFA newer
                s_id1, orient1, s_id2, orient2, cigar, _, _ = row
            else:
                raise Exception("Unknown GFA format!")
            
            if orient1 == '+' and orient2 == '+':
                src_real = r2n[s_id1][0]
                dst_real = r2n[s_id2][0]
                src_virt = r2n[s_id2][1]
                dst_virt = r2n[s_id1][1]
            elif orient1 == '+' and orient2 == '-':
                src_real = r2n[s_id1][0]
                dst_real = r2n[s_id2][1]
                src_virt = r2n[s_id2][0]
                dst_virt = r2n[s_id1][1]
            elif orient1 == '-' and orient2 == '+':
                src_real = r2n[s_id1][1]
                dst_real = r2n[s_id2][0]
                src_virt = r2n[s_id2][1]
                dst_virt = r2n[s_id1][0]
            elif orient1 == '-' and orient2 == '-':
                src_real = r2n[s_id1][1]
                dst_real = r2n[s_id2][1]
                src_virt = r2n[s_id2][0]
                dst_virt = r2n[s_id1][0]  
            else:
                raise Exception("Unknown GFA format!")
            
            # Don't need to manually add reverse complement edge, cuz Hifiasm already creates a separate entry for it 
            edge_index[0].append(src_real); edge_index[1].append(dst_real)
            edge_ids[(src_real, dst_real)] = e_id
            e_id += 1

            # -----------------------------------------------------------------------------------
            # This enforces similarity between the edge and its "virtual pair"
            # Meaning if there is A -> B and B^rc -> A^rc they will have the same overlap_length
            # When parsing CSV that was not necessarily so:
            # Sometimes reads would be slightly differently aligned from their RC pairs
            # Thus resulting in different overlap lengths
            # -----------------------------------------------------------------------------------

            try:
                ol_length = int(cigar[:-1])  # Assumption: this is overlap length and not a CIGAR string
            except ValueError:
                print('Cannot convert CIGAR string into overlap length!')
                raise ValueError
            
            overlap_lengths[(src_real, dst_real)] = ol_length
            overlap_lengths[(src_virt, dst_virt)] = ol_length

            prefix_lengths[(src_real, dst_real)] = read_lengths[src_real] - ol_length
            prefix_lengths[(src_virt, dst_virt)] = read_lengths[src_virt] - ol_length

        r_ind += 1

    # Why is this the case? Is it because if there is even a single 'A' file in the .gfa, means the format is all 'S' to 'A' lines?
    if len(r2n2) != 0: r2n = r2n2

    print("Calculating similarities...")
    overlap_similarities = calculate_similarities(edge_ids, n2s, overlap_lengths)

    return edge_ids, edge_index, n2s, n2r, r2n, read_lengths, prefix_lengths, overlap_lengths, overlap_similarities

def parse_final_gfa(gfa_path):
    pass

def preprocess_gfa(gfa_path, source):
    if source == 'gnnome':
        edge_ref, edge_index, n2s, n2r, r2n, read_lens, prefix_lens, ol_lens, ol_sims = parse_raw_gfa(gfa_path)
    elif source == 'hifiasm':
        edge_ref, edge_index, n2s, n2r, r2n, read_lens, prefix_lens, ol_lens, ol_sims = parse_final_gfa(gfa_path)
    else:
        raise ValueError("Invalid source!")
    
    n_nodes, n_edges = len(n2s), len(edge_ref)
    g = Data(N_ID=torch.tensor([i for i in range(n_nodes)]), E_ID=torch.tensor([i for i in range(n_edges)]), edge_index=torch.tensor(edge_index))
    
    node_attrs, edge_attrs = ['N_ID', 'read_length'], ['E_ID', 'prefix_length', 'overlap_length', 'overlap_similarity']
    # Only convert to list right before creating graph data
    read_lens_list = [read_lens[i] for i in range(n_nodes)]
    prefix_lens_list, ol_lens_list, ol_sims_list = [0]*n_edges, [0]*n_edges, [0]*n_edges
    for k, eid in edge_ref.items():
        prefix_lens_list[eid] = prefix_lens[k]
        ol_lens_list[eid] = ol_lens[k]
        ol_sims_list[eid] = ol_sims[k]
    g['read_length'] = torch.tensor(read_lens_list)
    g['prefix_length'] = torch.tensor(prefix_lens_list)
    g['overlap_length'] = torch.tensor(ol_lens_list)
    g['overlap_similarity'] = torch.tensor(ol_sims_list)

    aux = {
        'r2n' : r2n,
        'n2s' : n2s,
        'n2r' : n2r,
        'node_attrs' : node_attrs,
        'edge_attrs' : edge_attrs,
        'successor_dict' : graph_to_successor_dict(g)
    }

    return g, aux