from collections import defaultdict
from copy import deepcopy
import edlib
from multiprocessing import Pool, Lock
from tqdm import tqdm

from misc.utils import get_seqs

HIFI_R2S, UL_R2S, R2N, SUCCESSOR_DICT, N2R, READS_PARSED = None, None, None, None, None, set()

# For pyfaidx
def init_lock(l):
    global lock
    lock = l

# We have to do this because cannot pickle defaultdicts created by lambda
def create_list_dd():
    return defaultdict(list)

def preprocess_rows(rows): 
    res, dupchecker, utgchecker, ghost_utg_checker = [], set(), set(), { '+' : defaultdict(set), '-' : defaultdict(set) }
    duplicates, rejected = 0, 0
    print("Preprocessing paf...")
    for row in tqdm(rows, ncols=120):
        row_split = row.strip().split()
        id1, len1, start1, end1, orientation, id2, len2, start2, end2, _, _, _ = row_split

        if (id1, id2) in dupchecker or (id2, id1) in dupchecker:
            duplicates += 1
            continue
        else:
            len1, start1, end1, len2, start2, end2 = int(len1), int(start1), int(end1), int(len2), int(start2), int(end2)
            src, dst = None, None
            if orientation == '+':
                if start1 == 0 and start2 == 0:
                    rejected += 1
                    continue
                elif end1 == len1 and end2 == len2:
                    rejected += 1
                    continue
                elif end1 == len1 and start2 == 0:
                    src, dst = (id1, '+'), (id2, '+')
                    src_rev, dst_rev = (id2, '-'), (id1, '-')
                elif start1 == 0 and end2 == len2:
                    src, dst = (id2, '+'), (id1, '+')
                    src_rev, dst_rev = (id1, '-'), (id2, '-')
                else:
                    rejected += 1
                    continue
            else:
                if start1 == 0 and end2 == len2:
                    rejected += 1
                    continue
                elif end1 == len1 and start2 == 0:
                    rejected += 1
                    continue
                elif end1 == len1 and end2 == len2:
                    src, dst = (id1, '+'), (id2, '-')
                    src_rev, dst_rev = (id2, '+'), (id1, '-')
                elif start1 == 0 and start2 == 0:
                    src, dst = (id1, '-'), (id2, '+')
                    src_rev, dst_rev = (id2, '-'), (id1, '+')
                else:
                    rejected += 1
                    continue

            src_id, dst_id = src[0], dst[0]

            # handling edge cases from unitigs
            if src_id in R2N and dst_id in R2N:
                nids1, nids2 = R2N[src_id], R2N[dst_id]

                # Overlaps between reads in the same unitig 
                if nids1 == nids2:
                    duplicates += 1
                    continue

                # Overlaps where one or both nodes are unitigs
                if src[1] == '+' and dst[1] == '+':
                    src_n_id, dst_n_id = R2N[src_id][0], R2N[dst_id][0]
                    src_rev_n_id, dst_rev_n_id = R2N[src_rev[0]][1], R2N[dst_rev[0]][1]
                elif src[1] == '+' and dst[1] == '-':
                    src_n_id, dst_n_id = R2N[src_id][0], R2N[dst_id][1]
                    src_rev_n_id, dst_rev_n_id = R2N[src_rev[0]][0], R2N[dst_rev[0]][1]
                elif src[1] == '-' and dst[1] == '+':
                    src_n_id, dst_n_id = R2N[src_id][1], R2N[dst_id][0]
                    src_rev_n_id, dst_rev_n_id = R2N[src_rev[0]][1], R2N[dst_rev[0]][0]

                # Overlaps where edge is already in gfa
                if dst_n_id in SUCCESSOR_DICT[src_n_id] or dst_rev_n_id in SUCCESSOR_DICT[src_rev_n_id]:
                    duplicates += 1
                    continue

                src_reads, dst_reads = N2R[src_n_id], N2R[dst_n_id]
                if isinstance(src_reads, list) and len(src_reads) > 1:
                    if src_id != src_reads[0][0] and src_id != src_reads[-1][0]:
                        rejected += 1
                        continue

                    if src_id == id1:
                        c_start, c_end, c_len = start1, end1, len1
                    else:
                        c_start, c_end, c_len = start2, end2, len2

                    if src_id == src_reads[0][0]:
                        if src_reads[0][1] == '+':
                            if c_start != 0:
                                rejected += 1
                                continue
                        else:
                            if c_end != c_len:
                                rejected += 1
                                continue
                    else:
                        if src_reads[-1][1] == '+':
                            if c_end != c_len:
                                rejected += 1
                                continue
                        else:
                            if c_start != 0:
                                rejected += 1
                                continue

                if isinstance(dst_reads, list) and len(dst_reads) > 1:
                    if dst_id != dst_reads[0][0] and dst_id != dst_reads[-1][0]:
                        rejected += 1
                        continue

                    if dst_id == id1:
                        c_start, c_end, c_len = start1, end1, len1
                    else:
                        c_start, c_end, c_len = start2, end2, len2

                    if dst_id == dst_reads[0][0]:
                        if dst_reads[0][1] == '+':
                            if c_start != 0:
                                rejected += 1
                                continue
                        else:
                            if c_end != c_len:
                                rejected += 1
                                continue
                    else:
                        if dst_reads[-1][1] == '+':
                            if c_end != c_len:
                                rejected += 1
                                continue
                        else:
                            if c_start != 0:
                                rejected += 1
                                continue

                if (src_n_id, dst_n_id) in utgchecker or (src_rev_n_id, dst_rev_n_id) in utgchecker:
                    duplicates += 1
                    continue
                else:
                    utgchecker.add((src_n_id, dst_n_id))

            elif src_id in R2N:
                if src[1] == '+':
                    src_n_id = R2N[src_id][0]
                else:
                    src_n_id = R2N[src_id][1]

                src_reads = N2R[src_n_id]

                if isinstance(src_reads, list) and len(src_reads) > 1:
                    if src_id != src_reads[0][0] and src_id != src_reads[-1][0]:
                        rejected += 1
                        continue

                    if src_id == id1:
                        c_start, c_end, c_len = start1, end1, len1
                    else:
                        c_start, c_end, c_len = start2, end2, len2

                    if src_id == src_reads[0][0]:
                        if src_reads[0][1] == '+':
                            if c_start != 0:
                                rejected += 1
                                continue
                        else:
                            if c_end != c_len:
                                rejected += 1
                                continue
                    else:
                        if src_reads[-1][1] == '+':
                            if c_end != c_len:
                                rejected += 1
                                continue
                        else:
                            if c_start != 0:
                                rejected += 1
                                continue

                    if src_n_id in ghost_utg_checker[dst[1]][dst_id]:
                        duplicates += 1
                        continue
                    else:
                        ghost_utg_checker[dst[1]][dst_id].add(src_n_id)

            elif dst_id in R2N:
                if dst[1] == '+':
                    dst_n_id = R2N[dst_id][0]
                else:
                    dst_n_id = R2N[dst_id][1]

                dst_reads = N2R[dst_n_id]

                if isinstance(dst_reads, list) and len(dst_reads) > 1:
                    if dst_id != dst_reads[0][0] and dst_id != dst_reads[-1][0]:
                        rejected += 1
                        continue

                    if dst_id == id1:
                        c_start, c_end, c_len = start1, end1, len1
                    else:
                        c_start, c_end, c_len = start2, end2, len2

                    if dst_id == dst_reads[0][0]:
                        if dst_reads[0][1] == '+':
                            if c_start != 0:
                                rejected += 1
                                continue
                        else:
                            if c_end != c_len:
                                rejected += 1
                                continue
                    else:
                        if dst_reads[-1][1] == '+':
                            if c_end != c_len:
                                rejected += 1
                                continue
                        else:
                            if c_start != 0:
                                rejected += 1
                                continue

                    if dst_n_id in ghost_utg_checker[src[1]][src_id]:
                        duplicates += 1
                        continue
                    else:
                        ghost_utg_checker[src[1]][src_id].add(dst_n_id)

            dupchecker.add((id1, id2))
            res.append(row)

    print("Preprocessing done! Number of duplicates:", duplicates, "Number of rejected:", rejected)
    return res

# For multiprocessing
def parse_row(row):
    '''
    Returns
    'code' : 0 if rejected, 1 if both src and dst are in gfa, 2 if only either src or dst is in gfa
    'data' : None if code == 0, respective information otherwise
    '''
    data = None

    if not HIFI_R2S or not R2N or not N2R or not SUCCESSOR_DICT or not READS_PARSED:
        raise ValueError("Global objects not set!")

    row_split = row.strip().split()

    ## What are these last 3 fields? ##
    id1, len1, start1, end1, orientation, id2, len2, start2, end2, _, _, _ = row_split
    len1, start1, end1, len2, start2, end2 = int(len1), int(start1), int(end1), int(len2), int(start2), int(end2)

    src, dst = None, None
    if orientation == '+':
        if start1 == 0 and start2 == 0:
            return 0, data
        elif end1 == len1 and end2 == len2:
            return 0, data
        elif end1 == len1 and start2 == 0:
            src, dst = (id1, '+'), (id2, '+')
            src_rev, dst_rev = (id2, '-'), (id1, '-')
        elif start1 == 0 and end2 == len2:
            src, dst = (id2, '+'), (id1, '+')
            src_rev, dst_rev = (id1, '-'), (id2, '-')
        else:
            return 0, data
    else:
        if start1 == 0 and end2 == len2:
            return 0, data
        elif end1 == len1 and start2 == 0:
            return 0, data
        elif end1 == len1 and end2 == len2:
            src, dst = (id1, '+'), (id2, '-')
            src_rev, dst_rev = (id2, '+'), (id1, '-')
        elif start1 == 0 and start2 == 0:
            src, dst = (id1, '-'), (id2, '+')
            src_rev, dst_rev = (id2, '-'), (id1, '+')
        else:
            return 0, data
    
    src_id, dst_id = src[0], dst[0]
    
    if str(src_id) not in READS_PARSED and str(dst_id) not in READS_PARSED: 
        return 3, row

    lock.acquire()
    if src[1] == '+' and dst[1] == '+':
        src_seq, _ = get_seqs(src_id, HIFI_R2S, UL_R2S)
        dst_seq, _ = get_seqs(dst_id, HIFI_R2S, UL_R2S)
    elif src[1] == '+' and dst[1] == '-':
        src_seq, _ = get_seqs(src_id, HIFI_R2S, UL_R2S)
        _, dst_seq = get_seqs(dst_id, HIFI_R2S, UL_R2S)
    elif src[1] == '-' and dst[1] == '+':
        _, src_seq = get_seqs(src_id, HIFI_R2S, UL_R2S)
        dst_seq, _ = get_seqs(dst_id, HIFI_R2S, UL_R2S)
    else:
        raise Exception("Unrecognised orientation pairing.")
    lock.release()

    if str(src_id) in READS_PARSED and str(dst_id) in READS_PARSED:
        c_ol_len = end1-start1 # overlapping region length might not always be equal between source and target. but we always take source for ol length
        edit_dist = edlib.align(src_seq, dst_seq)['editDistance']
        c_ol_similarity = 1 - edit_dist / c_ol_len
        if src[0] == id1:
            src_len, dst_len = len1, len2
            c_prefix_len, c_prefix_len_rev = len1-c_ol_len, len2-c_ol_len
        else:
            src_len, dst_len = len2, len1
            c_prefix_len, c_prefix_len_rev = len2-c_ol_len, len1-c_ol_len

        data = defaultdict(list)

        data['ol_similarity'].append(c_ol_similarity)
        data['ol_len'].append(c_ol_len)
        data['prefix_len'].append(c_prefix_len)
        data['valid_src'].append(src)
        data['valid_dst'].append(dst)

        data['ol_similarity'].append(c_ol_similarity)
        data['ol_len'].append(c_ol_len)
        data['prefix_len'].append(c_prefix_len_rev)
        data['valid_src'].append(src_rev)
        data['valid_dst'].append(dst_rev)

        return 1, data

    if str(src_id) not in READS_PARSED:
        data = { 
            '+' : defaultdict(create_list_dd),
            '-' : defaultdict(create_list_dd)
        }

        c_ol_len = end1-start1 # overlapping region length might not always be equal between source and target. but we always take source for ol length
        edit_dist = edlib.align(src_seq, dst_seq)['editDistance']
        c_ol_similarity = 1 - edit_dist / c_ol_len
        if src[0] == id1:
            src_len, dst_len = len1, len2
            c_prefix_len, c_prefix_len_rev = len1-c_ol_len, len2-c_ol_len
        else:
            src_len, dst_len = len2, len1
            c_prefix_len, c_prefix_len_rev = len2-c_ol_len, len1-c_ol_len

        data[src[1]][src_id]['outs'].append(dst)
        data[src[1]][src_id]['ol_len_outs'].append(c_ol_len)
        data[src[1]][src_id]['ol_similarity_outs'].append(c_ol_similarity)   
        data[src[1]][src_id]['prefix_len_outs'].append(c_prefix_len)
        data[src[1]][src_id]['read_len'] = src_len

        data[dst_rev[1]][src_id]['ins'].append(src_rev)
        data[dst_rev[1]][src_id]['ol_len_ins'].append(c_ol_len)
        data[dst_rev[1]][src_id]['ol_similarity_ins'].append(c_ol_similarity)
        data[dst_rev[1]][src_id]['prefix_len_ins'].append(c_prefix_len_rev)
        data[dst_rev[1]][src_id]['read_len'] = src_len

        return 2, data

    if str(dst_id) not in READS_PARSED:
        data = { 
            '+' : defaultdict(create_list_dd),
            '-' : defaultdict(create_list_dd)
        }

        c_ol_len = end1-start1 # overlapping region length might not always be equal between source and target. but we always take source for ol length
        edit_dist = edlib.align(src_seq, dst_seq)['editDistance']
        c_ol_similarity = 1 - edit_dist / c_ol_len
        if src[0] == id1:
            src_len, dst_len = len1, len2
            c_prefix_len, c_prefix_len_rev = len1-c_ol_len, len2-c_ol_len
        else:
            src_len, dst_len = len2, len1
            c_prefix_len, c_prefix_len_rev = len2-c_ol_len, len1-c_ol_len

        data[dst[1]][dst_id]['ins'].append(src)
        data[dst[1]][dst_id]['ol_len_ins'].append(c_ol_len)
        data[dst[1]][dst_id]['ol_similarity_ins'].append(c_ol_similarity)
        data[dst[1]][dst_id]['prefix_len_ins'].append(c_prefix_len)
        data[dst[1]][dst_id]['read_len'] = dst_len

        data[src_rev[1]][dst_id]['outs'].append(dst_rev)
        data[src_rev[1]][dst_id]['ol_len_outs'].append(c_ol_len)
        data[src_rev[1]][dst_id]['ol_similarity_outs'].append(c_ol_similarity)
        data[src_rev[1]][dst_id]['prefix_len_outs'].append(c_prefix_len_rev)
        data[src_rev[1]][dst_id]['read_len'] = dst_len

        return 2, data

def parse_paf(paf_path, aux):
    '''
    paf_data = {
        ghost_edges = {
            'valid_src' : source nodes,
            'valid_dst' : destination nodes,
            'ol_len' : respective overlap lengths,
            'ol_similarity' : respective overlap similarities,
            'prefix_len' : respective prefix lengths,
            'edge_hops' : respective edge hops,
        },
        ghost_nodes = {
            'hop_<n>' {
                '+' : {
                    read_id : {
                        'read_len' : Read length for this read
                        'outs' : [read_id, ...]
                        'ol_len_outs' : [ol_len, ...],
                        'ol_similarity_outs' : [ol_similarity, ...],
                        'prefix_len_outs' : [prefix_len, ...],
                        'ins' : [read_id, ...],
                        'ol_len_ins' : [ol_len, ...],
                        'ol_similarity_ins' : [ol_similarity, ...],
                        'prefix_len_ins' : [prefix_len, ...],
                    }, 
                    read_id_2 : { ... },
                    ...
                },
                '-' : { ... }
            },
            'hop_<n+1>' : { ... }
        }
    }
    '''
    print("Parsing paf file...")
    
    global HIFI_R2S, UL_R2S, R2N, SUCCESSOR_DICT, N2R, READS_PARSED
    R2N, SUCCESSOR_DICT, N2R, READS_PARSED = aux['r2n'], aux['successor_dict'], aux['n2r'], set()
    HIFI_R2S, UL_R2S = aux['hifi_r2s'], aux['ul_r2s']

    for c_n_id in sorted(N2R.keys()):
        if c_n_id % 2 != 0: continue # Skip all virtual nodes
        read_id = N2R[c_n_id]
        if isinstance(read_id, list):
            READS_PARSED.add(read_id[0][0]); READS_PARSED.add(read_id[-1][0])
        else:
            READS_PARSED.add(read_id)

    with open(paf_path) as f:
        rows = f.readlines()

    rows = preprocess_rows(rows)
    curr_rows = deepcopy(rows)
    cutoff = len(curr_rows) * 0.01

    valid_src, valid_dst, ol_len, ol_similarity, prefix_len, edge_hops = [], [], [], [], [], []
    ghosts = {}
    
    next_rows, hop = [], 1

    while len(curr_rows) > cutoff and hop <= 1: # to include data beyond the first hop, simply remove the "and hop <= 1"
        print(f"Starting run for Hop {hop}. nrows: {len(curr_rows)}, cutoff: {cutoff}")
        curr_ghost_info = {'+':defaultdict(create_list_dd), '-':defaultdict(create_list_dd)}

        with Pool(40, initializer=init_lock, initargs=(Lock(),)) as pool:
            results = pool.imap_unordered(parse_row, iter(curr_rows), chunksize=160)
            for code, data in tqdm(results, total=len(curr_rows), ncols=120):
                if code == 0: 
                    continue
                elif code == 1:
                    ol_similarity.extend(data['ol_similarity'])
                    ol_len.extend(data['ol_len'])
                    prefix_len.extend(data['prefix_len'])
                    valid_src.extend(data['valid_src']) 
                    valid_dst.extend(data['valid_dst'])
                    edge_hops.extend([hop]*len(data['valid_src']))
                elif code == 2:
                    for orient, d in data.items():
                        for id, curr_data in d.items():
                            for label in ['outs', 'ol_len_outs', 'ol_similarity_outs', 'prefix_len_outs', 'ins', 'ol_len_ins', 'ol_similarity_ins', 'prefix_len_ins']:
                                curr_ghost_info[orient][id][label].extend(curr_data[label])

                            curr_ghost_info[orient][id]['read_len'] = curr_data['read_len']
                elif code == 3:
                    next_rows.append(data)

        assert set(curr_ghost_info['+'].keys()) == set(curr_ghost_info['-'].keys()), "Missing real-virtual node pair."
        for read_id in curr_ghost_info['+'].keys():
            READS_PARSED.add(str(read_id))

        print(f"Finished run for Hop {hop}. nrows in hop: {len(curr_rows) - len(next_rows)}")
        ghosts['hop_'+str(hop)] = curr_ghost_info
        curr_rows = deepcopy(next_rows)
        next_rows = []
        hop += 1

    data = {
        'ghost_edges' : {
            'valid_src' : valid_src,
            'valid_dst' : valid_dst,
            'ol_len' : ol_len,
            'ol_similarity' : ol_similarity,
            'prefix_len' : prefix_len,
            'edge_hops' : edge_hops,
        },
        'ghost_nodes' : ghosts
    }

    return data

