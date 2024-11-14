from Bio import Seq, SeqIO
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import dgl, math, pickle, os, random, torch

from .SymGatedGCN import SymGatedGCNModel
from misc.utils import asm_metrics, timedelta_to_str

# Hardcoded values for feature normalisation
STDS_AND_MEANS = {
    'degree_mean': 10.10,
    'degree_std': 2.74,
    'ol_len_mean': 10341,
    'ol_len_std': 5521
}

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)

def preprocess_graph(g):
    ol_len = g.edata['overlap_length'].float()
    ol_len = (ol_len - STDS_AND_MEANS["ol_len_mean"]) / STDS_AND_MEANS["ol_len_std"]
    ol_sim = g.edata['overlap_similarity']
    edge_features = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)

    pe_in = g.in_degrees().float().unsqueeze(1)
    pe_out = g.out_degrees().float().unsqueeze(1)
    pe_in = ((pe_in - STDS_AND_MEANS["degree_mean"]) / STDS_AND_MEANS["degree_std"])
    pe_out = ((pe_out - STDS_AND_MEANS["degree_mean"]) / STDS_AND_MEANS["degree_std"])
    x = torch.ones(g.num_nodes(), 1)
    node_features = torch.cat((pe_in, pe_out, x), dim=1)
        
    return g.int(), node_features, edge_features

def graph_to_successor_dict(graph):
    successors_dict = {}
    for node in graph.nodes():
        node = node.item()  # Convert to Python int
        successors = graph.successors(node).tolist()  # Get successors and convert to list
        successors_dict[node] = successors
    return successors_dict

def graph_to_predecessor_dict(graph):
    predecessors_dict = {}
    for node in graph.nodes():
        node = node.item()  # Convert to Python int
        predecessors = graph.predecessors(node).tolist()  # Get predecessors and convert to list
        predecessors_dict[node] = predecessors
    return predecessors_dict

def get_contig_length(walk, graph):
    total_length = 0
    idx_src = walk[:-1]
    idx_dst = walk[1:]
    prefix = graph.edges[idx_src, idx_dst].data['prefix_length']
    total_length = prefix.sum().item()
    total_length += graph.ndata['read_length'][walk[-1]]
    return total_length.item()

def get_subgraph(g, visited, device):
    """Remove the visited nodes from the graph."""
    remove_node_idx = torch.LongTensor([item for item in visited])
    list_node_idx = torch.arange(g.num_nodes())
    keep_node_idx = torch.ones(g.num_nodes())
    keep_node_idx[remove_node_idx] = 0
    keep_node_idx = list_node_idx[keep_node_idx==1].int().to(device)

    sub_g = dgl.node_subgraph(g, keep_node_idx, store_ids=True)
    sub_g.ndata['idx_nodes'] = torch.arange(sub_g.num_nodes()).to(device)
    map_subg_to_g = sub_g.ndata[dgl.NID]
    return sub_g, map_subg_to_g

def sample_edges(prob_edges, nb_paths):
    """Sample edges where the probability is not zero."""
    if prob_edges.shape[0] > 2 ** 24:
        prob_edges = prob_edges[:2 ** 24]  # Limit size due to torch limitations

    # Convert probabilities to 0 or 1, where 1 indicates a non-zero probability
    eligible_edges = (prob_edges > 0).float()

    # Ensure there's at least one edge to sample from
    if eligible_edges.sum() == 0:
        raise ValueError("All edge probabilities are zero. Cannot sample an edge.")

    # Repeat the eligible_edges tensor to match the number of paths we want to sample
    eligible_edges_nb_paths = eligible_edges.repeat(nb_paths, 1)

    # Initialize an empty tensor for idx_edges with the correct type for indexing
    idx_edges = torch.empty(nb_paths, dtype=torch.long)

    for i in range(nb_paths):
        # Find indices of eligible edges for each path
        eligible_indices = eligible_edges_nb_paths[i].nonzero().squeeze()

        # Check if eligible_indices is not empty to avoid runtime error
        if eligible_indices.nelement() == 0:
            raise ValueError("No eligible edges found for path {}. This should not happen since we check for sum > 0.".format(i))

        # Randomly select one of the eligible indices and ensure it is of type long
        selected_index = eligible_indices[torch.randint(0, len(eligible_indices), (1,))]

        # Assign the selected index directly into idx_edges
        idx_edges[i] = selected_index

    return idx_edges

def get_contigs_greedy(g, succs, preds, edges, nb_paths=50, len_threshold=20):
    """Iteratively search for contigs in a graph until the threshold is met."""
    g = g.to('cpu')
    all_contigs = []
    all_walks_len = []
    all_contigs_len = []
    visited = set()
    idx_contig = -1
    B = 1

    scores = g.edata['score'].to('cpu')
    print(f'Starting to decode with greedy...')
    print(f'num_candidates: {nb_paths}, len_threshold: {len_threshold}\n')

    while True:
        idx_contig += 1
        time_start_sample_edges = datetime.now()
        sub_g, map_subg_to_g = get_subgraph(g, visited, 'cpu')
        if sub_g.num_edges() == 0:
            break

        prob_edges = torch.sigmoid(sub_g.edata['score']).squeeze()

        idx_edges = sample_edges(prob_edges, nb_paths)

        elapsed = timedelta_to_str(datetime.now() - time_start_sample_edges)
        print(f'Elapsed time (sample edges): {elapsed}')

        all_walks = []
        all_visited_iter = []

        all_contig_lens = []
        all_sumLogProbs = []
        all_meanLogProbs = []
        all_meanLogProbs_scaled = []

        print(f'\nidx_contig: {idx_contig}, nb_processed_nodes: {len(visited)}, ' \
              f'nb_remaining_nodes: {g.num_nodes() - len(visited)}, nb_original_nodes: {g.num_nodes()}')

        # Get nb_paths paths for a single iteration, then take the longest one
        time_start_get_candidates = datetime.now()

        with ThreadPoolExecutor(1) as executor:
            results = {}
            start_times = {}

            for e, idx in enumerate(idx_edges):
                src_init_edges = map_subg_to_g[sub_g.edges()[0][idx]].item()
                dst_init_edges = map_subg_to_g[sub_g.edges()[1][idx]].item()
                start_times[e] = datetime.now()
                future = executor.submit(run_greedy_both_ways, src_init_edges, dst_init_edges, scores, succs, edges, visited)
                results[(src_init_edges, dst_init_edges)] = (future, e)

            indx = 0
            for k, (f, e) in results.items():  # key, future 
                walk_f, walk_b, visited_f, visited_b, sumLogProb_f, sumLogProb_b = f.result()
                walk_it = walk_b + walk_f
                visited_iter = visited_f | visited_b
                sumLogProb_it = sumLogProb_f.item() + sumLogProb_b.item()
                len_walk_it = len(walk_it)
                len_contig_it = get_contig_length(walk_it, g)
                if k[0] == k[1]:
                    len_walk_it = 1

                if len_walk_it > 2:
                    meanLogProb_it = sumLogProb_it / (len_walk_it - 2)  # len(walk_f) - 1 + len(walk_b) - 1  <-> starting edge is neglected
                    try:
                        meanLogProb_scaled_it = meanLogProb_it / math.sqrt(len_contig_it)
                    except ValueError:
                        print(f'{indx:<3}: src={k[0]:<8} dst={k[1]:<8} len_walk={len_walk_it:<8} len_contig={len_contig_it:<12}')
                        print(f'Value error: something is wrong here!')
                        meanLogProb_scaled_it = 0
                elif len_walk_it == 2:
                    meanLogProb_it = 0.0
                    try:
                        meanLogProb_scaled_it = meanLogProb_it / math.sqrt(len_contig_it)
                    except ValueError:
                        print(f'{indx:<3}: src={k[0]:<8} dst={k[1]:<8} len_walk={len_walk_it:<8} len_contig={len_contig_it:<12}')
                        print(f'Value error: something is wrong here!')
                        meanLogProb_scaled_it = 0
                else:  # len_walk_it == 1 <-> SELF-LOOP!
                    len_contig_it = 0
                    sumLogProb_it = 0.0
                    meanLogProb_it = 0.0
                    meanLogprob_scaled_it = 0.0
                    print(f'SELF-LOOP!')
                print(f'{indx:<3}: src={k[0]:<8} dst={k[1]:<8} len_walk={len_walk_it:<8} len_contig={len_contig_it:<12} ' \
                      f'sumLogProb={sumLogProb_it:<12.3f} meanLogProb={meanLogProb_it:<12.4} meanLogProb_scaled={meanLogProb_scaled_it:<12.4}')

                indx += 1
                all_walks.append(walk_it)
                all_visited_iter.append(visited_iter)
                all_contig_lens.append(len_contig_it)
                all_sumLogProbs.append(sumLogProb_it)
                all_meanLogProbs.append(meanLogProb_it)
                all_meanLogProbs_scaled.append(meanLogProb_scaled_it)

        best = max(all_contig_lens)
        idxx = all_contig_lens.index(best)

        elapsed = timedelta_to_str(datetime.now() - time_start_get_candidates)
        print(f'Elapsed time (get_candidates): {elapsed}')

        best_walk = all_walks[idxx]
        best_visited = all_visited_iter[idxx]

        # Add all jumped-over nodes
        time_start_get_visited = datetime.now()
        trans = set()
        for ss, dd in zip(best_walk[:-1], best_walk[1:]):
            t1 = set(succs[ss]) & set(preds[dd])
            t2 = {t^1 for t in t1}
            trans = trans | t1 | t2
        best_visited = best_visited | trans

        best_contig_len = all_contig_lens[idxx]
        best_sumLogProb = all_sumLogProbs[idxx]
        best_meanLogProb = all_meanLogProbs[idxx]
        best_meanLogProb_scaled = all_meanLogProbs_scaled[idxx]

        elapsed = timedelta_to_str(datetime.now() - time_start_get_visited)
        print(f'Elapsed time (get visited): {elapsed}')

        print(f'\nChosen walk with index: {idxx}')
        print(f'len_walk={len(best_walk):<8} len_contig={best_contig_len:<12} ' \
              f'sumLogProb={best_sumLogProb:<12.3f} meanLogProb={best_meanLogProb:<12.4} meanLogProb_scaled={best_meanLogProb_scaled:<12.4}\n')

        if best_contig_len < 70000:
            break
        if len(best_walk) < len_threshold:
            break
        all_contigs.append(best_walk)
        all_walks_len.append(len(best_walk))
        all_contigs_len.append(best_contig_len)
        visited |= best_visited
        print(f'All walks len: {all_walks_len}')
        print(f'All contigs len: {all_contigs_len}\n')

    return all_contigs

def run_greedy_both_ways(src, dst, scores, succs, edges, visited):
    temp_visited = visited | {src, src^1, dst, dst^1}
    walk_f, visited_f, sumLogProb_f  = greedy(dst, scores, succs, edges, temp_visited, backwards=False)
    walk_b, visited_b, sumLogProb_b = greedy(src, scores, succs, edges, temp_visited | visited_f, backwards=True)
    return walk_f, walk_b, visited_f, visited_b, sumLogProb_f, sumLogProb_b

def greedy(start, scores, neighbors, edges, visited_old, backwards=False):
    """Greedy walk."""
    if backwards:
        current = start ^ 1
    else:
        current = start
    walk = []
    visited = set()
    iteration = 0
    while True:
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        neighs_current = neighbors[current]
        if len(neighs_current) == 0:
            break
        if len(neighs_current) == 1:
            neighbor = neighs_current[0]
            if neighbor in visited_old or neighbor in visited:
                break
            else:
                current = neighbor
                continue
        
        masked_neighbors = [n for n in neighs_current if not (n in visited_old or n in visited)]
        neighbor_edges = [edges[current, n] for n in masked_neighbors]

        if not neighbor_edges:
            break
        neighbor_p = scores[neighbor_edges]
        score, index = torch.topk(neighbor_p, k=1, dim=0)
        iteration += 1
        current = masked_neighbors[index]
    if backwards:
        walk = list(reversed([w ^ 1 for w in walk]))

    return walk, visited, torch.tensor(0.0)

def walk_to_sequence(walks, graph, aux):
    edges, n2s = aux['edges_full'], aux['n2s']
    contigs = []
    for i, walk in enumerate(walks):
        prefixes = [(src, graph.edata['prefix_length'][edges[src,dst]]) for src, dst in zip(walk[:-1], walk[1:])]

        res = []
        for (src, prefix) in prefixes:
            seq = str(n2s[src])
            res.append(seq[:prefix])

        contig = Seq.Seq(''.join(res) + str(n2s[walk[-1]]))  # TODO: why is this map here? Maybe I can remove it if I work with strings
        contig = SeqIO.SeqRecord(contig)
        contig.id = f'contig_{i+1}'
        contig.description = f'length={len(contig)}'
        contigs.append(contig)

    return contigs

def gnnome_decoding(genome, gnnome_config, paths):
    train_config = gnnome_config['training']
    decode_config = gnnome_config['decoding']

    set_seed(train_config['seed'])
    time_start = datetime.now()

    print(f"Initialising... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    g = dgl.load_graphs(paths['graph']+f'{genome}.dgl')[0][0]
    g, x, e = preprocess_graph(g)

    save_path = paths['baseline']
    if not os.path.isdir(save_path): os.makedirs(save_path)

    # Get scores
    print(f"Processing scores... (Time: {timedelta_to_str(datetime.now() - time_start)})")
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
        model.load_state_dict(torch.load(paths['model'], map_location=torch.device('cpu')))
        model.eval()
        print(f'Computing the scores with the model...\n')
        edge_predictions, stop_predictions = model(g, x, e)

        g.edata['score'] = edge_predictions.squeeze()
        torch.save(g.edata['score'], os.path.join(save_path, f'predicts.pt'))

        succs = graph_to_successor_dict(g)
        preds = graph_to_predecessor_dict(g)

        # Create a list of all edges
        edges = {}
        for idx, (src, dst) in enumerate(zip(g.edges()[0], g.edges()[1])):
            src, dst = src.item(), dst.item()
            edges[(src, dst)] = idx

        # Get walks
        print(f"Generating walks... (Time: {timedelta_to_str(datetime.now() - time_start)})")
        g.edata['prefix_length'] = g.edata['prefix_length'].masked_fill(g.edata['prefix_length'] < 0, 0)

        walks = get_contigs_greedy(g, succs, preds, edges, decode_config['num_decoding_paths'], decode_config['len_threshold'])
        inference_path = os.path.join(save_path, f'walks.pkl')
        pickle.dump(walks, open(f'{inference_path}', 'wb'))

    aux = {}
    # Create a list of all edges
    edges_full = {}  ## I dont know why this is necessary. but when cut transitives some eges are wrong otherwise.
    for idx, (src, dst) in enumerate(zip(g.edges()[0], g.edges()[1])):
        src, dst = src.item(), dst.item()
        edges_full[(src, dst)] = idx
    aux['edges_full'] = edges_full

    print(f"Generating contigs... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    with open(paths['n2s'], 'rb') as f:
        aux['n2s'] = pickle.load(f)
    contigs = walk_to_sequence(walks, g, aux)

    print(f"Calculating assembly metrics... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    asm_metrics(contigs, save_path, paths['ref'], paths['minigraph'], paths['paftools'])

    print(f"Run finished! (Time: {timedelta_to_str(datetime.now() - time_start)})")
    return