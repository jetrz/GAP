import dgl, glob, os, random, subprocess, torch
from Bio import SeqIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pickle

def timedelta_to_str(delta):
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}h {minutes}m {seconds}s'

def pyg_to_dgl(g, node_attrs, edge_attrs):
    def to_tensor(x):
        if isinstance(x, list):
            return torch.tensor(x)
        return x
    
    u, v = g.edge_index
    dgl_g = dgl.graph((u, v), num_nodes=g[node_attrs[0]].shape[0])

    # Adding node features
    for attr in node_attrs:
        dgl_g.ndata[attr] = to_tensor(g[attr])

    # Adding edge features
    for attr in edge_attrs:
        dgl_g.edata[attr] = to_tensor(g[attr])

    return dgl_g

def asm_metrics(contigs, save_path, ref_path, minigraph_path, paftools_path):
    """
    Saves the assembly and runs minigraph.
    """
    print(f"Saving assembly...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    asm_path = save_path+"0_assembly.fasta"
    SeqIO.write(contigs, asm_path, 'fasta')

    # need to replace ids this way for some reason else T2T_chromosomes tool won't work. idk why
    temp_name = f"temp_{random.randint(1,9999999)}.fasta"
    cmd = 'seqkit replace -p .+ -r \"ctg{nr}\" --nr-width 10 0_assembly.fasta > ' + temp_name
    subprocess.run(cmd, shell=True, cwd=save_path[:-1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    os.remove(asm_path)
    os.rename(save_path+temp_name, asm_path)

    print(f"Running minigraph...")
    paf = save_path+"asm.paf"
    cmd = f'{minigraph_path} -t32 -xasm -g10k -r10k --show-unmap=yes {ref_path} {asm_path}'.split(' ')
    with open(paf, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    p.wait()

    print(f"Running paftools...")
    cmd = f'k8 {paftools_path} asmstat {ref_path+".fai"} {paf}'.split()
    report = save_path+"minigraph.txt"
    with open(report, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    p.wait()
    with open(report) as f:
        report = f.read()
        print(report)

def yak_metrics(save_path, yak1, yak2, yak_path):
    """
    IMPT: asm_metrics have to be run before this to generate the assembly!
    
    Yak triobinning result files have following info:
    C       F  seqName     type      startPos  endPos    count
    C       W  #switchErr  denominator  switchErrRate
    C       H  #hammingErr denominator  hammingErrRate
    C       N  #totPatKmer #totMatKmer  errRate
    """
    print("Running yak trioeval...")
    save_file = save_path+"phs.txt"
    cmd = f'{yak_path} trioeval -t16 {yak1} {yak2} {save_path}0_assembly.fasta > {save_file}'.split()
    with open(save_file, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    p.wait()

    switch_err, hamming_err = None, None
    with open(save_file, 'r') as file:
        # Read all the lines and reverse them
        lines = file.readlines()
        reversed_lines = reversed(lines)
        for line in reversed_lines:
            if line.startswith('W'):
                switch_err = float(line.split()[3])
            elif line.startswith('H'):
                hamming_err = float(line.split()[3])
            if switch_err is not None and hamming_err is not None:
                break

    if switch_err is None or hamming_err is None:
        print("YAK Switch/Hamming error not found!")
    else:
        print(f"YAK Switch Err: {switch_err*100:.4f}%, YAK Hamming Err: {hamming_err*100:.4f}%")

def t2t_metrics(save_path, t2t_chr_path, ref_path, motif):
    print("Running T2T eval...")
    if os.path.exists(save_path+"0_assembly.fasta.seqkit.fai"):
        # This has been run before, delete and re-run
        os.remove(save_path+"0_assembly.fasta.seqkit.fai")
        pattern = os.path.join(save_path, "T2T*")
        ftd = glob.glob(pattern)
        for f in ftd:
            os.remove(f)

    cmd = f"{t2t_chr_path} -a {save_path}0_assembly.fasta -r {ref_path} -m {motif} -t 10"
    print("Command:", cmd)
    subprocess.run(cmd, shell=True, cwd=save_path[:-1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    aligned_path = f"{save_path}T2T_sequences_alignment_T2T.txt"
    with open(aligned_path, 'r') as f:
        aligned_count = sum(1 for _ in f)
    unaligned_path = f"{save_path}T2T_sequences_motif_T2T.txt"
    with open(unaligned_path, 'r') as f:
        unaligned_count = sum(1 for _ in f)
        
    print(f"Unaligned T2T: {unaligned_count} | Aligned T2T: {aligned_count}")
    return

def get_seqs(id, hifi_r2s, ul_r2s):
    if id in hifi_r2s:
        return str(hifi_r2s[id][:]), str(-hifi_r2s[id][:])
    elif ul_r2s is not None and id in ul_r2s:
        return str(ul_r2s[id][:]), str(-ul_r2s[id][:])
    else:
        raise ValueError("Read not present in seq dataset FASTAs!")
    
def analyse_graph(adj_list, telo_ref, walks, save_path):
    # For debugging
    nxg = nx.DiGraph()
    for source, neighs in adj_list.adj_list.items():
        for n in neighs:
            nxg.add_edge(source, n.new_dst_nid)
    pickle.dump(nxg, open(save_path+"nx_graph.pkl", 'wb'))

    colors = []
    for n in nxg.nodes():
        if n not in telo_ref:
            colors.append(0)
        elif telo_ref[n]['start'] and telo_ref[n]['end']:
            colors.append(1)
        elif telo_ref[n]['start']:
            if telo_ref[n]['start'] == '+':
                colors.append(2)
            else:
                colors.append(3)
        elif telo_ref[n]['end']:
            if telo_ref[n]['end'] == '+':
                colors.append(4)
            else:
                colors.append(5)
        else:
            colors.append(0)

    color_map = {
        0: 'grey',
        1: 'green',
        2: 'red',
        3: 'blue',
        4: 'yellow',
        5: 'purple'
    }
    labels = {
        0: 'No telomere',
        1: 'Both start and end',
        2: 'Start (+)',
        3: 'Start (-)',
        4: 'End (+)',
        5: 'End (-)'
    }
    colors = [color_map[c] for c in colors]

    pos = nx.spring_layout(nxg, seed=42)
    plt.figure(figsize=(25,25))
    nx.draw(nxg, pos=pos, with_labels=True, node_color=colors, node_size=50, font_size=9)
    legend_handles = [mpatches.Patch(color=color_map[key], label=labels[key]) for key in sorted(color_map)]
    plt.legend(handles=legend_handles, loc='best')
    plt.savefig(save_path+'nx_graph_before.png')
    plt.clf()

    nxg.remove_edges_from(list(nxg.edges()))
    for w in walks:
        for i, n in enumerate(w[:-1]):
            nxg.add_edge(n, w[i+1])

    nx.draw(nxg, pos=pos, with_labels=True, node_color=colors, node_size=50, font_size=9)
    legend_handles = [mpatches.Patch(color=color_map[key], label=labels[key]) for key in sorted(color_map)]
    plt.legend(handles=legend_handles, loc='best')
    plt.savefig(save_path+'nx_graph_after.png')

    return

def print_ascii():
    """hehe"""
    ascii = '''
    \n
    .·´¯`·. ·´¯·.
    __|__
    | |__ ╲╲    ╲
    |ロ |  ╲╲ (\~/) 
    |ロ |   ╲╲( •ω•)    Running GAP...
    |ロ |    ╲⊂   づ
    |ロ |     ╲╲ ⊃⊃╲
    |ロ |___   ╲|___  ╲|___
    \n
    '''
    print(ascii)
    