import dgl, os, subprocess, torch
from Bio import SeqIO

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
    Runs and saves minigraph.
    """
    print(f"Saving assembly...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    asm_path = save_path+"0_assembly.fasta"
    SeqIO.write(contigs, asm_path, 'fasta')

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
    