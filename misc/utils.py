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

    print(f"YAK Switch Err: {switch_err*100:.4f}%, YAK Hamming Err: {hamming_err*100:.4f}%")

def get_seqs(id, hifi_r2s, ul_r2s):
    if id in hifi_r2s:
        return str(hifi_r2s[id][:]), str(-hifi_r2s[id][:])
    elif ul_r2s is not None and id in ul_r2s:
        return str(ul_r2s[id][:]), str(-ul_r2s[id][:])
    else:
        raise ValueError("Read not present in seq dataset FASTAs!")

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
    