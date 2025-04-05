from collections import defaultdict

class Edge():
    def __init__(self, new_src_nid, new_dst_nid, old_src_nid, old_dst_nid, prefix_len, ol_len, ol_sim):
        self.new_src_nid = new_src_nid
        self.new_dst_nid = new_dst_nid
        self.old_src_nid = old_src_nid
        self.old_dst_nid = old_dst_nid
        self.prefix_len = prefix_len
        self.ol_len = ol_len
        self.ol_sim = ol_sim

    def __str__(self):
        return f"New NIDs: {self.new_src_nid}->{self.new_dst_nid} | Old NIDs:{self.old_src_nid}->{self.old_dst_nid} | OL Len: {self.ol_len} | OL Sim: {self.ol_sim} | Prefix Len: {self.prefix_len}"

class AdjList():
    """
    Maps new_src_nid to edges.
    """

    def __init__(self):
        self.adj_list = defaultdict(set)
        self.rev_adj_list = defaultdict(set)
        self.e2re = {} # edge to reverse edge
        self.re2e = {} # reverse edge to edge

    def add_edge(self, edge):
        self.adj_list[edge.new_src_nid].add(edge)
        rev_edge = Edge(
            new_src_nid=edge.new_dst_nid,
            new_dst_nid=edge.new_src_nid,
            old_src_nid=edge.old_dst_nid,
            old_dst_nid=edge.old_src_nid,
            prefix_len=edge.prefix_len,
            ol_len=edge.ol_len,
            ol_sim=edge.ol_sim
        )
        self.rev_adj_list[edge.new_dst_nid].add(rev_edge)
        self.e2re[edge] = rev_edge
        self.re2e[rev_edge] = edge

    def remove_edge(self, edge):
        neighbours = self.adj_list[edge.new_src_nid]
        if edge not in neighbours:
            print("WARNING: Removing an edge that does not exist!")
        re = self.e2re[edge]
        self.adj_list[edge.new_src_nid].discard(edge)
        self.rev_adj_list[edge.new_dst_nid].discard(re)
        del self.e2re[edge]
        del self.re2e[re]

    def get_edge(self, new_src_nid, new_dst_nid):
        for e in self.adj_list[new_src_nid]:
            if e.new_dst_nid == new_dst_nid: 
                return e
            
        print("WARNING: Retrieving an edge that does not exist!")
        return None
            
    def remove_node(self, n_id):
        # Remove outgoing edges
        for e in self.adj_list[n_id]:
            re = self.e2re[e]
            self.rev_adj_list[e.new_dst_nid].discard(re)
            del self.re2e[re]
            del self.e2re[e]
        del self.adj_list[n_id]

        # Remove incoming edges
        for re in self.rev_adj_list[n_id]:
            e = self.re2e[re]
            self.adj_list[re.new_dst_nid].discard(e)
            del self.re2e[re]
            del self.e2re[e]
        del self.rev_adj_list[n_id]

    def get_in_out_deg(self, n_id):
        return len(self.rev_adj_list[n_id]), len(self.adj_list[n_id])

    def sanity_check(self):
        """
        Checks if the two adjacency lists are equivalent.
        """
        simple_adj_list, simple_rev_adj_list = defaultdict(set), defaultdict(set)
        for src, neighs in self.adj_list.items():
            simple_adj_list[src] = set(e.new_dst_nid for e in neighs)
        for src, neighs in self.rev_adj_list.items():
            simple_rev_adj_list[src] = set(e.new_dst_nid for e in neighs)

        for src, neighs in simple_adj_list.items():
            for dst in neighs:
                if src not in simple_rev_adj_list.get(dst, []):
                    print("Adjacency list sanity check failed!")
                    return False
        for src, neighs in simple_rev_adj_list.items():
            for dst in neighs:
                if src not in simple_adj_list.get(dst, []):
                    print("Adjacency list sanity check failed!")
                    return False

        print("Adjacency list sanity check passed!")
        return True 

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