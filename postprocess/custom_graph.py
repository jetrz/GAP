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
        return f"New NIDs: {self.new_src_nid}->{self.new_dst_nid}, Old NIDs:{self.old_src_nid}->{self.old_dst_nid}"

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
            
        print("WARNING: Retrieving an edge that does not exist!")
        return None
            
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