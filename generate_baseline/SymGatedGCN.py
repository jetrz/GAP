import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class SymGatedGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers,
                 hidden_edge_scores, batch_norm, nb_pos_enc=0, nr_classes=1, dropout=None, stop_head=False, pred_dropout=0):
        super().__init__()
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features)
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features)
        self.gnn = SymGatedGCN_processor(num_layers, hidden_features, batch_norm, dropout=dropout)
        self.predictor = ScorePredictor(hidden_features, hidden_edge_scores, nr_classes, dropout=pred_dropout)

        self.stop_head = stop_head
        if self.stop_head:
            self.stop_node_predictor = StopNodePredictor(hidden_features, hidden_edge_scores, 1, dropout=pred_dropout)
    def forward(self, graph, x, e):
        x = self.linear1_node(x)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)

        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e)
        if self.stop_head:
            stop_scores = self.stop_node_predictor(x)
            return scores, stop_scores
        else:
            return scores, None
        
class SymGatedGCN_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, batch_norm, dropout=None):
        super().__init__()
        self.convs = nn.ModuleList([
            SymGatedGCN(hidden_features, hidden_features, batch_norm, dropout) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)):
            h, e = self.convs[i](graph, h, e)
        return h, e

class SymGatedGCN(nn.Module):
    """
    Symmetric GatedGCN, based on the idea of  GatedGCN from 'Residual Gated Graph ConvNets'
    paper by Xavier Bresson and Thomas Laurent, ICLR 2018.
    https://arxiv.org/pdf/1711.07553v2.pdf
    """

    def __init__(self, in_channels, out_channels, batch_norm, dropout=None, residual=True):
        super().__init__()
        if dropout:
            # print(f'Using dropout: {dropout}')
            self.dropout = dropout
        else:
            # print(f'Using dropout: 0.00')
            self.dropout = 0.0
        self.batch_norm = batch_norm
        self.residual = residual

        if in_channels != out_channels:
            self.residual = False

        dtype = torch.float32

        self.A_1 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.A_2 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.A_3 = nn.Linear(in_channels, out_channels, dtype=dtype)

        self.B_1 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.B_2 = nn.Linear(in_channels, out_channels, dtype=dtype)
        self.B_3 = nn.Linear(in_channels, out_channels, dtype=dtype)

        if batch_norm:  # batch normalization
            self.bn_h = nn.BatchNorm1d(out_channels, track_running_stats=True)
            self.bn_e = nn.BatchNorm1d(out_channels, track_running_stats=True)
        else:  # layer normalization
            self.bn_h = nn.LayerNorm(out_channels)
            self.bn_e = nn.LayerNorm(out_channels)
    def forward(self, g, h, e):
        """Return updated node representations."""
        with g.local_scope():
            h_in = h.clone()
            e_in = e.clone()

            g.ndata['h'] = h
            g.edata['e'] = e

            g.ndata['A1h'] = self.A_1(h)
            g.ndata['A2h'] = self.A_2(h)
            g.ndata['A3h'] = self.A_3(h)

            g.ndata['B1h'] = self.B_1(h)
            g.ndata['B2h'] = self.B_2(h)
            g.edata['B3e'] = self.B_3(e)

            g_reverse = dgl.reverse(g, copy_ndata=True, copy_edata=True)

            # Reference: https://github.com/graphdeeplearning/benchmarking-gnns/blob/master-dgl-0.6/layers/gated_gcn_layer.py

            # Forward-message passing
            g.apply_edges(fn.u_add_v('B1h', 'B2h', 'B12h'))
            e_ji = g.edata['B12h'] + g.edata['B3e']
            e_ji = self.bn_e(e_ji)
            e_ji = F.relu(e_ji)
            if self.residual:
                e_ji = e_ji + e_in
            g.edata['e_ji'] = e_ji
            g.edata['sigma_f'] = torch.sigmoid(g.edata['e_ji'])
            g.update_all(fn.u_mul_e('A2h', 'sigma_f', 'm_f'), fn.sum('m_f', 'sum_sigma_h_f'))
            g.update_all(fn.copy_e('sigma_f', 'm_f'), fn.sum('m_f', 'sum_sigma_f'))
            g.ndata['h_forward'] = g.ndata['sum_sigma_h_f'] / (g.ndata['sum_sigma_f'] + 1e-6)

            # Backward-message passing
            g_reverse.apply_edges(fn.u_add_v('B2h', 'B1h', 'B21h'))
            e_ik = g_reverse.edata['B21h'] + g_reverse.edata['B3e']
            e_ik = self.bn_e(e_ik)
            e_ik = F.relu(e_ik)
            if self.residual:
                e_ik = e_ik + e_in
            g_reverse.edata['e_ik'] = e_ik
            g_reverse.edata['sigma_b'] = torch.sigmoid(g_reverse.edata['e_ik'])
            g_reverse.update_all(fn.u_mul_e('A3h', 'sigma_b', 'm_b'), fn.sum('m_b', 'sum_sigma_h_b'))
            g_reverse.update_all(fn.copy_e('sigma_b', 'm_b'), fn.sum('m_b', 'sum_sigma_b'))
            g_reverse.ndata['h_backward'] = g_reverse.ndata['sum_sigma_h_b'] / (g_reverse.ndata['sum_sigma_b'] + 1e-6)

            h = g.ndata['A1h'] + g.ndata['h_forward'] + g_reverse.ndata['h_backward']

            h = self.bn_h(h)

            h = F.relu(h)

            if self.residual:
                h = h + h_in

            h = F.dropout(h, self.dropout, training=self.training)
            e = g.edata['e_ji']

            return h, e
        
class ScorePredictor(nn.Module):
    def __init__(self, in_features, hidden_edge_scores, nr_classes, dropout=0.0):
        super().__init__()
        self.W1 = nn.Linear(3 * in_features, hidden_edge_scores)
        #self.Wa = nn.Linear(hidden_edge_scores, hidden_edge_scores)
        #self.Wb = nn.Linear(hidden_edge_scores, hidden_edge_scores)
        self.W2 = nn.Linear(hidden_edge_scores, 32)
        self.W3 = nn.Linear(32, nr_classes)
        self.dropout = nn.Dropout(p=dropout)

    def apply_edges(self, edges):
        data = torch.cat((edges.src['x'], edges.dst['x'], edges.data['e']), dim=1)
        h = self.W1(data)
        h = torch.relu(h)
        h = self.dropout(h)
        #####
        """h = self.Wa(h)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.Wb(h)
        h = torch.relu(h)
        h = self.dropout(h)"""
        #####
        h = self.W2(h)
        h = torch.relu(h)
        h = self.dropout(h)
        score = self.W3(h)
        #score = self.act(score)
        return {'score': score}

    def forward(self, graph, x, e):
        with graph.local_scope():
            graph.ndata['x'] = x
            graph.edata['e'] = e
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class StopNodePredictor(nn.Module):
    def __init__(self, dim_latent, hidden_edge_scores, output_dim, dropout=0.0):
        super().__init__()
        self.W1 = nn.Linear(dim_latent, dim_latent)
        self.W2 = nn.Linear(dim_latent, hidden_edge_scores)
        self.W3 = nn.Linear(hidden_edge_scores, output_dim)
        self.to_squeeze = output_dim == 1
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h = self.W1(x)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.W2(h)
        h = torch.relu(h)
        h = self.dropout(h)
        score = self.W3(h)
        if self.to_squeeze:
            score = score.squeeze()
        return score