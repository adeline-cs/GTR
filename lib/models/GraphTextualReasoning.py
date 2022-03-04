import os, sys, time, random, string, math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from einops import reduce, rearrange, repeat
from sklearn.metrics.pairwise import cosine_similarity
from GCR_utils import EuclideanDistances, normalize_adj, dense_diff_pool
import cv2




"""The original graph example to generate sub-graph"""
class GraphGeneration(object):
    def __init__(self, k_at_hop, active_connection, pooling, pst_dim, is_train=True):
        self.pst_dim = pst_dim
        self.NodePooling = pooling
        self.is_train = is_train
        self.k_at_hop = k_at_hop
        self.depth = len(self.k_at_hop)
        self.active_connection = active_connection
        self.cluster_threshold = 0.75

    def PositionalEncoding(geo_map, model_dim):
        shape = geo_map.shape
        model_dim = model_dim // shape[1]

        # t0 = time.time()
        pp = np.array([np.power(1000, 2.0 * (j // 2) / model_dim)
                       for j in range(model_dim)]).reshape(model_dim, 1, 1)
        ps = np.repeat(np.expand_dims(geo_map, axis=0), model_dim, axis=0)
        pst_encoding = ps / pp
        # print("A time: {}".format(time.time() - t0))
        pst_encoding[:, 0::2] = np.sin(pst_encoding[:, 0::2])
        pst_encoding[:, 1::2] = np.cos(pst_encoding[:, 1::2])
        pst_encoding = pst_encoding.reshape((shape[0], -1))

        return pst_encoding

    def IndexandClassEncoding(input, dim):
        self.conv2 = nn.Conv2d(input.shape[-1], dim, 1, stride=1 )
	return self.conv2(input)

    def localgraph(self, knn_graph, labels_gt=None):
        # hops[0] for 1-hop neighbors, hops[1] for 2-hop neighbors
        knn_graph = knn_graph[:, :self.k_at_hop[0] + 1]
        hops_list = list()
        one_hops_list = list()
        for index, cluster in enumerate(knn_graph):
            hops = list()
            center_idx = index

            h0 = set(knn_graph[center_idx][1:])
            hops.append(h0)
            # Actually we dont need the loop since the depth is fixed here,
            # But we still remain the code for further revision
            for d in range(1, self.depth):
                hops.append(set())
                for h in hops[-2]:
                    hops[-1].update(set(knn_graph[h][1:self.k_at_hop[d] + 1]))

            hops_set = set([h for hop in hops for h in hop])
            nodes_list = list(hops_set)
            nodes_list.insert(0, center_idx)

            for idx, ips in enumerate(hops_list):
                union = len(list(set(ips).union(set(nodes_list))))
                intersection = len(list(set(ips).intersection(set(nodes_list))))
                iou = intersection / (union + 1e-5)
                if iou > self.cluster_threshold \
                        and center_idx in one_hops_list[idx] \
                        and labels_gt[ips[0]] == labels_gt[center_idx] \
                        and labels_gt[ips[0]] != 0:
                    break

            else:  # not break for loop , performance this code
                hops_list.append(nodes_list)
                one_hops_list.append(h0)

        return hops_list, one_hops_list

    def graphbuild(self, feat_bin, labels_bin, hops_bin, one_hops_bin, knn_graph_bin):

        max_num_nodes = max([len(ips) for hops in hops_bin for ips in hops])

        feat_batch = list()
        adj_batch = list()
        h1id_batch = list()
        gtmat_batch = list()
        for bind, knn_graph in enumerate(knn_graph_bin):
            feat_map = feat_bin[bind]
            hops_list = hops_bin[bind]
            one_hops_list = one_hops_bin[bind]
            labels_gt = labels_bin[bind]

            for idx, ips in enumerate(hops_list):
                num_nodes = int(len(ips))
                center_idx = ips[0]
                one_hops = one_hops_list[idx]
                unique_nodes_map = {j: i for i, j in enumerate(ips)}

                one_hop_idcs = torch.tensor([unique_nodes_map[i] for i in one_hops], dtype=torch.long)
                center_feat = feat_map[torch.tensor(center_idx, dtype=torch.long)]
                feat = feat_map[torch.tensor(ips, dtype=torch.long)] - center_feat

                A = np.zeros((num_nodes, num_nodes))
                feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, feat.shape[1]).cuda()], dim=0)

                for node in ips:
                    neighbors = knn_graph[node, 1:self.active_connection + 1]
                    for n in neighbors:
                        if n in ips:
                            A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                            A[unique_nodes_map[n], unique_nodes_map[node]] = 1

                A = normalize_adj(A, type="DAD")
                A_ = torch.zeros(max_num_nodes, max_num_nodes)
                A_[:num_nodes, :num_nodes] = A

                labels = torch.from_numpy(labels_gt[ips]).type(torch.long)
                one_hop_labels = labels[one_hop_idcs]
                edge_labels = ((labels_gt[center_idx] == one_hop_labels)
                               & labels_gt[center_idx] > 0).long()

                feat_batch.append(feat)
                adj_batch.append(A_)
                h1id_batch.append(one_hop_idcs)
                gtmat_batch.append(edge_labels)

        feat_bth = torch.stack(feat_batch, 0)
        adj_bth = torch.stack(adj_batch, 0)
        h1id_bth = torch.stack(h1id_batch, 0)
        gtmat_bth = torch.stack(gtmat_batch, 0)

        return feat_bth, adj_bth, h1id_bth, gtmat_bth

    def forward(self, feats, gt_data=None):

        knn_graph_bin = list()
        hops_bin = list()
        one_hops_bin = list()
        feat_bin = list()
        labels_bin = list()
        gt_data = gt_data.numpy()
        for bind in range(gt_data.shape[0]):
            ro_num = int(gt_data[bind, 0, 0])
            img_size = int(gt_data[bind, 0, 8])
            geo_map = gt_data[bind, :ro_num, 1:7]
            label = gt_data[bind, :ro_num, 7].astype(np.int32)

            # ## 1. compute euclidean similarity
            ctr_xy = geo_map[:, 0:2]
            similarity_e = np.array(EuclideanDistances(ctr_xy, ctr_xy), dtype=np.float) / img_size

            # ## 2. embedding
            pos_feat = self.PositionalEncoding(geo_map, self.pst_dim)
            pos_feat = torch.from_numpy(pos_feat).cuda().float()
            batch_id = np.zeros((geo_map.shape[0], 1), dtype=np.float32) * bind
            index_feat = self.IndexandClassEncoding(batch_id, self.pst_dim)
            index_feat = torch.from_numpy(index_feat).cuda().float()
            class_feat = self.IndexandClassEncoding(geo_map[], self.pst_dim)
            class_map = np.hstack((batch_id, class_feat.astype(np.float32, copy=False)))
            class_feat = torch.from_numpy(class_map).cuda()
            # ## 3. generate graph node feature
            node_feat = torch.cat((class_map, pos_feat), dim=-1)
            node_feat = torch.cat((node_feat, index_feat), dim=-1)
  
            # # ## 4. computing similarity of Node feature
            similarity_c = 1.0001 - cosine_similarity(class_feat)
            # similarity_matrix = (similarity_e + similarity_c) / 2.0
            similarity_matrix = similarity_e * similarity_c

            # ## 5. compute the knn graph
            knn_graph = np.argsort(similarity_matrix, axis=1)[:, :]
            hops, one_hops = self.localgraph(knn_graph, label)

            # ## 6. Packing data
            feat_bin.append(node_feat)
            labels_bin.append(label)
            hops_bin.append(hops)
            one_hops_bin.append(one_hops)
            knn_graph_bin.append(knn_graph)

        batch_data = self.graphbuild(feat_bin, labels_bin, hops_bin, one_hops_bin, knn_graph_bin)

        return batch_data


"""The meanaggregator for GNN"""
class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x

"""The meta layer of GCN"""
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)
        agg_feats = self.agg(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out

class GraphConvolution(Module):

  def __init__(self, in_features, out_features, bias=True):
    super(GraphConvolution, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = Parameter(torch.FloatTensor(in_features, out_features))
    if bias:
      self.bias = Parameter(torch.FloatTensor(out_features))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.uniform_(-stdv, stdv)

  def norm(self, adj):
    node_num = adj.shape[-1]
    # add remaining self-loops
    self_loop = torch.eye(node_num)
    self_loop = self_loop.reshape((1, node_num, node_num))
    self_loop = self_loop.repeat(adj.shape[0], 1, 1)
    adj_post = adj + self_loop
    # signed adjacent matrix
    deg_abs = torch.sum(torch.abs(adj_post), dim=-1)
    deg_abs_sqrt = deg_abs.pow(-0.5)
    diag_deg = torch.diag_embed(deg_abs_sqrt, dim1=-2, dim2=-1)

    norm_adj = torch.matmul(torch.matmul(diag_deg, adj_post), diag_deg)
    return norm_adj

  def forward(self, input, adj):
    support = torch.matmul(input, self.weight)
    adj_norm = self.norm(adj)
    output = torch.matmul(support.transpose(1, 2), adj_norm.transpose(1, 2))
    output = output.transpose(1, 2)
    if self.bias is not None:
      return output + self.bias
    else:
      return output

  def __repr__(self):
    return self.__class__.__name__ + ' (' \
           + str(self.in_features) + ' -> ' \
           + str(self.out_features) + ')'

class GraphPool(nn.Module):
  def __init__(self, in_feature, hidden_feature, out_feature, in_node, hidden_node, out_node):
    super(ReasoningPool, self).__init__()

    self.build_graph(in_feature, hidden_feature, out_feature, in_node, hidden_node, out_node)
    self.reset_parameters()

    self.pool_tensor = None

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, gcn_layer.GraphConvolution):
        m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
        if m.bias is not None:
          m.bias.data = torch.nn.init.constant_(m.bias.data, 0.0)

  def build_graph(self, in_feature, hidden_feature, out_feature, in_node, hidden_node, out_node):

    # embedding blocks

    self.embed_conv_first = GraphConvolution(
      in_features=in_feature,
      out_features=hidden_feature,
    )
    self.embed_conv_block = GraphConvolution(
      in_features=hidden_feature,
      out_features=hidden_feature,
      
    )
    self.embed_conv_last = GraphConvolution(
      in_features=hidden_feature,
      out_features=out_feature,
      
    )

    # pooling blocks

    self.pool_conv_first = GraphConvolution(
      in_features=in_node,
      out_features=hidden_node,
      
    )
    self.pool_conv_block = GraphConvolution(
      in_features=hidden_node,
      out_features=hidden_node,
      
    )
    self.pool_conv_last = GraphConvolution(
      in_features=hidden_node,
      out_features=out_node,
      
    )

    self.pool_linear = torch.nn.Linear(hidden_node * 2 + out_node, out_node)

  def forward(self, embedding_tensor, pool_x_tensor, adj, embedding_mask):

    pooling_tensor = self.gcn_forward(
      pool_x_tensor, adj,
      self.pool_conv_first, self.pool_conv_block, self.pool_conv_last,
      embedding_mask
    )
    pooling_tensor = F.softmax(self.pool_linear(pooling_tensor), dim=-1)
    if embedding_mask is not None:
      pooling_tensor = pooling_tensor * embedding_mask

    x_pool, adj_pool, _, _ = dense_diff_pool(embedding_tensor, adj, pooling_tensor)

    embedding_tensor = self.gcn_forward(
      x_pool, adj_pool,
      self.embed_conv_first, self.embed_conv_block, self.embed_conv_last,
    )

    output, _ = torch.max(embedding_tensor, dim=1)

    self.pool_tensor = pooling_tensor
    return output, adj_pool, x_pool, embedding_tensor

  def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):
    out_all = []

    layer_out_1 = F.relu(conv_first(x, adj))
    layer_out_1 = self.apply_bn(layer_out_1)
    out_all.append(layer_out_1)

    layer_out_2 = F.relu(conv_block(layer_out_1, adj))
    layer_out_2 = self.apply_bn(layer_out_2)
    out_all.append(layer_out_2)

    layer_out_3 = conv_last(layer_out_2, adj)
    out_all.append(layer_out_3)
    out_all = torch.cat(out_all, dim=2)
    if embedding_mask is not None:
      out_all = out_all * embedding_mask

    return out_all

  def apply_bn(self, x):
      ''' Batch normalization of 3D tensor x
      '''
      bn_module = torch.nn.BatchNorm1d(x.size()[1]).to(self._device)
      return bn_module(x)


class Reasoning(nn.Module):
    def __init__(self, input, output):
        super(GCN, self).__init__()
        self.bn0 = nn.BatchNorm1d(input, affine=False)
        self.conv1 = GraphConv(input, 512, MeanAggregator)
        self.conv2 = GraphConv(512, 256, MeanAggregator)
        self.conv3 = GraphConv(256, 128, MeanAggregator)
        self.conv4 = GraphConv(128, 64, MeanAggregator)
        in_node = input.shape[0] * input.shape[1]
        self.pool1 = GraphPool(64, 256, 64, in_node, 256, in_node// 4)
        self.pool2 = GraphPool(64, 256, 64, in_node//4, 256, in_node// 16)
 	self.pool3 = GraphPool(64, 256, 64, in_node//8, 256, 30)
  
        self.classifier = nn.Sequential(
            nn.Linear(64, output),
            nn.PReLU(output),
            nn.Linear(output, 2))

    def forward(self, x, A, one_hop_idcs, train=True):
        # data normalization l2 -> bn
        B, N, D = x.shape

        x = x.view(-1, D)
        x = self.bn0(x)
        x = x.view(B, N, D)

        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)

        x = self.pool1(x, A)
        x = self.pool2(x, A)
        x = self.pool3(x, A)
        k1 = one_hop_idcs.size(-1)
        dout = x.size(-1)
        edge_feat = torch.zeros(B, k1, dout).cuda()
        for b in range(B):
            edge_feat[b, :, :] = x[b, one_hop_idcs[b]]
        edge_feat = edge_feat.view(-1, dout)
        pred = self.classifier(edge_feat)

        # shape: (B*k1)x2
        return pred, x, edge_feat




"""The main interface"""
class GraphTextualReasoning(nn.Module):

    def __init__(self, is_training=True, max_len_labels, output):
        super(GraphTextualReasoning, self).__init__()
        self.k_at_hop = [8, 4]
        self.post_dim = 120
        self.active_connection = 3
        self.is_training = is_training
        self.max_len_labels = max_len_labels
        self.reasoning = Reasoning() 
        self.gcn_model = GraphConv(480,120)

        # ## graph generation branch
        if is_training:
            self.graph = GraphGeneration(self.k_at_hop, self.active_connection, self.pooling, 120, self.is_training)
        else:
            self.graph = GraphTest(self.pooling, 120)


    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict['model'])

    def forward(self, x,  to_device=None):
        # nB,nT,nC, nH, nW = x.size()
        graph_feat = x
        feat_batch, adj_batch, h1id_batch, gtmat_batch = self.graph(graph_feat)
        gcn_pred1, feat_batch1, adj_batch1, h1id_batch1 = self.reasoning(feat_batch, adj_batch, h1id_batch)

        return (gcn_pred1, to_device(gtmat_batch))

    def forward_test_graph(self, x):
        # nB,nT,nC, nH, nW = x.size()
        graph_feat = x

        flag, datas = self.graph(img, predict_out, graph_feat)
        feat, adj, cid, h1id, node_list, proposals, output = datas
        if flag:

            return None, None, None, output

        adj, cid, h1id = map(lambda x: x.cuda(), (adj, cid, h1id))
        gcn_pred = self.gcn_model(feat, adj, h1id)

        pred = F.softmax(gcn_pred, dim=1)

        edges = list()
        scores = list()
        node_list = node_list.long().squeeze().cpu().numpy()
        bs = feat.size(0)

        for b in range(bs):
            cidb = cid[b].int().item()
            nl = node_list[b]
            for j, n in enumerate(h1id[b]):
                n = n.item()
                edges.append([nl[cidb], nl[n]])
                scores.append(pred[b * (h1id.shape[1]) + j, 1].item())

        edges = np.asarray(edges)
        scores = np.asarray(scores)

        return edges, scores, proposals, output
