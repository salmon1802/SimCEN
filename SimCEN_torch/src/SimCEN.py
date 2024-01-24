import torch
import torch.nn.functional as F
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.pytorch.torch_utils import get_activation

class SimCEN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="SimCEN",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 alpha=0.2,
                 cl_temperature=0.2,
                 hidden_units=[64],
                 ego_hidden_activations='relu',
                 v1_hidden_activations='relu',
                 v2_hidden_activations='relu',
                 ego_batch_norm=False,
                 v1_batch_norm=False,
                 v2_batch_norm=False,
                 through_dropout=0.1,
                 ego_dropout=0,
                 v1_dropout=0,
                 v2_dropout=0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(SimCEN, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=embedding_regularizer,
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.alpha = alpha
        self.cl_temperature = cl_temperature
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.embedding_dim = embedding_dim
        flatten_dim = feature_map.sum_emb_out_dim()
        num_fields = feature_map.num_fields

        self.mlep = MLEP(input_dim=flatten_dim * 3,
                         hidden_units=hidden_units,
                         ego_hidden_activations=ego_hidden_activations,
                         v1_hidden_activations=v1_hidden_activations,
                         v2_hidden_activations=v2_hidden_activations,
                         ego_dropout=ego_dropout,
                         v1_dropout=v1_dropout,
                         v2_dropout=v2_dropout,
                         ego_batch_norm=ego_batch_norm,
                         v1_batch_norm=v1_batch_norm,
                         v2_batch_norm=v2_batch_norm)

        self.segmentation = Segmentation(num_fields=num_fields,
                                         embedding_dim=embedding_dim,
                                         flatten_dim=flatten_dim)
        self.dropout = nn.Dropout(p=through_dropout)
        self.W = nn.Linear(hidden_units[-1], 1, bias=True)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        ego_feature_emb = self.embedding_layer(X)
        ego, view1, view2 = self.segmentation(ego_feature_emb)
        V = self.mlep(torch.cat([ego, view1, view2], dim=-1))
        ego, v1, v2 = torch.chunk(V, chunks=3, dim=-1)
        ego4v1v2 = self.dropout(ego)
        v1 = ego4v1v2 + v1
        v2 = ego4v1v2 + v2
        V = torch.cat([ego, v1, v2], dim=-1)
        y_pred = self.W(V)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred,
                       "ego": ego,
                       "v1": v1,
                       "v2": v2}
        return return_dict

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        ego = return_dict["ego"]
        v1 = return_dict["v1"]
        v2 = return_dict["v2"]
        cl_loss = self.InfoNCE(ego, v1, v2, cl_temperature=self.cl_temperature)
        loss = loss + self.alpha * cl_loss
        return loss

    def InfoNCE(self, ego, embedding_1, embedding_2, cl_temperature):
        ego = torch.nn.functional.normalize(ego)
        embedding_1 = torch.nn.functional.normalize(embedding_1)
        embedding_2 = torch.nn.functional.normalize(embedding_2)

        pos_score_e_1 = (ego * embedding_1).sum(dim=-1)
        pos_score_e_2 = (ego * embedding_2).sum(dim=-1)
        pos_score = (pos_score_e_1 + pos_score_e_2) * 0.5

        pos_score = torch.exp(pos_score / cl_temperature)

        ttl_score = torch.matmul(embedding_1, embedding_2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / cl_temperature).sum(dim=-1)
        loss = - torch.log(pos_score / ttl_score + 10e-6)
        return torch.mean(loss)


class Segmentation(nn.Module):
    def __init__(self, num_fields, embedding_dim, flatten_dim):
        super(Segmentation, self).__init__()
        self.triu_mask = nn.Parameter(torch.triu(torch.ones(num_fields, num_fields), 0).bool(),
                                      requires_grad=False)
        self.tril_mask = nn.Parameter(torch.tril(torch.ones(num_fields, num_fields), 0).bool(),
                                      requires_grad=False)
        self.kp_dim = int(num_fields * (num_fields + 1) / 2)
        self.kp_W = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim), requires_grad=True)
        nn.init.xavier_normal_(self.kp_W)
        self.project_triu = nn.Linear(self.kp_dim, flatten_dim, bias=False)
        self.project_tril = nn.Linear(self.kp_dim, flatten_dim, bias=False)

    def forward(self, feature_emb):
        embs_kp = torch.matmul(torch.matmul(feature_emb, self.kp_W), feature_emb.transpose(1, 2))
        triu = torch.masked_select(embs_kp, self.triu_mask).view(-1, self.kp_dim)
        tril = torch.masked_select(embs_kp, self.tril_mask).view(-1, self.kp_dim)
        embs_flatten = feature_emb.flatten(start_dim=1)
        view_1 = self.project_triu(triu)
        view_2 = self.project_tril(tril)
        return embs_flatten, view_1, view_2


class MLEP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_units=[],
                 ego_hidden_activations=None,
                 v1_hidden_activations=None,
                 v2_hidden_activations=None,
                 ego_dropout=0.0,
                 v1_dropout=0.0,
                 v2_dropout=0.0,
                 ego_batch_norm=False,
                 v1_batch_norm=False,
                 v2_batch_norm=False):
        super(MLEP, self).__init__()
        if type(ego_dropout) != list:
            ego_dropout = [ego_dropout] * len(hidden_units)
        if type(v1_dropout) != list:
            v1_dropout = [v1_dropout] * len(hidden_units)
        if type(v2_dropout) != list:
            v2_dropout = [v2_dropout] * len(hidden_units)
        if type(ego_hidden_activations) != list:
            ego_hidden_activations = [ego_hidden_activations] * len(hidden_units)
        if type(v1_hidden_activations) != list:
            v1_hidden_activations = [v1_hidden_activations] * len(hidden_units)
        if type(v2_hidden_activations) != list:
            v2_hidden_activations = [v2_hidden_activations] * len(hidden_units)
        self.layer = nn.ModuleList()
        self.norm_ego = nn.ModuleList()
        self.norm_view_1 = nn.ModuleList()
        self.norm_view_2 = nn.ModuleList()
        self.dropout_ego = nn.ModuleList()
        self.dropout_view_1 = nn.ModuleList()
        self.dropout_view_2 = nn.ModuleList()
        self.activation_ego = nn.ModuleList()
        self.activation_view_1 = nn.ModuleList()
        self.activation_view_2 = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            one_third = int(hidden_units[idx + 1] / 3)
            self.layer.append(Linear_unit(hidden_units[idx], hidden_units[idx + 1]))
            if ego_batch_norm:
                self.norm_ego.append(nn.BatchNorm1d(one_third))
            if v1_batch_norm:
                self.norm_view_1.append(nn.BatchNorm1d(one_third))
            if v2_batch_norm:
                self.norm_view_2.append(nn.BatchNorm1d(one_third))
            if ego_dropout[idx] > 0:
                self.dropout_ego.append(nn.Dropout(ego_dropout[idx]))
            if v1_dropout[idx] > 0:
                self.dropout_view_1.append(nn.Dropout(v1_dropout[idx]))
            if v2_dropout[idx] > 0:
                self.dropout_view_2.append(nn.Dropout(v2_dropout[idx]))
            self.activation_ego.append(get_activation(ego_hidden_activations[idx]))
            self.activation_view_1.append(get_activation(v1_hidden_activations[idx]))
            self.activation_view_2.append(get_activation(v2_hidden_activations[idx]))


    def forward(self, X):
        V_i = X
        for i in range(len(self.layer)):
            if i == 0:
                ego, v1, v2 = self.layer[i].forward1(V_i)
            else:
                ego, v1, v2 = self.layer[i].forward2(V_i)
            if len(self.norm_ego) > i:
                ego = self.norm_ego[i](ego)
            if len(self.norm_view_1) > i:
                v1 = self.norm_view_1[i](v1)
            if len(self.norm_view_2) > i:
                v2 = self.norm_view_2[i](v2)
            if self.activation_ego[i] is not None:
                ego = self.activation_ego[i](ego)
            if self.activation_view_1[i] is not None:
                v1 = self.activation_view_1[i](v1)
            if self.activation_view_2[i] is not None:
                v2 = self.activation_view_2[i](v2)
            if len(self.dropout_ego) > i:
                ego = self.dropout_ego[i](ego)
            if len(self.dropout_view_1) > i:
                v1 = self.dropout_view_1[i](v1)
            if len(self.dropout_view_2) > i:
                v2 = self.dropout_view_2[i](v2)
            V_i = torch.cat([ego, v1, v2], dim=-1)
        return V_i


class Linear_unit(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(Linear_unit, self).__init__()
        assert output_dim % 3 == 0, "output_dim should be divisible by 3."
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        one_third = int(output_dim / 3)
        self.gate = nn.Sequential(nn.Linear(one_third, one_third, bias=True),
                                  nn.Sigmoid())
        self.gate_tau = nn.Parameter(torch.ones(one_third), requires_grad=True)
        self.noise = nn.Parameter(torch.empty(2 * one_third), requires_grad=True)
        nn.init.uniform_(self.noise.data)

    def forward1(self, V):
        V = self.linear(V)
        ego, v1, v2 = torch.chunk(V, chunks=3, dim=-1)
        ego_gate = self.gate(ego) / self.gate_tau
        v1 = ego_gate * v1 + v1
        v2 = ego_gate * v2 + v2
        return ego, v1, v2

    def forward2(self, V):
        ego_V = V
        _, ego_v1, ego_v2 = torch.chunk(ego_V, chunks=3, dim=-1)
        V = self.linear(V)
        ego, v1, v2 = torch.chunk(V, chunks=3, dim=-1)
        ego_gate = self.gate(ego) / self.gate_tau
        noise_v1, noise_v2 = torch.chunk(self.noise, chunks=2, dim=-1)
        v1 = (ego_gate * v1 + v1 + noise_v1) + ego_v1
        v2 = (ego_gate * v2 + v2 + noise_v2) + ego_v2
        return ego, v1, v2

