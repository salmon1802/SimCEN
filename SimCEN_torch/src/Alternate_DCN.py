import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, get_activation
import os


class Emb_DCN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="Emb_DCN",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(Emb_DCN, self).__init__(feature_map,
                                  model_id=model_id,
                                  gpu=gpu,
                                  embedding_regularizer=embedding_regularizer,
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        self.EmbDCN = CN_MLP_Block(input_dim=input_dim,
                                num_cross_layers=num_cross_layers,
                                hidden_activations=dnn_activations,
                                dropout_rates=net_dropout,
                                batch_norm=batch_norm)
        self.fc = nn.Linear(input_dim, 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, dynamic_emb_dim=True)
        final_out = self.EmbDCN(feature_emb)
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        torch.save(y_pred, os.path.abspath(os.path.join(self.model_dir, self.model_id + ".final_prediction")))
        torch.save(self.get_labels(inputs), os.path.abspath(os.path.join(self.model_dir, self.model_id + ".true_label")))
        return_dict = {"y_pred": y_pred}
        return return_dict


class CN_MLP_Block(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 hidden_activations="ReLU",
                 dropout_rates=0.0,
                 batch_norm=False):
        super(CN_MLP_Block, self).__init__()
        hidden_units = [input_dim for _ in range(num_cross_layers)]
        self.cross_net = nn.ModuleList(CrossInteraction(input_dim)
                                       for _ in range(num_cross_layers))
        if type(dropout_rates) != list:
            dropout_rates = [dropout_rates] * len(hidden_units)
        if type(hidden_activations) != list:
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.layer = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.activation = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            self.layer.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1]))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                self.dropout.append(nn.Dropout(dropout_rates[idx]))
            self.activation.append(get_activation(hidden_activations[idx]))

    def forward(self, X):
        X_i = X
        for i in range(len(self.layer)):
            X_i = self.layer[i](X_i)
            X_i = self.cross_net[i](X, X_i) + X_i
            if len(self.norm) > i:
                X_i = self.norm[i](X_i)
            if self.activation[i] is not None:
                X_i = self.activation[i](X_i)
            if len(self.dropout) > i:
                X_i = self.dropout[i](X_i)
        return X_i


class CrossInteraction(nn.Module):
    def __init__(self, input_dim):
        super(CrossInteraction, self).__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interact_out = self.weight(X_i) * X_0 + self.bias
        return interact_out