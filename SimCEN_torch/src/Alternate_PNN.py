from torch import nn
import torch, os
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, InnerProductInteraction, get_activation


class emb_PNN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="emb_PNN",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_layers=3,
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(emb_PNN, self).__init__(feature_map,
                                           model_id=model_id,
                                           gpu=gpu,
                                           embedding_regularizer=embedding_regularizer,
                                           net_regularizer=net_regularizer,
                                           **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        num_fields = feature_map.num_fields

        self.dnn = PNN_MLP_Block(input_dim=num_fields * embedding_dim,
                                 num_fields=num_fields,
                                 dnn_layers=dnn_layers,
                                 embedding_dim=embedding_dim,
                                 hidden_activations=hidden_activations,
                                 dropout_rates=net_dropout,
                                 batch_norm=batch_norm)
        self.fc = nn.Linear(num_fields * embedding_dim, 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        dnn_out = self.dnn(feature_emb.flatten(start_dim=1))
        y_pred = self.fc(dnn_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        torch.save(y_pred, os.path.abspath(os.path.join(self.model_dir, self.model_id + ".final_prediction")))
        torch.save(self.get_labels(inputs),
                   os.path.abspath(os.path.join(self.model_dir, self.model_id + ".true_label")))
        return return_dict


class PNN_MLP_Block(nn.Module):
    def __init__(self,
                 input_dim,
                 dnn_layers=3,
                 num_fields=24,
                 embedding_dim=16,
                 hidden_activations="ReLU",
                 dropout_rates=0.0,
                 batch_norm=False):
        super(PNN_MLP_Block, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        interaction_units = int(num_fields * (num_fields - 1) / 2)
        hidden_units = [input_dim for _ in range(dnn_layers)]
        self.inner_product_layer = InnerProductInteraction(num_fields, output="inner_product")
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
            self.layer.append(nn.Linear(hidden_units[idx] + interaction_units, hidden_units[idx + 1]))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                self.dropout.append(nn.Dropout(dropout_rates[idx]))
            self.activation.append(get_activation(hidden_activations[idx]))

    def forward(self, X):
        X_i = X
        for i in range(len(self.layer)):
            feature_emb_i = X_i.view(-1, self.num_fields, self.embedding_dim)
            inner_products = self.inner_product_layer(feature_emb_i)
            X_i = torch.cat([X_i, inner_products], dim=-1)
            X_i = self.layer[i](X_i)
            if len(self.norm) > i:
                X_i = self.norm[i](X_i)
            if self.activation[i] is not None:
                X_i = self.activation[i](X_i)
            if len(self.dropout) > i:
                X_i = self.dropout[i](X_i)
        return X_i
