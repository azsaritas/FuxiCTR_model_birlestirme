# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, LogisticRegression


class WideDeep(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="WideDeep", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(WideDeep, self).__init__(feature_map, 
                                       model_id=model_id, 
                                       gpu=gpu, 
                                       embedding_regularizer=embedding_regularizer, 
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.lr_layer = LogisticRegression(feature_map, use_bias=False)
        self.dnn = MLP_Block(input_dim=embedding_dim * feature_map.num_fields,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred = self.lr_layer(X)
        y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
