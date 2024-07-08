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
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, LogisticRegression, HolographicInteraction
from itertools import combinations


class HFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="HFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 interaction_type="circular_convolution",
                 bilinear_type="field_all",
                 use_dnn=True,
                 hidden_units=[64, 64],
                 hidden_activations=["relu", "relu"],
                 batch_norm=False,
                 net_dropout=0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(HFM, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.lr_layer = LogisticRegression(feature_map)
        self.hfm_layer = HolographicInteraction(feature_map.num_fields, interaction_type=interaction_type)

        self.use_dnn = True
        if self.use_dnn:
            #input_dim = int(feature_map.num_fields * (feature_map.num_fields - 1) / 2) * embedding_dim
            input_dim=feature_map.num_fields * embedding_dim
            self.dnn = MLP_Block(input_dim=input_dim,
                                 output_dim=1, 
                                 hidden_units=hidden_units,
                                 hidden_activations=hidden_activations,
                                 output_activation=None,
                                 dropout_rates=net_dropout, 
                                 batch_norm=batch_norm)
        else:
            self.proj_h = nn.Linear(embedding_dim, 1, bias=False)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

            
    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        #interact_out = self.hfm_layer(feature_emb)
        no_interact_out=feature_emb
        print(feature_emb.shape)
        # print(interact_out.shape)
        # print(bilinear_out.shape)
        
        #print((torch.flatten(torch.cat([bilinear_out,interact_out], dim=1), start_dim=1)).shape)
        #self.dnn(feature_emb)
        # if self.use_dnn:
        #     hfm_out = self.dnn(torch.flatten(no_interact_out, start_dim=1))
            #hfm_out = self.dnn(torch.flatten(torch.cat([bilinear_out,interact_out], dim=1), start_dim=1)) #eklendi
            #print("dnn calisti")
        # else:
        #     hfm_out = self.proj_h(interact_out.sum(dim=1))
        #     #print("dnn calismadi")
            
        #sadece dnn
        #hfm_out = self.dnn(torch.flatten(no_interact_out, start_dim=1)) 0.9838 -- 0.9806
        
        #sadece hfm
        #hfm_out = self.proj_h(interact_out.sum(dim=1)) 0.9102
        
        y_pred = hfm_out 
        #+ self.lr_layer(X)  
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

