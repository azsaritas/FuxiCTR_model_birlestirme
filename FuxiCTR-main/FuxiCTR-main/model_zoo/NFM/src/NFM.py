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


from torch import nn
import torch
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, LogisticRegression, InnerProductInteraction,HolographicInteraction,SqueezeExcitation, BilinearInteractionV2


class NFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="NFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 interaction_type="circular_convolution",
                 bilinear_type="field_all",
                 excitation_activation="ReLU",
                 reduction_ratio=2,
                 num_cross_layers=3,
                 embedding_dropout=0,
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(NFM, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs) 
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.lr_layer = LogisticRegression(feature_map, use_bias=False)        
        self.senet_layer = SqueezeExcitation(feature_map.num_fields, reduction_ratio, excitation_activation)

        self.bi_pooling_layer = InnerProductInteraction(feature_map.num_fields, output="bi_interaction")
        
        self.bilinear_interaction1 = BilinearInteractionV2(feature_map.num_fields, embedding_dim, bilinear_type)
        self.bilinear_interaction2 = BilinearInteractionV2(feature_map.num_fields, embedding_dim, bilinear_type)
        
        self.hfm_layer = HolographicInteraction(feature_map.num_fields, interaction_type=interaction_type)

     
        input_dim = feature_map.num_fields *feature_map.num_fields * embedding_dim #input değeri
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm) 
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
        
          
#NFM + FİBİNET MODELİ BİRLEŞTİRİLMİŞ   
    def forward(self, inputs):
      """
      Inputs: [X, y]
      """
      X = self.get_inputs(inputs)
       
      feature_emb = self.embedding_layer(X)
      bi_pooling_vec = self.bi_pooling_layer(feature_emb)
      bi_pooling_vec=bi_pooling_vec.unsqueeze(2).expand(-1, -1, 10)        

      senet_emb = self.senet_layer(feature_emb)
      
      bilinear_p = self.bilinear_interaction1(feature_emb)
      bilinear_q = self.bilinear_interaction2(senet_emb)
      # print(bi_pooling_vec.shape)
      # print(bilinear_p.shape)
      # print(bilinear_q.shape)
      
      comb_out = torch.cat([bilinear_p, bilinear_q,bi_pooling_vec], dim=1)
      comb_outx=torch.flatten((comb_out), start_dim=1)
      
      y_pred= self.dnn(comb_outx)+self.lr_layer(X)
      
      y_pred = self.output_activation(y_pred)
      return_dict = {"y_pred": y_pred}
      return return_dict  
        
#NFM modelinin aslı        
    # def forward(self, inputs):
    #     """
    #     Inputs: [X, y]
    #     """
    #     X = self.get_inputs(inputs)
    #     feature_emb = self.embedding_layer(X)
    #     bi_pooling_vec = self.bi_pooling_layer(feature_emb)
    #     y_pred = self.dnn(bi_pooling_vec)+self.lr_layer(X)
    #     y_pred = self.output_activation(y_pred)
    #     return_dict = {"y_pred": y_pred}
    #     return return_dict       
        
      
        
#ABLATION STUDY İÇİN KULLANILAN KODLAR       
    # def forward(self, inputs):
    #     """
    #     Inputs: [X, y]
    #     """
    #     X = self.get_inputs(inputs)
    #     #y_pred = self.lr_layer(X)
    #     feature_emb = self.embedding_layer(X)
    #     #print(feature_emb.shape)
    #     bi_pooling_vec = self.bi_pooling_layer(feature_emb)
    #     #print("bi pooling: ",bi_pooling_vec.shape)
    #     #y_pred = self.dnn(torch.flatten(feature_emb, start_dim=1))
    #     #print(feature_emb.shape)
    #     #y_pred = bi_pooling_vec.view(-1,1)
    #     y_pred = torch.mean(bi_pooling_vec, -1)
    #     y_pred=y_pred.unsqueeze(1).expand(-1, 1)        

    #     #print("y_pred: ",y_pred.shape)

    #     #y_pred = self.dnn(feature_emb)
    #     #+ self.lr_layer(X)
    #     y_pred+=self.lr_layer(X)
    #     y_pred = self.output_activation(y_pred)
    #     return_dict = {"y_pred": y_pred}
    #     return return_dict
    
    
    
    