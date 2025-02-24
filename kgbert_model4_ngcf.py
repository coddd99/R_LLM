import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import re
import loader_base
from loader_kgat import *
from box import Box
import pickle

from transformers import BertModel, BertTokenizer
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader


from metrics import *
from model_helper import *
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time

kgat_config_dict = {
        'seed': 2019,
        'data_name': 'ml_1M_preprocessed',
        'data_dir': '/root/vol1/recsys_justdata',
        'use_pretrain': 0,
        'pretrain_embedding_dir': 'datasets/pretrain/',
        'pretrain_model_path': 'trained_model/model.pth',
        'cf_batch_size': 8192,
        'kg_batch_size': 8192,
        'test_batch_size': 10000,
        'embed_dim': 64,
        'relation_dim': 64,
        'laplacian_type': 'symmetric',
        'aggregation_type': 'gcn',
        'conv_dim_list': '[64, 32, 16]',
        'mess_dropout': '[0.1, 0.1, 0.1]',
        'kg_l2loss_lambda': 1e-5,
        'cf_l2loss_lambda': 1e-5,
        'lr': 0.0001,
        'n_epoch': 1,
        'stopping_steps': 20,
        'cf_print_every': 10,
        'kg_print_every': 10,
        'evaluate_every': 10,
        'Ks': '[10, 20]',
        'save_dir': '/root/vol1/grad_thesis_final/kgbert_model_ngcf2'
    }

kgat_config = Box(kgat_config_dict)





# movie_id_remap2 변수를 피클 파일에서 불러오기

with open('/root/vol1/grad_thesis_final/movie_id_remap.pkl', 'rb') as f:
    item_dict = pickle.load(f)
item_dict = {value : key for key, value in item_dict.items()}



h_list = []
t_list = []
r_list = []

kg_dict = defaultdict(list)
kg_rt = defaultdict(list)

with open('/root/vol1/grad_thesis_final/bert_kg_item.txt', 'r') as f:
    knowledge = f.read().splitlines()


for dataa in knowledge:
    h, r, t = dataa.split('[SEP]')
    h = h.strip()
    r = r.strip()
    t = t.strip()
    
    kg_dict[h].append((r, t))
    if t not in kg_rt[r]:
        kg_rt[r].append(t)


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        

        if self.aggregator_type == 'gcn':
            # 모듈 생성과 동시에 CUDA 장치와 bfloat16 데이터 타입으로 설정
            self.linear = nn.Linear(self.in_dim, self.out_dim).to(device='cuda', dtype=torch.bfloat16)
            # Xavier 초기화를 적용
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            # 모듈을 생성하고 바로 CUDA와 bfloat16으로 설정
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim).to(device='cuda', dtype=torch.bfloat16)
            # Xavier 초기화 적용
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)      
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)      
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        elif self.aggregator_type == 'ngcf':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)
            self.bias1 = nn.Parameter(torch.empty(1, self.out_dim))  # nn.Embedding 대신 nn.Parameter 사용
            self.bias2 = nn.Parameter(torch.empty(1, self.out_dim))  
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
            nn.init.xavier_uniform_(self.bias1)  # nn.Embedding이 아닌, 직접 nn.Parameter를 초기화
            nn.init.xavier_uniform_(self.bias2)

        else:
            raise NotImplementedError
        
    def forward(self, ego_embeddings, A_in, A_in_plusI):
        """
        ego_embeddings:  (n_users + n_entities, in_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """
        # Equation (3)
        #A_in = A_in.to('cpu').to(torch.float32)
        #ego_embeddings = ego_embeddings.to('cpu').to(torch.float32)
        #side_embeddings = torch.matmul(A_in, ego_embeddings).to('cuda', dtype=torch.bfloat16)
        
        # ego_embeddings도 동일하게 CUDA로 이동시키면서 데이터 타입을 bfloat16으로 변경
        #ego_embeddings = ego_embeddings.to('cuda', dtype=torch.bfloat16)
        #side_embeddings = torch.sparse.mm(A_in, ego_embeddings)
        #side_embeddings = torch.sparse.mm(A_in, ego_embeddings.float())
        side_embeddings = torch.sparse.mm(A_in, ego_embeddings.float()).bfloat16()
        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings


        elif self.aggregator_type == 'ngcf':
            side_L_plus_I_embeddings = torch.sparse.mm(A_in_plusI, ego_embeddings.float()).bfloat16()
            simple_embeddings = self.linear1(side_L_plus_I_embeddings) + self.bias1
            

            #side_embeddings = torch.sparse.mm(A_in, ego_embeddings.float()).bfloat16()
            interaction_embeddings = self.linear2(torch.mul(side_embeddings, ego_embeddings)) + self.bias2

            embeddings = F.leaky_relu(simple_embeddings + interaction_embeddings)


        #embeddings = self.message_dropout(embeddings)           # (n_users + n_entities, out_dim)
        return embeddings



class BertKG(nn.Module):
    def __init__(self, n_users, n_items, graph, model_name, tokenizer, kg_dict, item_dict):
        super(BertKG, self).__init__()
        # Bert LLM parameters
        self.device = 'cuda'
        self.tokenizer = tokenizer
        self.bert = BertModel.from_pretrained(model_name)
        self.kg_outputs = nn.Linear(self.bert.config.hidden_size, 2)
        self.kg_outputs_ht = nn.Linear(self.bert.config.hidden_size, 256) #128은 최종 아이템 임베딩 사이즈임
        self.kg_outputs_r = nn.Linear(self.bert.config.hidden_size, 256) #128은 최종 아이템 임베딩 사이즈임
        nn.init.xavier_uniform_(self.kg_outputs.weight)
        nn.init.xavier_uniform_(self.kg_outputs_ht.weight)
        nn.init.xavier_uniform_(self.kg_outputs_r.weight)
        self.item_dict = item_dict
        self.item_embed_dim = 256
        self.user_embed_dim = 2 * self.item_embed_dim

        #self.templayer = nn.Linear(1536, 128)
        #nn.init.xavier_uniform_(self.templayer.weight)

      
        self.cf_l2loss_lambda = 0.1
        self.n_items = n_items
        self.n_users = n_users
        

        self.A_in_plusI = graph + sp.eye(graph.shape[0])
        self.A_in_plusI = self._convert_sp_mat_to_sp_tensor(self.A_in_plusI)
        self.A_in = self._convert_sp_mat_to_sp_tensor(graph)

        #self.A_in = self._convert_sp_mat_to_sp_tensor(graph)
        #self.A_in = graph
        #self.A_in = self.A_in.to('cuda')
        #self.A_in = self.A_in.to(torch.float16)
        self.item_embed = nn.Embedding(self.n_items, self.item_embed_dim)
        self.user_embed = nn.Embedding(self.n_users, self.user_embed_dim)
        nn.init.xavier_uniform_(self.item_embed.weight)
        nn.init.xavier_uniform_(self.user_embed.weight)
        #print(f'first. user_embed.shpae {self.user_embed.weight.shape}')
        #print(f'first.  item_embed.shpae {self.item_embed.weight.shape}')
        
        self.aggregator_layers = nn.ModuleList()
        
        self.aggregation_type = 'ngcf'
        self.n_layers = 4
        self.conv_dim_list = [self.user_embed_dim, 128, 128, 128, 64]
        self.mess_dropout = [0.05, 0.1, 0.1, 0.1]
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))

        # 데이터 포인트 수와 임베딩 차원을 기반으로 임시 헤드 및 테일 파라미터 초기화
        self.temp_head = nn.Parameter(torch.randn(self.n_items, self.bert.config.hidden_size), requires_grad=False)
        self.temp_tail = nn.Parameter(torch.randn(self.n_items, self.bert.config.hidden_size), requires_grad=False)
        # 이 파라미터들의 그래디언트를 비활성화
        #self.temp_head.requires_grad = False
        #self.temp_tail.requires_grad = False
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.asmatrix([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        res = torch.sparse.FloatTensor(i, v, coo.shape).to(self.device)
        return res

    def train_kgbert_v1(self, input_ids, attention_mask):
        outputs = self.bert(input_ids,attention_mask = attention_mask)
        cls_output = outputs.last_hidden_state[:,0,:]
        logits = self.kg_outputs(cls_output)
        logits = logits.squeeze(-1)
        porbs = torch.softmax(logits, dim= -1)
        return porbs


    def train_kgbert_v2(self, pos_input_ids, neg_input_ids, pos_attention_mask, neg_attention_mask):
        pos_h_batch, pos_r_batch, pos_t_batch = self.aggregate_bert_representation(pos_input_ids, attention_mask = pos_attention_mask)
        neg_h_batch, neg_r_batch, neg_t_batch = self.aggregate_bert_representation(neg_input_ids, attention_mask = neg_attention_mask)
        pos_h_final = self.kg_outputs_ht(pos_h_batch)
        pos_t_final = self.kg_outputs_ht(pos_t_batch)
        pos_r_final = self.kg_outputs_r(pos_r_batch)

        neg_h_final = self.kg_outputs_ht(neg_h_batch)
        neg_t_final = self.kg_outputs_ht(neg_t_batch)
        neg_r_final = self.kg_outputs_r(neg_r_batch)
        negative_distance = torch.norm(neg_h_final + neg_r_final - neg_t_final, p=2, dim=1)
        positive_distance = torch.norm(pos_h_final + pos_r_final - pos_t_final, p=2, dim=1)

        margin = 0.05
        loss = F.relu(margin + positive_distance - negative_distance).mean()

        return loss

    def calc_cf_embeddings(self):
        #print(f'user_embed.shpae {self.user_embed.weight.shape}')
        #print(f'item_embed.shpae {self.item_embed.weight.shape}')
        user_item_emb = torch.cat([self.user_embed.weight, self.item_embed.weight], dim=0)
        #user_item_emb = torch.cat([self.user_embed, self.item_embed], dim = 0)
        ego_embed = user_item_emb
        all_embed = [ego_embed]
        
        #print(f'ego_embed.shape {ego_embed.shape}')
        #print(f'all_embed length {len(all_embed)}')
        for idx, layer in enumerate(self.aggregator_layers):
            #print(f'idx {idx}')
            ego_embed = layer(ego_embed, self.A_in, self.A_in_plusI) #처음 돌면 ego_embed에 (1)순환한게 저장되어 있다.
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Equation (11)
        all_embed = torch.cat(all_embed, dim=1)         # (n_users + n_entities, concat_dim)
        #print(f'all_embed size : {all_embed.shape}')
        return all_embed


    def train_cf(self, user_ids, item_pos_ids, item_neg_ids):
        combined_vector = torch.cat((self.kg_outputs_ht(self.temp_head), self.kg_outputs_ht(self.temp_tail)), dim=1)
        #print(combined_vector.shape)
        self.item_embed.weight.data = combined_vector

        all_embed = self.calc_cf_embeddings()
        user_embed = all_embed[user_ids]
        item_pos_embed = all_embed[item_pos_ids]
        item_neg_embed = all_embed[item_neg_ids]

        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)

        cf_loss = (-1.) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        return cf_loss

        #l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        #loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        # return loss


    def calc_score(self, user_ids, item_ids):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        all_embed = self.calc_cf_embeddings()           # (n_users + n_entities, concat_dim)
        user_embed = all_embed[user_ids]                # (n_users, concat_dim)
        item_embed = all_embed[item_ids]                # (n_items, concat_dim)

        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))    # (n_users, n_items)
        return cf_score
        

    """
    def get_item_embedding(self, user_ids, item_pos_ids, item_neg_ids):
        #item_dict : item_id와 실제 head(영화)를 연결해 주는 자료

        pos_heads = [self.item_dict[item_pos_id] for item_pos_id in item_pos_ids]
        neg_heads = [self.item_dict[item_neg_id] for item_neg_id in item_neg_ids]

        pos_kg_data = [pos_head + '[SEP]' + r + '[SEP]' + t for pos_head in pos_heads for r, t in kg_dict[pos_head]]
        neg_kg_data = [neg_head + '[SEP]' + r + '[SEP]' + t for neg_head in neg_heads for r, t in kg_dict[neg_head]]

        pos_relation = tokenizer(pos_kg_data, truncation=True, padding='max_length', max_length=256)
        neg_relation = tokenizer(neg_kg_data, truncation=True, padding='max_length', max_length=256)
        
        pos_relation_input_ids = torch.tensor(pos_relation['input_ids'], dtype=torch.long).to(device)
        neg_relation_input_ids = torch.tensor(neg_relation['input_ids'], dtype=torch.long).to(device)
        
        pos_attention_mask = torch.tensor(pos_relation['attention_mask'], dtype=torch.long).to(device)
        neg_attention_mask = torch.tensor(neg_relation['attention_mask'], dtype=torch.long).to(device)

        # Expected output : 
        # head relation tail triple vector each is mean
        # Each component of the head-relation-tail triple vector represents 
        # the average of the vectors for the head, relation, and tail respectively.
        pos_final, _, _ = aggregate_bert_representation(pos_relation_input_ids, pos_attention_mask) 
        neg_final, _, _ = aggregate_bert_representation(neg_relation_input_ids, neg_attention_mask)

        user_final = self.user_embed[user_embd]
    """
    
    def aggregate_bert_representation(self, input_ids, attention_mask=None, tokenizer=None, mode=None):
        """
        inputs :
        input_ids : 배치사이즈 * 토큰 패딩 길이
        attention_mask : 배치사이즈 * 어텐션 마스크 길이
        """
        if mode is None:
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size) 
        else:
            with torch.no_grad():
                outputs = self.bert(input_ids, attention_mask=attention_mask)
                sequence_output = outputs.last_hidden_state  # (batch_size, seq_len(token 개수), hidden_size) 
                #print(f"outputs.lasst_hidden_state {sequence_output.shape}")

        sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        if not mode:
            # 헤드, 관계, 테일 벡터들을 저장할 리스트
            all_h_vectors = []
            all_r_vectors = []
            all_t_vectors = []

            for batch_idx in range(input_ids.size(0)):  # 배치 크기만큼 반복
                input_ids_batch = input_ids[batch_idx]  # 배치 내 한 문장(토크나이제이션까지만 된 것)
                sequence_output_batch = sequence_output[batch_idx]  # 해당 문장의 BERT 출력(토크나이제이션 이후 출력값)

                # 해당 문장의 SEP 토큰 인덱스 찾기
                sep_indices = (input_ids_batch == sep_token_id).nonzero(as_tuple=True)[0].tolist()

                # SEP로 구분된 텍스트 구간 추출 (CLS 토큰을 제외하고 구간 설정)
                entity_spans = []
                prev_index = 1  # CLS 토큰 다음부터 시작 (1번 인덱스)

                for index in sep_indices:
                    if prev_index < index:  # 비어있는 구간을 방지
                        entity_spans.append((prev_index, index))
                    prev_index = index + 1

                # SEP로 나눠진 3개 구간만 처리 (헤드, 관계, 테일 순으로)
                if len(entity_spans) >= 3:
                    h_tokens = sequence_output_batch[entity_spans[0][0]:entity_spans[0][1]]
                    r_tokens = sequence_output_batch[entity_spans[1][0]:entity_spans[1][1]]
                    t_tokens = sequence_output_batch[entity_spans[2][0]:entity_spans[2][1]]
                    
                    h_vector = torch.mean(h_tokens, dim=0)
                    r_vector = torch.mean(r_tokens, dim=0)
                    t_vector = torch.mean(t_tokens, dim=0)

                    all_h_vectors.append(h_vector)
                    all_r_vectors.append(r_vector)
                    all_t_vectors.append(t_vector)

                    del h_tokens, r_tokens, t_tokens, h_vector, r_vector, t_vector
                    torch.cuda.empty_cache()
                else:
                    raise ValueError(f"Expected at least 3 SEP-separated spans for head, relation, and tail in batch {batch_idx}")

            # 각 배치의 헤드, 관계, 테일 벡터를 텐서로 변환
            h_batch = torch.stack(all_h_vectors)
            r_batch = torch.stack(all_r_vectors)
            t_batch = torch.stack(all_t_vectors)
            
            return h_batch, r_batch, t_batch
        else:
            # 헤드와 테일 벡터들을 저장할 딕셔너리
            # 각 헤드와 테일 벡터를 저장할 딕셔너리
            head_to_tail_vectors = defaultdict(list)
            head_vectors = {}  # 헤드 벡터 저장
            #tail_vectors = {}
            unique_heads = set()  # 헤드의 고유한 ID 저장

            for batch_idx in range(input_ids.size(0)):
                input_ids_batch = input_ids[batch_idx]
                sequence_output_batch = sequence_output[batch_idx]

                # SEP 토큰 인덱스 찾기
                sep_indices = (input_ids_batch == sep_token_id).nonzero(as_tuple=True)[0].tolist()

                entity_spans = []
                prev_index = 1  # CLS 토큰 다음부터 시작

                for index in sep_indices:
                    if prev_index < index:
                        entity_spans.append((prev_index, index))
                    prev_index = index + 1

                if len(entity_spans) >= 3:
                    # 헤드와 테일 벡터를 추출하여 평균을 계산
                    h_tokens = sequence_output_batch[entity_spans[0][0]:entity_spans[0][1]]
                    t_tokens = sequence_output_batch[entity_spans[2][0]:entity_spans[2][1]]

                    h_vector = torch.mean(h_tokens, dim=0)
                    t_vector = torch.mean(t_tokens, dim=0)

                    head_id = hash(tuple(input_ids_batch[entity_spans[0][0]:entity_spans[0][1]].tolist()))
                    head_to_tail_vectors[head_id].append(t_vector)
                    head_vectors[head_id] = h_vector
                    #tail_vectors[head_id] = t_vector
                    unique_heads.add(head_id)  # 헤드 ID를 세트에 추가

            # 각 헤드에 대한 최종 벡터 생성
            head_final_vectors = []
            tail_final_vectors = []

            for head_id in unique_heads:
                # 평균 테일 벡터 계산
                mean_tail_vector = torch.mean(torch.stack(head_to_tail_vectors[head_id]), dim=0).to(device) 

                # 해당 head_id의 head_vector 가져오기
                h_vector = head_vectors[head_id].to(device)
                
                # 두 벡터를 concat하여 1536차원으로 구성
                #combined_vector = torch.cat((h_vector, mean_tail_vector), dim=0)
                #head_final_vectors.append(combih_vecned_vector)
                head_final_vectors.append(h_vector)
                tail_final_vectors.append(mean_tail_vector)
                
            # 최종 벡터를 텐서로 변환하여 반환
            head_final_tensor = torch.stack(head_final_vectors)
            tail_final_tensor = torch.stack(tail_final_vectors)
            return head_final_tensor, tail_final_tensor

        
    def update_item_emb(self):
        #item_dict : item_id와 실제 head(영화)를 연결해 주는 자료
        item_pos_ids = list(self.item_dict.keys())
        #print(len(item_pos_ids))
        heads = [self.item_dict[item_pos_id] for item_pos_id in item_pos_ids]
        kg_data = [head + '[SEP]' + r + '[SEP]' + t for head in heads for r, t in kg_dict[head]]
        #print(f'len.kg_data : {len(kg_data)}')

        relation = self.tokenizer(kg_data, truncation=True, padding='max_length', max_length=128)
        #print('=====basic token for Toy story Head =====')
        #print(np.array(relation['input_ids'][:8]))
        relation_input_ids = torch.tensor(relation['input_ids'], dtype=torch.long).to(device)
        attention_mask = torch.tensor(relation['attention_mask'], dtype=torch.long).to(device)
        
        with torch.no_grad():
            head_final_tensor, tail_final_tensor = self.aggregate_bert_representation(relation_input_ids, attention_mask, mode = 'only_head') 
        
        #head_final = head_final.to(torch.bfloat16)
        
        #print(f'head_final_tensor.shape {head_final_tensor.shape}')
        #print(f'tail_final_tensor.shape {tail_final_tensor.shape}')
        #item_finish = self.templayer(head_final)
        self.temp_head.data = head_final_tensor
        self.temp_tail.data = tail_final_tensor

        #self.item_embed.weight.data = item_finish
            

    def forward(self, *input, mode):
        if mode == 'train_kgbert_v2':
            return self.train_kgbert_v2(*input)
        if mode == 'train_cf':
            #self.update_item_emb()
            return self.train_cf(*input)
        if mode == 'predict':
            return self.calc_score(*input)



kgat_config = Box(kgat_config_dict)
data = DataLoaderKGAT(kgat_config)
data.n_users


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertKG(data.n_users, data.n_items, data.A_in, "bert-base-uncased", tokenizer, kg_dict, item_dict)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(dtype=torch.bfloat16, device = device)

import random

def sample_pos_triples_for_h(kg_dict, head, n_sample_pos_triples=1):
    pos_triples = kg_dict[head]
    n_pos_triples = len(pos_triples)

    if n_pos_triples <n_sample_pos_triples:
         raise ValueError(f"Requested {n_sample_pos_triples} samples, but only {n_pos_triples} available.")

    sample_relations, sample_pos_tails = [], []
    
    while True:
        if len(sample_relations) == n_sample_pos_triples:
            break
        
        pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
        relation = pos_triples[pos_triple_idx][0]
        tail = pos_triples[pos_triple_idx][1]

        if tail not in sample_pos_tails:
            sample_relations.append(relation)
            sample_pos_tails.append(tail)
            
    return sample_relations, sample_pos_tails


# 이게 tail을 숫자로 맵핑하다 보니 쉽게 가능한데.... 그냥 영어로하려면 흠
# relation별 tail은 전부 다 해 놓고 set()-set()해서 샘플링을 할까?

def sample_neg_triples_for_h(kg_dict, head, relation, n_sample_neg_triples):
    pos_triples = kg_dict[head]
    neg_tail_candidate = list(set(kg_rt[relation])-{item[1] for item in kg_dict[head]})
    
    sample_neg_tails = []

    while True:
        if len(sample_neg_tails) == n_sample_neg_triples:
            break

        tail = random.choice(neg_tail_candidate)
        if (relation, tail) not in pos_triples and tail not in sample_neg_tails:
            sample_neg_tails.append(tail)
    return sample_neg_tails


def generate_kg_batch(kg_dict, batch_size, mode):
    """
    return : pos relation id, neg realation id, atten mask
    """
    exist_heads = list(kg_dict.keys()) # For DeprecationWarning
    if batch_size <= len(exist_heads):
        batch_head = random.sample(exist_heads, batch_size)
    else:
        batch_head = [random.choice(exist_heads) for _ in range(batch_size)]
    
    batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
    for h in batch_head:
        relation, pos_tail = sample_pos_triples_for_h(kg_dict, h, 1)
        batch_relation += relation
        batch_pos_tail += pos_tail

        neg_tail = sample_neg_triples_for_h(kg_dict, h, relation[0], 1)
        batch_neg_tail += neg_tail

    pos_relation = []
    neg_relation = []
    pos_labels = []
    neg_labels = []

    for h, r, pt, nt in zip(batch_head, batch_relation, batch_pos_tail, batch_neg_tail):
        pos_relation.append(f'{h}[SEP]{r}[SEP]{pt}')
        neg_relation.append(f'{h}[SEP]{r}[SEP]{nt}')
        pos_labels.append(1)
        neg_labels.append(0)
    
    pos_relation = tokenizer(pos_relation, truncation=True, padding='max_length', max_length=128)
    neg_relation = tokenizer(neg_relation, truncation=True, padding='max_length', max_length=128)
    
    pos_relation_input_ids = torch.tensor(pos_relation['input_ids'], dtype=torch.long).to(device)
    neg_relation_input_ids = torch.tensor(neg_relation['input_ids'], dtype=torch.long).to(device)
    
    pos_attention_mask = torch.tensor(pos_relation['attention_mask'], dtype=torch.long).to(device)
    neg_attention_mask = torch.tensor(neg_relation['attention_mask'], dtype=torch.long).to(device)

    pos_labels = torch.tensor(pos_labels, dtype=torch.long).to(device)
    neg_labels = torch.tensor(neg_labels, dtype=torch.long).to(device)

    if mode == 'train_kgbert':
        relation_input_ids = torch.cat([pos_relation_input_ids, neg_relation_input_ids], dim=0)
        attention_mask = torch.cat([pos_attention_mask, neg_attention_mask], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)        
        indices = torch.randperm(relation_input_ids.size(0))
        return relation_input_ids[indices], attention_mask[indices], labels[indices]
    
    if mode == 'train_kgbert_v2':
        return pos_relation_input_ids, neg_relation_input_ids, pos_attention_mask, neg_attention_mask

    if mode == 'train_cf':
        pass


loss_fn = torch.nn.CrossEntropyLoss()
kgbert_optimizer = optim.AdamW(model.parameters(), lr = 3e-5)
cf_optimizer = optim.AdamW(model.parameters(), lr = 1e-3)

def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg', 'serendipity']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}
    print(metrics_dict[10].keys())
    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, mode='predict')       # (n_batch_users, n_items)

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)
            batch_scores = batch_scores.float() 
            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict

    

import time

def train(optimizer, loss_fn):    
    print('NGCF version')
    print('model is 256 text embed, and 128 128 128 64 (4 layer)')
    print('kg_lr : 5e-5, cf_lr : 1e-3')
    num_batches = 41
    model.train()
    for epoch in range(200):
        
       
        kgbert_total_loss = 0.
        for i in tqdm(range(40)):
        #for i in tqdm(range(2)):
            # 배치 생성
            pos_id, neg_id, pos_att, neg_att = generate_kg_batch(kg_dict, batch_size=256, mode='train_kgbert_v2')
            
            loss = model(pos_id, neg_id, pos_att, neg_att, mode='train_kgbert_v2')

            
              # 이전 단계의 기울기 초기화
            
            loss.backward()  # 역전파를 통해 기울기 계산
            kgbert_optimizer.step()  # 기울기를 기반으로 파라미터 업데이트
            kgbert_optimizer.zero_grad()

            kgbert_total_loss += loss.item()  # loss 값을 누적


        avg_loss = kgbert_total_loss / num_batches
        print(f"Epoch {epoch+1}, Loss : {avg_loss}===")



        with torch.no_grad():
            model.update_item_emb()
            
        cf_total_loss = 0.
        for i in tqdm(range(183)):

            user_id, pos_id, neg_id = data.generate_cf_batch(data.train_user_dict, 4096)

            user_id = user_id.to(device)
            pos_id = pos_id.to(device)
            neg_id = neg_id.to(device)

            cf_batch_loss = model(user_id, pos_id, neg_id, mode='train_cf')

            

            cf_batch_loss.backward()

            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()
        
        print(f"Epoch {epoch+1}, cf loss: {cf_total_loss/num_batches}")

            

        
        Ks = [10, 15, 20]

        k_min = min(Ks)
        k_max = max(Ks)
        
        epoch_list = []
        metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': [], 'serendipity': []} for k in Ks}

        best_epoch = -1
        best_recall = 0

        
        #evaluate CF
        _, metrics_dict = evaluate(model, data, Ks, device)
        start_time = time.time()  # 시작 시간 기록
        print('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}], Serendipity [{:.4f}, {:.4f}]'.format(
            epoch, time.time() - start_time, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg'], metrics_dict[k_min]['serendipity'], metrics_dict[k_max]['serendipity']))

        epoch_list.append(epoch)
        for k in Ks:
            for m in ['precision', 'recall', 'ndcg', 'serendipity']:
                metrics_list[k][m].append(metrics_dict[k][m])
        best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], kgat_config.stopping_steps)

        if should_stop:
            break

        if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
            save_model(model, kgat_config.save_dir, epoch, best_epoch)
            print('Save model on epoch {:04d}!'.format(epoch))
            best_epoch = epoch



train(kgbert_optimizer, loss_fn)