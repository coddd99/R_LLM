import os
import json
import ast
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from box import Box
from datasets import Dataset
from transformers import AutoTokenizer, BertModel, BertTokenizer

import utils.loader_base
from utils.loader_kgat import *
from utils.model_helper import *


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
            self.linear = nn.Linear(self.in_dim, self.out_dim).to(device='cuda', dtype=torch.bfloat16)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim).to(device='cuda', dtype=torch.bfloat16)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)      
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)      
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        elif self.aggregator_type == 'ngcf':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        else:
            raise NotImplementedError
        
    def forward(self, ego_embeddings, A_in, A_in_plusI):
        """
        ego_embeddings:  (n_users + n_entities, in_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """
        side_embeddings = torch.sparse.mm(A_in, ego_embeddings.float()).bfloat16()
        if self.aggregator_type == 'gcn':
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'graphsage':
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings


        elif self.aggregator_type == 'ngcf':
            side_L_plus_I_embeddings = torch.sparse.mm(A_in_plusI, ego_embeddings.float()).bfloat16()
            simple_embeddings = self.linear1(side_L_plus_I_embeddings)
            interaction_embeddings = self.linear2(torch.mul(side_embeddings, ego_embeddings))
            embeddings = F.leaky_relu(simple_embeddings + interaction_embeddings)

        embeddings = self.message_dropout(embeddings)           # (n_users + n_entities, out_dim)
        return embeddings



class BertKG(nn.Module):
    def __init__(self, n_users, n_items, graph, model_name, tokenizer, kg_dict, kg_rt, item_dict, data, config):
        super(BertKG, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = tokenizer
        self.bert = BertModel.from_pretrained(model_name)
        self.item_dict = item_dict
        self.kg_dict = kg_dict
        self.item_embed_dim = self.bert.config.hidden_size 
        self.user_embed_dim = self.item_embed_dim
        self.cls_mlp = nn.Linear(self.bert.config.hidden_size, 2)
        self.data = data
        self.rel = list(kg_rt.keys())[0]
        nn.init.xavier_uniform_(self.cls_mlp.weight)
        self.cf_l2loss_lambda = 0.1
        self.n_items = n_items
        self.n_users = n_users
        

        self.A_in_plusI = graph + sp.eye(graph.shape[0])
        self.A_in_plusI = self._convert_sp_mat_to_sp_tensor(self.A_in_plusI)
        self.A_in = self._convert_sp_mat_to_sp_tensor(graph)

        self.item_embed = nn.Embedding(self.n_items, self.item_embed_dim)
        self.user_embed = nn.Embedding(self.n_users, self.user_embed_dim)
        nn.init.xavier_uniform_(self.item_embed.weight)
        nn.init.xavier_uniform_(self.user_embed.weight)

        self.aggregator_layers = nn.ModuleList()
        
        self.aggregation_type = 'ngcf'
        
        self.conv_dim_list = [self.user_embed_dim] + ast.literal_eval(config.conv_dim_list)
        self.n_layers = len(self.conv_dim_list)-1
        self.mess_dropout = [config.dropout] * self.n_layers
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))


        self.user_bert = nn.Embedding(self.n_users, self.bert.config.hidden_size)
        self.item_bert = nn.Embedding(self.n_items, self.bert.config.hidden_size)
        self.user_bert.weight.requires_grad = False 
        self.item_bert.weight.requires_grad = False 
        
        self.att_temp_dim = 128
        self.query_proj = nn.Linear(self.bert.config.hidden_size, self.att_temp_dim)
        self.key_proj = nn.Linear(self.bert.config.hidden_size, self.att_temp_dim)
        self.value_proj = nn.Linear(self.bert.config.hidden_size, self.att_temp_dim)
        self.out_proj = nn.Linear(self.att_temp_dim, 64)
        
        self.att_dropout = nn.Dropout(0.1)
        self.s_query_proj = nn.Linear(self.bert.config.hidden_size, self.att_temp_dim)
        self.s_key_proj = nn.Linear(self.bert.config.hidden_size, self.att_temp_dim)
        self.s_value_proj = nn.Linear(self.bert.config.hidden_size, self.att_temp_dim)
        self.s_out_proj = nn.Linear(self.att_temp_dim, 64)

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
        pos_cls = self.aggregate_bert_representation(pos_input_ids, attention_mask = pos_attention_mask, mode='kg_train')
        neg_cls = self.aggregate_bert_representation(neg_input_ids, attention_mask = neg_attention_mask, mode='kg_train')
        
        pos_score = F.leaky_relu(self.cls_mlp(pos_cls))  
        neg_score = F.leaky_relu(self.cls_mlp(neg_cls))  

        pos_labels = torch.ones(pos_score.size(0), dtype=torch.long).to(pos_score.device) 
        neg_labels = torch.zeros(neg_score.size(0), dtype=torch.long).to(neg_score.device)  

        pos_loss = F.cross_entropy(pos_score, pos_labels)  
        neg_loss = F.cross_entropy(neg_score, neg_labels)
  
        total_loss = pos_loss + neg_loss
        return total_loss

    def calc_cf_embeddings(self):
    
        user_item_emb = torch.cat([self.item_embed.weight, self.user_embed.weight], dim=0)
        ego_embed = user_item_emb
        all_embed = [ego_embed]

        for idx, layer in enumerate(self.aggregator_layers):

            ego_embed = layer(ego_embed, self.A_in, self.A_in_plusI) 
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        all_embed = torch.cat(all_embed, dim=1)      
        return all_embed


    def train_cf(self, user_ids, item_pos_ids, item_neg_ids):
        all_embed = self.calc_cf_embeddings()
        user_embed = all_embed[user_ids]
        item_pos_embed = all_embed[item_pos_ids]
        item_neg_embed = all_embed[item_neg_ids]
        
        all_bert_emb = torch.cat((self.item_bert.weight, self.user_bert.weight), dim=0)
        user_bert_emb = all_bert_emb[user_ids]
        pos_item_emb = all_bert_emb[item_pos_ids]
        neg_item_emb = all_bert_emb[item_neg_ids]
        
        u_attn_output = self.apply_self_attention(user_bert_emb)
        p_attn_output = self.apply_attention(user_bert_emb, pos_item_emb)
        n_attn_output = self.apply_attention(user_bert_emb, neg_item_emb)

        u_emb_combined = torch.cat([user_embed, u_attn_output], dim=1) # Concatenate along the embedding dimension
        p_emb_combined = torch.cat([item_pos_embed, p_attn_output], dim=1)
        n_emb_combined = torch.cat([item_neg_embed, n_attn_output], dim=1)

        pos_score = torch.sum(torch.mul(u_emb_combined, p_emb_combined), dim=1)
        neg_score = torch.sum(torch.mul(u_emb_combined, n_emb_combined), dim=1)

        cf_loss = (-1.) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        return cf_loss


    def calc_score(self, user_ids, item_ids):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        all_embed = self.calc_cf_embeddings()         
        
        all_bert_emb = torch.cat((self.item_bert.weight, self.user_bert.weight), dim=0)
        u_attn_output = self.apply_self_attention(all_bert_emb[user_ids])
        i_attn_output = self.apply_attention(all_bert_emb[user_ids], all_bert_emb[item_ids])


        user_embed = all_embed[user_ids]       
        item_embed = all_embed[item_ids]      
        user_embed = torch.cat((user_embed, u_attn_output), dim=1)
        item_embed = torch.cat((item_embed, i_attn_output), dim=1)

        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))   
        return cf_score
        
    
    def aggregate_bert_representation(self, input_ids, attention_mask=None, tokenizer=None, mode=None):
        sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')

        if mode == 'kg_train':
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state[:, 0, :] # (batch_size, hideen_size)
            return sequence_output
            
        elif mode == 'update_item_embed':
            with torch.no_grad():
                outputs = self.bert(input_ids, attention_mask=attention_mask)
                sequence_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, seq_len(#token), hidden_size) 
                
                return sequence_output
    
    def update_user_emb(self):
        self.bert.eval()

        reverse_item_dict = {value: key for key, value in self.item_dict.items()}  # tensorid : name

        sentence_freq = defaultdict(int)  
        user_sentence_map = defaultdict(list) 
        user = {key: list(set(value)) for key, value in self.data.train_user_dict.items()}

        for user_id, interacted_items in user.items():
            genre_count = defaultdict(int) 

            for item in interacted_items:
                item_key = reverse_item_dict.get(item)
                if not item_key or item_key not in self.kg_dict:
                    continue

                for knowledge_triple in self.kg_dict[item_key]:
                    if knowledge_triple[0] == self.rel:
                        genre = knowledge_triple[1]

                        if genre_count[genre] >= 5:
                            continue

                        sentence = f"{item_key}[SEP]{knowledge_triple[0]}[SEP]{genre}"
                        sentence_freq[sentence] += 1 
                        user_sentence_map[user_id - self.data.n_items].append(sentence)  # user - sentence mapping
                        genre_count[genre] += 1


        unique_sentences = list(sentence_freq.keys())
        sentence_to_index = {s: idx for idx, s in enumerate(unique_sentences)}

        dataset = Dataset.from_dict({'text': unique_sentences})
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

        tokenized_dataset = dataset.map(
            lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=64),
            batched=True,
            num_proc=4,
            remove_columns=["text"]
        )

        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

 
        batch_size = 8192 * 2  
        num_batches = len(tokenized_dataset) // batch_size + int(len(tokenized_dataset) % batch_size > 0)
        embeddings = torch.zeros((len(tokenized_dataset), self.bert.config.hidden_size), dtype=torch.bfloat16).to(self.device)

        with torch.no_grad():
            for i in range(num_batches):
                print(f'{i+1}/{num_batches} Batch...')
                start = i * batch_size
                end = min(start + batch_size, len(tokenized_dataset))
                batch = tokenized_dataset[start:end]
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    batch_embeddings = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
                    embeddings[start:end] = batch_embeddings


                del input_ids, attention_mask, batch_embeddings
                torch.cuda.empty_cache()

 
        del tokenized_dataset, dataset, tokenizer
        torch.cuda.empty_cache()

        all_indices = []
        all_weights = []
        all_user_ids = []

        for user_id, sentences in user_sentence_map.items():
            weights = [sentence_freq[s] for s in sentences]
            indices = [sentence_to_index[s] for s in sentences]
            all_indices.extend(indices)
            all_weights.extend(weights)
            all_user_ids.extend([user_id] * len(sentences))

        del sentence_freq, user_sentence_map, sentence_to_index, unique_sentences
        torch.cuda.empty_cache()


        total_data = len(all_indices)


        batch_size = 1000000 

        user_embeddings = torch.zeros((self.n_users, embeddings.size(1)), dtype=torch.bfloat16).to(self.device)
        weights_sum = torch.zeros(self.n_users, dtype=torch.bfloat16).to(self.device)

        num_batches = total_data // batch_size + int(total_data % batch_size > 0)
        for i in range(num_batches):
            print(f'{i+1}/{num_batches} Batch...')

            start = i * batch_size
            end = min(start + batch_size, total_data)

            indices_tensor = torch.tensor(all_indices[start:end], dtype=torch.long).to(self.device)
            weights_tensor = torch.tensor(all_weights[start:end], dtype=torch.bfloat16).to(self.device)
            user_ids_tensor = torch.tensor(all_user_ids[start:end], dtype=torch.long).to(self.device)

            user_specific_embeddings = embeddings.index_select(0, indices_tensor)
            weighted_embeddings = user_specific_embeddings * weights_tensor.unsqueeze(1)

            user_embeddings.index_add_(0, user_ids_tensor, weighted_embeddings)
            weights_sum.index_add_(0, user_ids_tensor, weights_tensor)

            del indices_tensor, weights_tensor, user_ids_tensor, user_specific_embeddings, weighted_embeddings
            torch.cuda.empty_cache()


        del embeddings
        torch.cuda.empty_cache()

        user_embeddings = user_embeddings / weights_sum.unsqueeze(1)

        del weights_sum, all_indices, all_weights, all_user_ids
        torch.cuda.empty_cache()

        return user_embeddings

    def update_item_emb(self, batch_size=8192*2):
        self.bert.eval()
        self.bert.to(torch.bfloat16)
        itemid_dict = self.item_dict 

        item_pos_ids = list(itemid_dict.values())  # item ID list (int)
        item_heads = list(itemid_dict.keys())      # head list (str)

        kg_data = [
            f"{head}[SEP]{r}[SEP]{t}"
            for head in self.kg_dict.keys()
            for r, t in self.kg_dict[head]
        ]

        head_sum = defaultdict(lambda: None)
        head_count = defaultdict(int)
        output_size = self.bert.config.hidden_size

        with tqdm(total=(len(kg_data) + batch_size - 1) // batch_size, desc='Processing batches') as pbar:
            for batch_kg_data in helper_generate_batches(kg_data, batch_size):
                relation = self.tokenizer(
                    batch_kg_data, truncation=True, padding='max_length', max_length=64
                )
                relation_input_ids = torch.tensor(relation['input_ids'], dtype=torch.long).to(self.device)
                attention_mask = torch.tensor(relation['attention_mask'], dtype=torch.long).to(self.device)

                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        cls_output = self.aggregate_bert_representation(
                            relation_input_ids, attention_mask, mode='update_item_embed'
                        )
                cls_output_cpu = cls_output.detach().cpu()


                for data, output in zip(batch_kg_data, cls_output_cpu):
                    head_str = data.split('[SEP]', 1)[0]  # head : string

                    # head_str -> item_id
                    if head_str in itemid_dict:
                        item_id = itemid_dict[head_str]
                        if head_sum[item_id] is None:
                            head_sum[item_id] = output.clone()
                        else:
                            head_sum[item_id] += output
                        head_count[item_id] += 1
                    else:
                        print(f"Warning: head '{head_str}' not found in itemid_dict.")

                del relation_input_ids, attention_mask, cls_output, cls_output_cpu
                torch.cuda.empty_cache()

                pbar.update(1)

        all_averaged_embeddings = []

        for item_id in item_pos_ids:
            if item_id in head_sum and head_count[item_id] > 0:
                averaged_cls_output = head_sum[item_id] / head_count[item_id]
                all_averaged_embeddings.append(averaged_cls_output)
                del head_sum[item_id]
            else:
                print(f"Warning: No CLS outputs found for item_id: {item_id}")
                all_averaged_embeddings.append(torch.empty(output_size).uniform_(-0.1, 0.1))


        concatenated_tensor = torch.stack(all_averaged_embeddings).to(self.device, dtype=torch.bfloat16)
        if concatenated_tensor.size() == self.item_bert.weight.size():
            with torch.no_grad():
                self.item_bert.weight.copy_(concatenated_tensor)
        else:
            raise ValueError("Concatenated tensor size does not match item_embed_bert weight size.")


        del all_averaged_embeddings, concatenated_tensor, head_sum, head_count
        torch.cuda.empty_cache()
 
    def apply_self_attention(self, user_embedding):
        if user_embedding.dim() == 1:
            user_embedding = user_embedding.unsqueeze(0)  # (1, embed_dim)

        Q = self.s_query_proj(user_embedding)  # (batch_size, embed_dim)
        K = self.s_key_proj(user_embedding)    # (batch_size, embed_dim)
        V = self.s_value_proj(user_embedding)  # (batch_size, embed_dim)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.att_temp_dim ** 0.5)  # (batch_size, 1)
        attn_scores = torch.clamp(attn_scores, min=-10, max=10) 

        attn_weights = F.softmax(attn_scores + 1e-9, dim=-1)
        attn_weights = self.att_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (batch_size, embed_dim)


        output = self.s_out_proj(F.leaky_relu(attn_output))  # (batch_size, embed_dim)

        if output.size(0) == 1:
            output = output.squeeze(0)  # (embed_dim,)

        return output

    def apply_attention(self, user_text_emb, item_text_emb):
        if user_text_emb.size(-1) != item_text_emb.size(-1):
            raise ValueError(f"Embedding sizes do not match: {user_text_emb.size(-1)} vs {item_text_emb.size(-1)}")

        Q = self.query_proj(item_text_emb)  # (num_items, embed_dim)
        K = self.key_proj(user_text_emb)    # (num_users, embed_dim)
        V = self.value_proj(user_text_emb)  # (num_users, embed_dim)

        if Q.size(0) > 1 and K.size(0) > 1:
            Q = Q.unsqueeze(1)  # (num_items, 1, embed_dim)
            K = K.unsqueeze(0)  # (1, num_users, embed_dim)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.att_temp_dim ** 0.5)  # (num_items, num_users)
        attn_scores = torch.clamp(attn_scores, min=-10, max=10) 

        attn_weights = F.softmax(attn_scores + 1e-9, dim=-1)
        attn_weights = self.att_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (num_items, embed_dim)
        output = self.out_proj(F.leaky_relu(attn_output))  # (num_items, 256)

        if item_text_emb.size(0) == 1:  # num_items == 1
            output = output.squeeze(0)  # (256,)

        output = output.squeeze(1)
        return output

        
    def forward(self, *input, mode):
        if mode == 'train_kgbert_v2':
            return self.train_kgbert_v2(*input)
        if mode == 'train_cf':
            return self.train_cf(*input)
        if mode == 'predict':
            return self.calc_score(*input)
