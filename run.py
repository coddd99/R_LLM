
import os
import random
import argparse
import ast
import pickle
from collections import defaultdict
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from box import Box
import deepspeed
from deepspeed.runtime.lr_schedules import WarmupLR
from transformers import AutoTokenizer, BertModel, BertTokenizer
from datasets import Dataset

from utils.loader_recsysllm import *
from utils.metrics3 import *
from utils.model_helper import *
from model import BertKG


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def sample_neg_triples_for_h(kg_dict, kg_rt, head, relation, n_sample_neg_triples):
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


def generate_kg_batch(kg_dict, kg_rt, batch_size, mode):
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

        neg_tail = sample_neg_triples_for_h(kg_dict, kg_rt, h, relation[0], 1)
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
    
    pos_relation = tokenizer(pos_relation, truncation=True, padding='max_length', max_length=64)
    neg_relation = tokenizer(neg_relation, truncation=True, padding='max_length', max_length=64)
    
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


def evaluate(model, dataloader, Ks, device):
    """
    추천 모델의 성능을 다양한 메트릭(K 값에 따른 precision, ndcg, serendipity)으로 평가

    Args:
        model (torch.nn.Module): 평가 추천 모델
        dataloader (DataLoader): 학습 및 테스트 데이터 정보를 포함한 데이터로더
        Ks (list of int): 평가할 K 값 목록

    Returns:
        metrics_dict (dict): K 별 precision, ndcg, serendipity의 평균 메트릭 값
        group_metrics (dict): 그룹별 메트릭 값
    """
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict
    n_items = dataloader.n_items

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(batch).to(device) for batch in user_ids_batches]


    item_ids = torch.arange(n_items, dtype=torch.int32, device=device)
    all_cf_scores = []
    metric_names = ['precision', 'ndcg', 'serendipity', 'serendipity_2']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}
    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            with torch.no_grad():  
                batch_scores = model(batch_user_ids, item_ids, mode='predict')
                all_cf_scores.append(batch_scores)
                batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids, item_ids, Ks)
                for k in Ks:
                    for m in metric_names:
                        metrics_dict[k][m].append(batch_metrics[k][m].item())
            pbar.update(1)

    all_cf_scores = torch.cat(all_cf_scores, dim=0)

    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.mean(metrics_dict[k][m])

    group_metrics = calculate_group_metrics(train_user_dict, test_user_dict, user_ids, item_ids, model, Ks)

    return metrics_dict, group_metrics






def train(model, kgbert_optimizer, cf_optimizer, config, **kwargs):
    kg_dict = kwargs.get('kg_dict')
    kg_rt = kwargs.get('kg_rt')
    kg_num_batches = config.kg_batch_size  
    cf_num_batches = config.cf_batch_size  
    loss_storage = {}
    model.train()
    
    Ks = ast.literal_eval(config.Ks)
    metrics_list = {k: {'precision': [], 'ndcg': [], 'serendipity': [], 'serendipity_2': []} for k in Ks}

    ds_config_kg = {
        "train_batch_size": kg_num_batches,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True}, 
        "zero_optimization": {
            "stage": 2 
        }
    }
    
    model, kgbert_optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=kgbert_optimizer,
        config=ds_config_kg
    )

    kg_scheduler = WarmupLR(kgbert_optimizer, warmup_min_lr=1e-5, warmup_max_lr=1e-3, warmup_num_steps=1000)
    
    for epoch in range(1):
        print(f'\nKG-BERT training..... Epoch {epoch+1}')
        kgbert_total_loss = 0.0
        num_batches = len(knowledge) // kg_num_batches
        for i in tqdm(range(num_batches)):
            pos_id, neg_id, pos_att, neg_att = generate_kg_batch(kg_dict, kg_rt, batch_size=kg_num_batches, mode='train_kgbert_v2')
            loss = model(pos_id, neg_id, pos_att, neg_att, mode='train_kgbert_v2')
            model.backward(loss)
            model.step()
            kgbert_total_loss += loss.item()
        
        avg_loss = kgbert_total_loss / num_batches
        print(f"Epoch {epoch+1}, KG-BERT Loss: {avg_loss:.4f}")
        kg_scheduler.step()
    

    with torch.no_grad():
        print('Updating item embeddings...')
        model.module.update_item_emb()
        print('Updating user embeddings...')
        model.module.update_user_emb()


    ds_config_cf = {
            "train_batch_size": cf_num_batches,
            "gradient_accumulation_steps": 1,
            "bf16": {"enabled": True}, 
            "zero_optimization": {
                "stage": 2  
        }
    }
    
    model, cf_optimizer, _, _ = deepspeed.initialize(
        model=model.module,
        optimizer=cf_optimizer,
        config=ds_config_cf
    )


    cf_scheduler = WarmupLR(cf_optimizer, warmup_min_lr=3e-4, warmup_max_lr=3e-3, warmup_num_steps=100)


    for epoch in range(config.n_epoch):
        model.train()
        cf_total_loss = 0.0
        num_batches_cf = data.n_cf_train // cf_num_batches
        
        for i in tqdm(range(num_batches_cf)):
            user_id, pos_id, neg_id = data.generate_cf_batch(data.train_user_dict, cf_num_batches)
            user_id = user_id.to(device)
            pos_id = pos_id.to(device)
            neg_id = neg_id.to(device)
            cf_loss = model(user_id, pos_id, neg_id, mode='train_cf')
            model.backward(cf_loss)
            model.step()
            cf_total_loss += cf_loss.item()
        
        avg_cf_loss = cf_total_loss / num_batches_cf
        print(f"Epoch {epoch+1}, CF Loss: {avg_cf_loss:.4f}")
        
        cf_scheduler.step()
        loss_storage[epoch] = [avg_loss, avg_cf_loss]
        

        if (epoch % config.evaluate_every) == 0 or (epoch+1) == config.n_epoch:
            metrics_dict, group_dict = evaluate(model.module, data, Ks, device)
            for k in Ks:
                print(f"K={k}: Precision [{metrics_dict[k]['precision']:.4f}], "
                        f"NDCG [{metrics_dict[k]['ndcg']:.4f}], "
                        f"Serendipity [{metrics_dict[k]['serendipity']:.4f}], "
                        f"Serendipity_2 [{metrics_dict[k]['serendipity_2']:.4f}]")
                for m in ['precision', 'ndcg', 'serendipity', 'serendipity_2']:
                    metrics_list[k][m].append(metrics_dict[k][m])
    
    save_result = {
        'loss_storage': loss_storage,
        'metrics_list': metrics_list,
        'group_dict': group_dict,
        'model_state_dict': model.module.state_dict(),  
        'optimizer_state_dict': cf_optimizer.state_dict(),
    }
    
    os.makedirs(config.save_dir, exist_ok=True)
    save_path = os.path.join(config.save_dir, 'trained_model.pth')
    torch.save(save_result, save_path)
    print(f"Model and training results saved at {save_path}")

    return save_result



def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('--data_name', type=str, default='LastFM')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--use_pretrain', type=int, default=0)

    parser.add_argument('--cf_batch_size', type=int, default=2048)
    parser.add_argument('--kg_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=8912)
    parser.add_argument('--laplacian_type', type=str, default='symmetric')
    parser.add_argument('--aggregation_type', type=str, default='gcn')
    parser.add_argument('--conv_dim_list', type=str, default='[128, 128, 64]')
    parser.add_argument('--mess_dropout', type=str, default='[0.1, 0.1, 0.1]')
    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5)
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5)
    
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--stopping_steps', type=int, default=20)
    parser.add_argument('--cf_print_every', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--evaluate_every', type=int, default=5)
    parser.add_argument('--Ks', type=str, default='[10, 20]')
    parser.add_argument('--save_dir', type=str, default='./models')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config_dict = vars(args)
    config = Box(config_dict)

    with open(os.path.join(config.data_dir, config.data_name, 'itemid_remap.pkl'), 'rb')as f:
        item_dict = pickle.load(f)

    with open(os.path.join(config.data_dir, config.data_name, 'bert_kg_item.txt') , 'r') as f:
        knowledge = f.read().splitlines()

    kg_dict = defaultdict(list)
    kg_rt = defaultdict(list)

    for triple in knowledge:
        h, r, t = triple.split('[SEP]')
        h = h.strip()
        r = r.strip()
        t = t.strip()
        kg_dict[h].append((r, t))
        if t not in kg_rt[r]:
            kg_rt[r].append(t)
    
    
    data = DataLoaderRecsysLLM(config)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    

    lr, dropout = config.lr, config.dropout
    model = BertKG(data.n_users, data.n_items, data.A_in, "bert-base-uncased", tokenizer, kg_dict, kg_rt, item_dict, data, config)
    model.to(dtype=torch.bfloat16, device=device)
    loss_ = torch.nn.CrossEntropyLoss()
    cf_optimizer = optim.AdamW(model.parameters(), lr = lr)
    kg_optimizer = optim.AdamW(model.parameters(), lr = 5e-5)
    

    result = train(model = model, kgbert_optimizer=kg_optimizer, cf_optimizer = cf_optimizer, config=config, kg_dict=kg_dict, kg_rt=kg_rt)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'result_data_{timestamp}.pkl'
    save_path = os.path.join(config.save_dir, file_name)


    with open(save_path, 'wb') as f:
        pickle.dump(result, f)
