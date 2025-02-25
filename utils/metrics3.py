import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from collections import defaultdict

def precision_at_k_batch(hits_k, k):
    return torch.mean(hits_k.sum(dim=1) / k)

def ndcg_at_k_batch(hits_k, k):
    discount = torch.log2(torch.arange(2, k + 2, device=hits_k.device))
    dcg = (hits_k / discount).sum(dim=1)
    idcg = torch.tensor([(1.0 / torch.log2(torch.arange(2, len(row) + 2, device=hits_k.device))).sum() 
                         for row in hits_k], device=hits_k.device)
    return torch.mean(dcg / idcg)

def serendipity_at_k_batch(rank_indices, train_user_dict, test_user_dict, user_ids, k):
    serendipity = []
    for idx, u in enumerate(user_ids):
        recommended_items = rank_indices[idx, :k]
        train_items = torch.tensor(train_user_dict[int(u)], device=rank_indices.device)
        test_items = torch.tensor(test_user_dict[int(u)], device=rank_indices.device)
        novelty_hits = torch.isin(recommended_items, test_items) & (~torch.isin(recommended_items, train_items))
        serendipity.append(novelty_hits.float().mean())
    return torch.tensor(serendipity, device=rank_indices.device).mean()



def serendipity_at_k_batch2(rank_indices, train_user_dict, test_user_dict, user_ids, k, device='cuda'):
    """
    Calculate Serendipity@k per user based on a simplified formula using bfloat16 on GPU.
    rank_indices: torch tensor of shape (n_users, n_items) on GPU (in bfloat16)
    train_user_dict: dictionary with train items for each user
    test_user_dict: dictionary with test items for each user
    user_ids: list of user IDs to evaluate
    k: top-k items to consider for Serendipity calculation
    """
    n_users = len(user_ids)
    n_total_items = rank_indices.shape[1]
    
    # 상위 k개의 추천 아이템 선택 (Shape: (n_users, k))
    recommended_items_k = rank_indices[:, :k]  # 상위 k개의 아이템 인덱스 선택
    
    # 아이템 순위 기반 확률 p(i) 계산
    item_positions = torch.arange(1, k + 1, device=device).to(torch.bfloat16)  # 순위 (1부터 k까지)
    p_i = ((n_total_items + 1 - item_positions) / n_total_items).to(torch.bfloat16)  # Shape: (k,)
    p_i = p_i.unsqueeze(0).expand(n_users, -1)  # 모든 사용자에 대해 확률 복사 (Shape: (n_users, k))
    
    # 테스트 아이템 집합 생성 및 GPU로 이동
    test_items_list = [torch.tensor(test_user_dict[int(u)], device=device, dtype=torch.long) for u in user_ids]
    
    # 관련성 점수 rel(i) 계산
    rel_i = torch.zeros((n_users, k), dtype=torch.bfloat16, device=device)  # Shape: (n_users, k)
    for idx in range(n_users):
        test_items = test_items_list[idx]
        rel_i[idx] = torch.isin(recommended_items_k[idx], test_items).to(torch.bfloat16)  # 추천된 아이템이 테스트 집합에 있는지 확인
    

    serendipity_sum = torch.sum(p_i * rel_i, dim=1)  # Shape: (n_users,)
    
    # Serendipity@k 계산
    serendipity_scores = serendipity_sum / k  # Shape: (n_users,)
    
    return serendipity_scores.mean()  # 전체 사용자에 대한 평균 Serendipity 점수 반환


def calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, Ks):
    for idx, u in enumerate(user_ids):
        train_pos_item_list = train_user_dict[int(u.item())]
        cf_scores[idx, train_pos_item_list] = -float('inf')

    max_K = max(Ks)
    _, rank_indices = torch.topk(cf_scores, k=max_K, dim=1)

    binary_hit = torch.zeros((len(user_ids), max_K), dtype=torch.bfloat16, device=cf_scores.device)
    for idx, u in enumerate(user_ids):
        test_pos_item_list = test_user_dict[int(u.item())]
        hits = torch.isin(rank_indices[idx], torch.tensor(test_pos_item_list, device=cf_scores.device)).float()
        binary_hit[idx, :len(hits)] = hits

    metrics_dict = {}
    for k in Ks:
        metrics_dict[k] = {}
        hits_k = binary_hit[:, :k]
        metrics_dict[k]['precision'] = precision_at_k_batch(hits_k, k)
        metrics_dict[k]['ndcg'] = ndcg_at_k_batch(hits_k, k)
        metrics_dict[k]['serendipity'] = serendipity_at_k_batch(rank_indices[:, :k], train_user_dict, test_user_dict, user_ids, k)
        metrics_dict[k]['serendipity_2'] = serendipity_at_k_batch2(rank_indices[:, :k], train_user_dict, test_user_dict, user_ids, k)

    del cf_scores, rank_indices, binary_hit
    torch.cuda.empty_cache()

    return metrics_dict



def calculate_group_metrics(train_user_dict, test_user_dict, user_ids_batches, item_ids, model, Ks):

    user_interactions = {u: len(items) for u, items in train_user_dict.items()}
    interaction_counts = np.array(list(user_interactions.values()))

    quantiles = np.percentile(interaction_counts, [25, 50, 75])
    print('quantiles : ')
    print(quantiles)
    user_groups = defaultdict(list)
    for user_id, count in user_interactions.items():
        if count <= quantiles[0]:
            user_groups['Q1'].append(user_id)
        elif count <= quantiles[1]:
            user_groups['Q2'].append(user_id)
        elif count <= quantiles[2]:
            user_groups['Q3'].append(user_id)
        else:
            user_groups['Q4'].append(user_id)


    all_metrics = {}
    for group, group_user_ids in user_groups.items():
        print(f"Processing group: {group}, Users: {len(group_user_ids)}")
        group_user_ids_batches = [
            torch.LongTensor(group_user_ids[i: i + len(user_ids_batches)]).to(item_ids.device)
            for i in range(0, len(group_user_ids), len(user_ids_batches))
        ]

        group_metrics = {k: {'precision': [], 'ndcg': []} for k in Ks}
        for batch_user_ids in group_user_ids_batches:
            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, mode='predict')
                batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids, item_ids, Ks)

               
                for k in Ks:
                    for metric in ['precision', 'ndcg']:
                        group_metrics[k][metric].append(batch_metrics[k][metric].item())

       
        for k in Ks:
            for metric in ['precision', 'ndcg']:
                group_metrics[k][metric] = np.mean(group_metrics[k][metric])

        all_metrics[group] = group_metrics
        for group, metrics in all_metrics.items():
            print(f"Group: {group}")
            print("=" * 30)
            for k, metric_values in metrics.items():
                precision = metric_values['precision']
                ndcg = metric_values['ndcg']
                print(f"K = {k}: Precision = {precision:.4f}, NDCG = {ndcg:.4f}")
            print("-" * 30)

    return all_metrics