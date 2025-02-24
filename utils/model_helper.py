import os
from collections import OrderedDict

import torch


def early_stopping(serendipity_list, stopping_steps):
    best_serendipity = max(serendipity_list)
    best_step = serendipity_list.index(best_serendipity)
    if len(serendipity_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_serendipity, should_stop


def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def helper_generate_batches(kg_data, batch_size=4096):
    batch = []
    current_length = 0
    head_to_data = {}

    for data in kg_data:
        head = data.split('[SEP]', 1)[0]
        if head not in head_to_data:
            head_to_data[head] = []
        head_to_data[head].append(data)


    for head, data_list in head_to_data.items():
        if current_length + len(data_list) <= batch_size:
            batch.extend(data_list)
            current_length += len(data_list)
        else:
            if batch:
                yield batch
                batch = []
                current_length = 0
   
            batch.extend(data_list)
            current_length = len(data_list)

    if batch:
        yield batch

def helper_generate_batches_numumoztanxcoedekfjjjfjfj(kg_data, batch_size):
    batch = []
    current_length = 0
    current_head = None

    for data in kg_data:
        # 각 데이터에서 첫번째 '[SEP]' 이전 부분을 head로 추출
        head = data.split('[SEP]', 1)[0]

        # 현재 헤드가 설정되지 않았거나 새로운 헤드가 시작되는 경우
        if current_head is None:
            raise RuntimeError("current_head is not set. Stopping execution.")

        if head != current_head or (current_length >= batch_size and head == current_head):
            # 새로운 헤드가 시작되거나 배치 크기를 넘어서면
            yield batch
            batch = []
            current_length = 0
            current_head = head  # 헤드를 새로운 헤드로 업데이트

        batch.append(data)
        current_length += 1  # 데이터를 배치에 추가하고 길이를 증가

    if batch:
        yield batch  # 남은 데이터가 있으면 마지막 배치 반환