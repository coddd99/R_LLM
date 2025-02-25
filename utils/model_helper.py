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