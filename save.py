import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'
import torch
from src.utils import get_llama, get_jamba, get_reorder_llama
from IPython import embed
from transformers import LlamaForCausalLM, JambaForCausalLM, AutoTokenizer


def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv

model_path = '/home/wanghz/GPTQ-triton/model/Meta-Llama-3.1-8B'
info_path = '/home/data/WeightReorder/order_info'
save_dir = 'reorder_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(save_dir)

model = get_llama(model_path)
model_dict = model.state_dict()

files = os.listdir(info_path)
for file in files:
    prefix = f'{info_path}/{file}'
    s = os.listdir(prefix)
    filenames = [f'{prefix}/{it}' for it in s]
    matrix = file[:-10]
    matrix = matrix.split('.')[-1]
    matrix = matrix.split('_')[0]
    p = file.split('.')
    matrix = f'{p[0]}.{p[1]}.{p[2]}.{p[3]}.{matrix}'
    for it in filenames:
        if 'txt' in it:
            name = it.split('/')[-1][:-4]
            shape = (int(name.split('_')[0]), int(name.split('_')[1]))
            key = f'{matrix}_block_size'
            value = torch.tensor(shape, dtype=torch.int16)
            
        elif 'col' in it:
            order_col = torch.load(it)
            key = f'{matrix}_order_col'
            value = order_col.type(torch.int16)

        elif 'row' in it:
            order_row = torch.load(it)
            key = f'{matrix}_order_row'
            value = order_row.type(torch.int64)
            value = inverse_permutation(value).type(torch.int16)

        value = value.to('cuda')
        print(key, value.shape)
        model_dict[key] = value
    
    key = f'{file[:-10]}.weight'
    print(key, model_dict[key].shape)
    model_dict[key] = model_dict[key][:, order_col.to(model_dict[key].device)]
    model_dict[key] = model_dict[key][order_row.to(model_dict[key].device), :]
    

# print(model_dict.keys())
model.save_pretrained(save_dir, state_dict=model_dict)
