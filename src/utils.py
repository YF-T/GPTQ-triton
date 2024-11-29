import torch
from transformers import LlamaForCausalLM, JambaForCausalLM
from accelerate import load_checkpoint_and_dispatch, dispatch_model, init_empty_weights

from .reorder_model import ReorderLlamaForCausalLM, ReorderLlamaDecoderLayer


def get_reorder_llama(model: str):
    """
    Load a pretrained Llama model
    """
    model_name = model

    with init_empty_weights():
        model = ReorderLlamaForCausalLM.from_pretrained(model_name, attn_implementation="eager")

    model = load_checkpoint_and_dispatch(model, checkpoint=model_name, device_map="auto", no_split_module_classes='ReorderLlamaDecoderLayer')

    return model


def get_llama(model: str):
    """
    Load a pretrained Llama model
    """
    def skip(*args, **kwargs):
        pass
    # NOTE: This is a nasty hack, but it speeds up model building by a huge amount
    old_inits = (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_)
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto', device_map='auto')
    model.seqlen = 2048

    # Restore the old initializers
    torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = old_inits

    return model


def get_jamba(model: str):
    """
    Load a pretrained Jamba model
    """
    # TODO
    def skip(*args, **kwargs):
        pass
    # NOTE: This is a nasty hack, but it speeds up model building by a huge amount
    old_inits = (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_)
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = JambaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map='auto', use_mamba_kernels=False)

    # Restore the old initializers
    torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = old_inits

    return model