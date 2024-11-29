import os
import argparse
import time
import torch
from transformers import AutoTokenizer
from src.utils import get_llama, get_jamba, get_reorder_llama
from IPython import embed


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Path to model, either a HuggingFace model or a quantized model')
parser.add_argument('--quant', action='store_true', help='Whether the model is quantized')
parser.add_argument('--prompt', type=str, default='The quick brown fox', help='Prompt to use for generation')
parser.add_argument('--max-length', type=int, default=2048, help='Maximum length of generated text')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation')
parser.add_argument('--top-k', type=int, default=0, help='Top-k for generation')
parser.add_argument('--top-p', type=float, default=0.0, help='Top-p for generation')
parser.add_argument('--repetition-penalty', type=float, default=1.0, help='Repetition penalty for generation')


def main():
    args = parser.parse_args()

    # args.model = 'model/Meta-Llama-3.1-8B'
    args.model = 'reorder_model'

    model = None
    if 'Llama' in args.model:
        model = get_llama(args.model)
    elif 'reorder' in args.model:
        model = get_reorder_llama(args.model)
    elif 'Jamba' in args.model:
        model = get_jamba(args.model)
    else:
        raise ValueError('Not implemented model type!')
    
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    encoded_prompt = tokenizer.encode(args.prompt, add_special_tokens=False, return_tensors='pt').to(model.device)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    start_time = time.time()
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=args.max_length + len(encoded_prompt[0]),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=1,
    )
    end_time = time.time()

    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()
    
    total_tokens_generated = 0
    
    for generated_sequence in output_sequences:
        generated_sequence = generated_sequence.tolist()
        total_tokens_generated += len(generated_sequence) - len(encoded_prompt[0])

        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        total_sequence = (
            args.prompt + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
        )

        print(total_sequence)
    
    print()
    print(f'Generation took {end_time - start_time:.2f} seconds')
    print(f'Total tokens generated: {total_tokens_generated}')
    print(f'Average generation speed: {total_tokens_generated / (end_time - start_time):.2f} tokens per second')


if __name__ == '__main__':
    main()