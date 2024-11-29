# python eval.py --model model/Meta-Llama-3.1-8B --tasks coqa --batch_size 10 --output_path results
# python eval.py --model model/Meta-Llama-3.1-8B --tasks truthfulqa_gen --batch_size 10 --output_path results
# python eval.py --model model/Meta-Llama-3.1-8B --tasks gsm8k --batch_size 10 --output_path results
python eval.py --model model/Meta-Llama-3.1-8B --tasks lambada --batch_size 10 --output_path results
python eval.py --model model/Meta-Llama-3.1-8B --tasks arc_challenge --batch_size 10 --output_path results
python eval.py --model model/Meta-Llama-3.1-8B --tasks hellaswag --batch_size 10 --output_path results
python eval.py --model model/Meta-Llama-3.1-8B --tasks mmlu --batch_size 10 --output_path results
python eval.py --model model/Meta-Llama-3.1-8B --tasks wikitext --batch_size 10 --output_path results
python eval.py --model model/Meta-Llama-3.1-8B --tasks paloma --batch_size 10 --output_path results
python eval.py --model model/Meta-Llama-3.1-8B --tasks piqa --batch_size 10 --output_path results
python eval.py --model model/Meta-Llama-3.1-8B --tasks truthfulqa --batch_size 10 --output_path results

# task_list = ['lambada', 'arc_challenge', 'hellaswag', 'mmlu', 'wikitext', 'gsm8k', 'truthfulqa', 'paloma', 'piqa']
