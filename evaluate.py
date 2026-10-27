
import pandas as pd
import fire
import torch
import json
import os
from transformers import GenerationConfig,  AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, LogitsProcessorList, TemperatureLogitsWarper
from data import  EvalD3Dataset, EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor
from accelerate import Accelerator
import random
import time



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
P = 998244353
MOD = int(1e9 + 9)
import numpy as np

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
def main(
    base_model: str = "",
    train_file: str = "",
    info_file: str = "",
    category: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    batch_size: int = 4,
    K: int = 0,
    seed: int = 42,
    length_penalty: float=0.0,
    max_new_tokens: int = 256,
    num_beams: int = 50,
    sample: int = -1,
    max_prompt_length: int = 512,
):
    if batch_size < 1 or num_beams < 2 or max_prompt_length < 4:
        raise ValueError("Require batch_size >= 1, num_beams >= 2, max_prompt_length >= 4")
    if sample == 0 or sample < -1:
        raise ValueError("sample must be -1 (all rows) or a positive integer")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    evaluation_start = time.perf_counter()
    random.seed(seed)
    set_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    category = category_dict[category]
    print(category)

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Parse new format: semantic_id \t item_title \t item_id
        semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
        item_titles = [line.split('\t')[1].strip() + "\n" for line in info if len(line.split('\t')) >= 2]
        
        # Format for tokenization
        info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]
        info_titles = [f'''### Response:\n{_}''' for _ in item_titles]


    tokenizer = AutoTokenizer.from_pretrained(base_model)
    embedding_rows = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embedding_rows:
        raise ValueError(f"Tokenizer/model vocabulary mismatch: {len(tokenizer)} vs {embedding_rows}")
    if model.config.tie_word_embeddings:
        if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
            raise ValueError("Expected tied input/output weights but loaded model does not share them")
    print(f"Model reload passed: vocab={embedding_rows}, tied_embeddings={model.config.tie_word_embeddings}")
    
    # Create prefixID for semantic IDs (existing functionality)
    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids[1:] for _ in info_titles]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids for _ in info_titles]
    if base_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3

    response_prefix = tokenizer.encode("### Response:\n", add_special_tokens=False)
    if len(response_prefix) != prefix_index:
        raise ValueError("Upstream constrained decoder prefix length does not match this tokenizer")
    
    # Build hash_dict for semantic IDs (existing functionality)
    hash_dict = dict()
    # print(f"eos token: {tokenizer.eos_token_id}")
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])

    # Build hash_dict_title for item titles (new functionality)
    hash_dict_title = dict()
    for index, ID in enumerate(prefixTitleID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict_title:
                hash_dict_title[hash_number] = set()
            hash_dict_title[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])

    # Convert sets to lists for both dictionaries
    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    for key in hash_dict_title.keys():
        hash_dict_title[key] = list(hash_dict_title[key])

    # Define prefix constraint functions
    def prefix_allowed_tokens_fn_semantic(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []
        
    def prefix_allowed_tokens_fn_title(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict_title:
            return hash_dict_title[hash_number]
        return []

    # Default to semantic constraints (backward compatibility)
    prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_semantic
    # prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_title
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # val_dataset = EvalD3Dataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=K, seed=seed)
    val_dataset = EvalSidDataset(train_file=test_data_path, tokenizer=tokenizer, max_len=max_prompt_length, category=category, test=True, K=K, seed=seed, sample=sample)
        
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    truncated_prompts = sum(len(encoding["input_ids"]) > max_prompt_length for encoding in encodings)
    for encoding in encodings:
        encoding["input_ids"] = encoding["input_ids"][-max_prompt_length:]
        encoding["attention_mask"] = encoding["attention_mask"][-max_prompt_length:]
        if encoding["input_ids"][-prefix_index:] != response_prefix:
            raise ValueError("Evaluation prompt does not end with expected response prefix")
    # encodings = [val_dataset[i] for i in indexes]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    def evaluate(
            encodings,
            num_beams=10,
            max_new_tokens=64,
            length_penalty=1.0,
            **kwargs,
    ):
        maxLen = max([len(_["input_ids"]) for _ in encodings])

        padding_encodings = {"input_ids": []}
        attention_mask = []

        for  _ in encodings:
            L = len(_["input_ids"])
            padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            attention_mask.append([0] * (maxLen - L) + [1] * L) 
        
        # print(f"num_beams: {num_beams}")
        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id = model.config.pad_token_id,
            eos_token_id = model.config.eos_token_id,
            max_new_tokens = max_new_tokens,
            do_sample=False,
            top_k=None,
            top_p=None,
            **kwargs
        )
        
        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                base_model=base_model,
                eos_token_id=model.config.eos_token_id
            )
            logits_processor = LogitsProcessorList([clp])

            generation_output = model.generate(
                torch.tensor(padding_encodings["input_ids"]).to(device),
                attention_mask=torch.tensor(attention_mask).to(device),
                generation_config=generation_config,
                # Passed as a generate() kwarg on purpose. Since transformers
                # 4.50, values in `generation_config` that equal the *global*
                # default are overwritten by the model's own generation_config
                # -- so `do_sample=False` set on the GenerationConfig above is
                # silently replaced by `do_sample=true` for checkpoints that
                # ship it (e.g. Qwen2.5-*-Instruct), turning constrained beam
                # search into sampling and producing invalid SIDs. Only kwargs
                # take priority over the model defaults.
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )
       
        batched_completions = generation_output.sequences[:, maxLen:]
       
        
        if base_model.lower().find("llama") > -1:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
            
        output = [_.split("Response:\n")[-1].strip() for _ in output]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs
    
    model = model.to(device)

    from tqdm import tqdm
    outputs = []
    new_encodings = []
    BLOCK = (len(encodings) + batch_size - 1) // batch_size
    for i in range(BLOCK):
        new_encodings.append(encodings[i * batch_size: (i + 1) * batch_size])

    
    for idx, encodings in enumerate(tqdm(new_encodings)):
        # Use standard evaluation
        output = evaluate(encodings, max_new_tokens=max_new_tokens, num_beams=num_beams, length_penalty=length_penalty)
        
        outputs = outputs + output
       
    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
        row = val_dataset.data.iloc[i]
        test["source_row_index"] = int(val_dataset.data.index[i])
        test["user_id"] = str(row["user_id"])
        test["target_item_id"] = str(row["item_id"])
  

    for i in range(len(test_data)):
        if 'dedup' in test_data[i]:
            test_data[i].pop('dedup')  
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)

    run_metrics = {
        "model": base_model,
        "data": test_data_path,
        "queries": len(test_data),
        "sample": sample,
        "seed": seed,
        "batch_size": batch_size,
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "length_penalty": length_penalty,
        "max_prompt_length": max_prompt_length,
        "truncated_prompts": truncated_prompts,
        "do_sample": False,
        "tokenizer_vocab": len(tokenizer),
        "tie_word_embeddings": model.config.tie_word_embeddings,
        "total_evaluation_seconds": time.perf_counter() - evaluation_start,
    }
    if torch.cuda.is_available():
        run_metrics.update({
            "gpu_name": torch.cuda.get_device_name(),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        })
    with open(result_json_data + ".run.json", "w") as handle:
        json.dump(run_metrics, handle, indent=2)

if __name__ == '__main__':
    fire.Fire(main)





