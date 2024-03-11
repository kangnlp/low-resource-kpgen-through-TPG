import json
import pandas as pd
import random

from datasets import Dataset
from torch.utils.data import DataLoader
from tokenizers.processors import TemplateProcessing


def TPG_LRFT_DataLoader(stage, fname, tokenizer, batch_size, max_length, order="pres_abs", mode="train"):
    """
    Build Data Loader

    """

    dataset = Dataset.from_json(fname)

    if not tokenizer.cls_token:
        tokenizer.cls_token = tokenizer.bos_token
    if not tokenizer.sep_token:
        tokenizer.sep_token = tokenizer.eos_token

    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{tokenizer.cls_token} $0 {tokenizer.sep_token}",
        pair=f"{tokenizer.cls_token} $A {tokenizer.sep_token} $B:1 {tokenizer.sep_token}:1",
        special_tokens=[(tokenizer.cls_token, tokenizer.cls_token_id), (tokenizer.sep_token, tokenizer.sep_token_id)],
    )

    

    def preprocess_function(examples):
    
        processed = {}

        if stage == 'TPG' or stage == 'TG':

            masked_abstract = examples['abstract']
            input_text = f'{masked_abstract}'

        elif stage == 'LRFT':

            input_text = f'{examples["title"]["text"]} [sep] {examples["abstract"]["text"]}'

        
        tokenizer_input = tokenizer(
            input_text,
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        processed["input_ids"] = tokenizer_input["input_ids"]
        processed["attention_mask"] = tokenizer_input["attention_mask"]

        if mode == "train":

            if stage == 'TPG':
               
                silver_keyphrases = examples['silver_keyphrases']
                target_text = ';'.join(silver_keyphrases)

            elif stage == 'LRFT':

                pres_keys = examples['present_kps']['text']
                abs_keys = examples['absent_kps']['text']

                if order == 'pres_abs':
                    gold_keyphrases = pres_keys + abs_keys
                elif order == 'abs_pres':
                    gold_keyphrases = abs_keys + pres_keys
                elif order == 'random': 
                    temp = pres_keys + abs_keys
                    gold_keyphrases = random.sample(temp, len(temp))

                target_text = ';'.join(gold_keyphrases)
            
            elif stage =='TG':
                target_text = examples['title']


            tokenizer_output = tokenizer(
                target_text,
                padding="max_length",
                max_length=256,
                truncation=True
            )
            processed["decoder_input_ids"] = tokenizer_output["input_ids"]
            processed["decoder_attention_mask"] = tokenizer_output["attention_mask"]

        return processed

    dataset = dataset.map(
        preprocess_function,
        num_proc=8,
        remove_columns=dataset.column_names
    ).with_format("torch")
    dataloader = DataLoader(dataset, shuffle=(True if mode=="train" else False), batch_size=batch_size, num_workers=8, pin_memory=True)

    return dataloader


def jsonlload(fname):
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        j_list = [json.loads(line) for line in lines]

    return j_list


def jsonldump(j_list, fname):
    with open(fname, "w", encoding='utf-8') as f:
        for json_data in j_list:
            f.write(json.dumps(json_data, ensure_ascii=False)+'\n')
