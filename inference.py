import re
import os
import json
import torch
import argparse
from nltk.stem import PorterStemmer
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

#nltk.download('stopwords')

from transformers import BartTokenizer, BartForConditionalGeneration


def generate_keyphrases(model, text, tokenizer, device, max_length=512, num_beams=20):
    
    model.eval()  
    model.to(device)  

    with torch.no_grad():
        inputs = tokenizer(text, max_length=512, truncation=True, padding='longest', return_tensors="pt").to(device) 
        outputs = model.generate(inputs['input_ids'], num_beams=num_beams, max_length=max_length, num_return_sequences=num_beams)
        #output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True) 

    return output_text


def extract_from_beams(seq_list):
    total_phrases = []
    for seq in seq_list:
        phrases = seq.split(';')
        phrases = [ p.strip() for p in phrases if p.strip() != '']
        for phrase in phrases:
            if phrase not in total_phrases:
                total_phrases.append(phrase)
    return total_phrases

def save_list_to_txt(str_list,filepath):
    with open(filepath, 'w') as file:
        for string in str_list:
            file.write(string + "\n")




if __name__ == '__main__':

    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))


    parser = argparse.ArgumentParser(description='Script converted from notebook.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the Test Dataset.')
    parser.add_argument('--output_path', type=str, required=True, help="Path to the pred.txt file")
    parser.add_argument('--num_beams', type=int, default=20, help='beam size for beam search.')

    args = parser.parse_args()

    model_path = args.model_path
    dataset_path = args.dataset_path
    num_beams = args.num_beams
    


    for dataset_name in ['inspec', 'semeval', 'nus', 'krapivin', 'kp20k']:
        
        dataset_path = f'{args.dataset_path}/{dataset_name}/test.json'

        with open(dataset_path, 'r') as f:
            data = [json.loads(line) for line in f]

        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        special_tokens_dict = {'additional_special_tokens': ['[sep]', '[digit]']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        model = BartForConditionalGeneration.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        model.to(device)

        total_preds = []

        for d in tqdm(data):

            text = f"{d['title']['text']} [sep] {d['abstract']['text']}"

            extracted_phrases_seq = generate_keyphrases(model, text, tokenizer, device=device, max_length=512, num_beams=num_beams)
            extracted_phrases = extract_from_beams(extracted_phrases_seq)
            
            extracted_phrases = [p for p in extracted_phrases if p.split()[0] not in stop_words and p.split()[-1] not in stop_words]
            #print(extracted_phrases)

            str_extracted_phrases = ' ; '.join(extracted_phrases)

            total_preds.append(str_extracted_phrases)

            #print(extracted_phrases)

        
        model_name = args.model_path.split('/')[-1]
        result_path = args.output_path + f'/{model_name}'

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print(f"Directory created: {result_path}")

        save_list_to_txt(total_preds, result_path + f'/{dataset_name}_pred.txt')

    print('Done!')

    
