import re
import json
import argparse
from tqdm import tqdm

DIGIT_token = '[digit]'


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def save_as_jsonl(data_list, file_path):
    """
    Save a list of dictionaries as a jsonl file.
    
    :param data_list: List of dictionaries.
    :param file_path: Path to the jsonl file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data_list:
            json_str = json.dumps(entry, ensure_ascii=False)
            file.write(json_str + '\n')


def meng17_tokenize(text):
    '''
    The tokenizer used in Meng et al. ACL 2017
    parse the feed-in text, filtering and tokenization
    keep [_<>,\(\)\.\'%], replace digits with <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    :param text:
    :return: a list of tokens
    '''
    # remove line breakers
    text = re.sub(r'[\r\n\t]', ' ', text)
    # pad spaces to the left and right of special punctuations
    text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
    # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
    tokens = list(filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\']', text)))

    return tokens


def replace_numbers_to_DIGIT(tokens, k=2):
    # replace big numbers (contain more than k digit) with <digit>
    tokens = [w if not re.match('^\d{%d,}$' % k, w) else DIGIT_token for w in tokens]

    return tokens


def process(text):
    tokens = meng17_tokenize(text)
    processed_tokens = replace_numbers_to_DIGIT(tokens)
    processed_str = ' '.join(processed_tokens)
    processed_str = processed_str.replace('< mask >', '<mask>')
    return processed_str



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Mining phrases from titles')

    parser.add_argument('--file_path', type=str, help='KP20k train.json path')
    parser.add_argument('--out_dir_path', type=str, help='output file path')

    args = parser.parse_args()

    kp20k_train_with_pseudo_label = read_jsonl(args.file_path)

    processed_json = []

    for j_data in tqdm(kp20k_train_with_pseudo_label):
        temp = {}

        title = j_data['title'].lower()

        abstract_present_phrases = j_data['abstract_present_phrases']
        abstract_present_phrases = abstract_present_phrases[:10]
        abstract_present_phrases = [p.lower() for p in abstract_present_phrases]

        masked_phrases = abstract_present_phrases[5:]  # masked in document
        not_masked_phrases = abstract_present_phrases[:5]

        masked_abstract = j_data["abstract"].lower()

        for phrase in masked_phrases:
            masked_abstract = re.sub(r'\b' + re.escape(phrase) + r'\b', '<mask>', masked_abstract, flags=re.IGNORECASE)

     
        title_absent_phrases = j_data['ranked_sub_title_absent_phrases'][:10]
        title_present_phrases = j_data['title_present_phrases']

        silver_keyphrases_list = title_present_phrases + not_masked_phrases + title_absent_phrases + masked_phrases 
        silver_keyphrases_list = [ p.lower() for p in silver_keyphrases_list ]


        processed_title = process(title)
        processed_abstract = process(masked_abstract)
        processed_silver_keyphrases = [process(s_key) for s_key in silver_keyphrases_list]

        temp['title'] = processed_title
        temp['abstract'] = processed_abstract
        temp['silver_keyphrases'] = processed_silver_keyphrases

        processed_json.append(temp)

    save_as_jsonl(processed_json[:350000], args.out_dir_path + '/kp20k_TPG_train.jsonl')
    save_as_jsonl(processed_json[350000:], args.out_dir_path + '/kp20k_TPG_valid.jsonl')
