python mining_phrases_in_titles.py  --file_path data/kp20k/train.json --out_dir_path data/tpg

python construct_pseudo_keyphrases.py  --file_path data/tpg/kp20k_title_phrases_mined_train.json --out_dir_path data/tpg

python preprocess.py  --file_path data/tpg/kp20k_with_pseudo_label.jsonl --out_dir_path data/tpg  