# Improving Low-Resource Keyphrase Generation through Unsupervised Title Phrase Generation
This code is for paper "[Improving Low-Resource Keyphrase Generation through Unsupervised Title Phrase Generation](https://aclanthology.org/2024.lrec-main.775)".

Our implementation is built on the source code from [low-resource-kpgen](https://github.com/xiaowu0162/low-resource-kpgen). Thanks for their work.

## Environment

```
conda env create -f environment.yml
conda activate tpgkeygen
```

## Data
- Download KP20k training set
```
bash scripts/prepare_data.sh
```

- Mine phrases from titles and create pseudo labels
```
bash scripts/mining_and_construct_pseudo_keyphrases.sh
```

## Traninig
### TPG
```
bash scripts/tpg.sh
```

### LRFT
```
bash scripts/lrft.sh
```

## Inference and Evaluation
```
bash scripts/inference.sh
bash scripts/evaluation.sh
```

## Citation

  
If you use this code, please cite our paper: 

```
@inproceedings{kang-shin-2024-improving-low,
    title = "Improving Low-Resource Keyphrase Generation through Unsupervised Title Phrase Generation",
    author = "Kang, Byungha  and
      Shin, Youhyun",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.775",
    pages = "8853--8865",
    abstract = "This paper introduces a novel approach called title phrase generation (TPG) for unsupervised keyphrase generation (UKG), leveraging a pseudo label generated from a document title. Previous UKG method extracts all phrases from a corpus to build a phrase bank, then draws candidate absent keyphrases related to a document from the phrase bank to generate a pseudo label. However, we observed that when separating the document title from the document body, a significant number of phrases absent from the document body are included in the title. Based on this observation, we propose an effective method for generating pseudo labels using phrases mined from the document title. We initially train BART using these pseudo labels (TPG) and then perform supervised fine-tuning on a small amount of human-annotated data, which we term low-resource fine-tuning (LRFT). Experimental results on five benchmark datasets demonstrate that our method outperforms existing low-resource keyphrase generation approaches even with fewer labeled data, showing strength in generating absent keyphrases. Moreover, our model trained solely with TPG, without any labeled data, surpasses previous UKG method, highlighting the effectiveness of utilizing titles over a phrase bank. The code is available at https://github.com/kangnlp/low-resource-kpgen-through-TPG.",
}
```
