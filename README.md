# GIANT
Code and data for paper "GIANT: Scalable Creation of a Web-scale Ontology"

<https://arxiv.org/pdf/2004.02118.pdf>

Please cite our paper if this project is helpful to your work or research, thanks.

## How to run

1. Download files
  Stanford CoreNLP (<https://stanfordnlp.github.io/CoreNLP/download.html>) and Chinese word embedding (<https://ai.tencent.com/ailab/nlp/embedding.html>). For word embedding, see note in the bottom.

  ​

2. Revise paths and put files in appropriate paths
  File paths are defined in common/constants.py. So just go to that file and change the paths according to your own setting. Similarly for other paths defined in some source files.


6. test run

    python3 GIANT_main.py \
        --data_type concept \
        --train_file    "../../../../Datasets/original/concept/concepts.json" \
        --emb_tags \
        --task_output_dims 2 \
        --tasks "phrase" \
        --edge_types_list "seq" "dep" "contain" "synonym" \
        --d_model 32 \
        --layers 3 \
        --num_bases 5 \
        --epochs 10 \
        --mode train \
        --debug

Note: add —processed_emb in above command can help to prevent re-processing word embeddings (as it is time consuming). In this case, you also don't need to download the Chinese word embedding file. It is quite big. Our experience shows that add word embedding feature as a part of node features is not quite helpful in our tasks. Therefore, I think it is safe to ignore the word embedding features in your experiments. If not using word embedding, you may need to revise data_loader.py to avoid some running errors. However, you can still try to improve by word embeddings.