# GIANT
Code and data for paper "GIANT: Scalable Creation of a Web-scale Ontology"

<https://arxiv.org/pdf/2004.02118.pdf>

Graph for user Interest ANd Text understanding.

## How to run

1. 代码路径
  some-path/GIANT/src/model/GIANT   最后这个GIANT就是我发给你的代码压缩包
2. Stanford NLP 路径
  some-path/GIANT/tool/stanford-corenlp-full-2018-10-05
3. 用于保存生成的中间结果的文件夹
  some-path/GIANT/output/checkpoint/
  some-path/GIANT/output/debug/
  some-path/GIANT/output/figure/
  some-path/GIANT/output/result/
  some-path/GIANT/output/log/
  some-path/GIANT/output/pkl/
4. 输入数据的放置位置。这里我把event和concept数据分开了
  some-path/Datasets/original/event/events.json
  some-path/Datasets/original/concept/concepts.json

5. 输出的数据放置位置
  some-path/Datasets/processed/event/
  some-path/Datasets/processed/concept/
  注意： 生成词向量花了很多时间。

在运行命令的时候，加上 --processed_emb 这个选项，就可以不会重新处理词向量，也不会需要下载词向量。

6. 测试运行

python3 GIANT_main.py \
    --data_type concept \
    --train_file "../../../../Datasets/original/concept/concepts.json" \
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
