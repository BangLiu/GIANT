# exp 0: start simple. only use "seq" edge, no embedding.
python3 GIANT_main.py  \
    --d_model 64 \
    --layers 3 \
    --num_bases 5 \
    --lr 0.001 \
    --data_type concept \
    --train_file "../../../../Datasets/original/concept/concepts.json" \
    --emb_tags \
    --task_output_dims 2 \
    --tasks "phrase" \
    --edge_types_list "seq" \
    --epochs 10 \
    --mode train \
    --not_processed_data \
    --processed_emb \
    --debug > debug.log.txt


# exp 1
python3 GIANT_main.py  \
    --d_model 32 \
    --layers 5 \
    --num_bases 2 \
    --lr 0.001 \
    --data_type concept \
    --train_file "../../../../Datasets/original/concept/concepts.json" \
    --emb_tags tag is_special is_stop \
    --task_output_dims 2 \
    --tasks "phrase" \
    --edge_types_list "seq" "req" "dep" \
    --epochs 20 \
    --mode train \
    --indicate_candidate > concept.log.txt


# exp 2
python3 GIANT_main.py  \
    --d_model 32 \
    --layers 5 \
    --num_bases 2 \
    --lr 0.001 \
    --data_type event \
    --train_file "../../../../Datasets/original/event/events.json" \
    --emb_tags tag is_special is_stop \
    --task_output_dims 2 \
    --tasks "phrase" \
    --edge_types_list "seq" "req" \
    --epochs 20 \
    --mode train \
    --indicate_candidate > event.log.txt

