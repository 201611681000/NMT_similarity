#!/bin/bash
rootdir=/home/wen/shike
dir=${rootdir}/1211_net_sim
python3 -u ${dir}/fairseq_cli/train.py ${rootdir}/data-bin/iwslt14.tokenized.de-en  \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --save-dir ${dir}/checkpoints \
    --restore-file ${rootdir}/090baseline_checkpoints/checkpoint_best.pt \
	--bert-model-path /data/sdb/solarawang/bert_base_model \
	--bert-german-path /data/sdb/solarawang/bert_german_model \
	--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --reset-optimizer \
    --beam 7 --cands-into-model 3 \
    --keep-last-epochs 10 \
	--bert-hidden-size 768 --ffn-net-hidden-size 512 --bert-vocab-size 30522 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-epoch 60  --distributed-world-size 2 \
    --max-tokens 256 --update-freq 3 --no-progress-bar --ddp-backend=no_c10d \
    --log-format json \
    --log-interval 2 


#--encoder-embed-dim	768 --decoder-embed-dim 768 \
