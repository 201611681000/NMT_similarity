#!/bin/bash
#beamsize=(4 8 12 20 50 100)
b=4
batchsize=30
CUDA_VISIBLE_DEVICES=1,0 python3 -u ./generate.py /home/wen/shike/data-bin/iwslt14.tokenized.de-en --path ./checkpoint51.pt --task translation --gen-subset test --batch-size ${batchsize} --beam ${b} --nbest ${b} --sacrebleu --remove-bpe --usebert \
		--bert-model-path ../bert_base_model \
#b=8
#batchsize=30
#CUDA_VISIBLE_DEVICES=1 nohup python3 -u ./generate.py /home/wen/shike/data-bin/iwslt14.tokenized.de-en --path /home/wen/shike/090baseline_checkpoints/checkpoint_best.pt --task translation --gen-subset test --batch-size ${batchsize} --beam ${b} --nbest ${b} --sacrebleu --remove-bpe --usebleu > usebert_de2en_${b}.log 
#beamsize=(4)
#for b in "${beamsize[@]}" 
#do
#		batchsize=10
#		CUDA_VISIBLE_DEVICES=1 python3 ./generate.py /home/wen/shike/data-bin/iwslt14.tokenized.de-en --path /home/wen/shike/090baseline_checkpoints/checkpoint_best.pt --task translation --gen-subset test --batch-size ${batchsize} --beam ${b} --nbest ${b} --sacrebleu --remove-bpe --usebleu > usebert_de2en_${b}.log 
#		echo ${b}
#done 

#batchsize=130
#if [ ${b} -eq 12 ]
#		then
#				batchsize=100
#		elif [ ${b} -eq 20 ]
#		then
#				batchsize=80
#		elif [ ${b} -eq 50 ]
#		then
#				batchsize=24
#		elif [ ${b} -eq 100 ]
#		then
#				batchsize=10
#		fi
