#!/bin/bash

export target=it
export SPEECH_DATA_ROOT=/mnt/nas/users/huaike.wc/data/MUSTC

export SAVE_ROOT=/mnt/nas/users/huaike.wc/checkpoints
export CHIMERA_MT_SAVE_DIR="$SAVE_ROOT/en-$target/chimera_mt"
export SAVE_DIR=$CHIMERA_MT_SAVE_DIR
mkdir -p $SAVE_DIR


# Train on WMT data
python -m torch.distributed.launch --nproc_per_node=2 \
  /mnt/nas/users/huaike.wc/pkg/fairseq/fairseq_cli/train.py $SPEECH_DATA_ROOT/en-$target/mt_data/bin \
  --source-lang en --target-lang $target \
  --user-dir zero_shot_wave_to_text \
  --task translation_mlm \
  --arch transformer_vq_memory --memory-num 64 --vq-vars 0 --vq-groups 128 \
  --use-mlm \
  --weight-proj-depth 2 --weight-proj-factor 2 \
  --encoder-normalize-before --decoder-normalize-before \
  --share-all-embeddings \
  --save-dir $SAVE_DIR \
  --update-freq 4 --batch-size 144 --max-tokens 8192 --max-update 100000 \
  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 5.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 \
  --lr 0.0007 \
  --criterion mlm_mt \
  --label-smoothing 0.1 --mlm-weight 1.0 \
  --weight-decay 0.0 \
  --no-progress-bar \
  --save-interval-updates 1000 \
  --save-interval 999999 \
  --validate-interval 999999 \
  --keep-interval-updates 1 \
  --seed 1 --report-accuracy \
  --log-interval 100 \
  --ddp-backend=legacy_ddp \
  --fp16 > $SAVE_DIR/log.txt


fairseq-generate $SPEECH_DATA_ROOT/en-$target/mt_data/bin \
  --source-lang en --target-lang $target \
  --user-dir zero_shot_wave_to_text \
  --task translation_mlm \
  --path $SAVE_DIR/checkpoint_best.pt \
  --max-tokens 8192 --beam 5 --scoring sacrebleu \
  --remove-bpe sentencepiece > $SAVE_DIR/res.txt


# Zero-shot ST
export CHIMERA_ST_SAVE_DIR="$SAVE_ROOT/en-$target/chimera_zero_shot_st"
export SAVE_DIR=$CHIMERA_ST_SAVE_DIR
mkdir -p $SAVE_DIR

python -m torch.distributed.launch --nproc_per_node=2 \
  /mnt/nas/users/huaike.wc/pkg/fairseq/fairseq_cli/train.py $SPEECH_DATA_ROOT/en-$target \
  --source-lang en --target-lang $target \
  --config-yaml config_wave.yaml \
  --train-subset train_wave_en_asr \
  --valid-subset dev_wave_triple \
  --parallel-text-data $SPEECH_DATA_ROOT/en-$target/mt_data/bin \
  --user-dir zero_shot_wave_to_text \
  --task zero_shot_wave_to_text \
  --arch wave_transformer_vq_memory --memory-num 64 --vq-vars 0 --vq-groups 128 \
  --use-mlm --weight-proj-depth 2 --weight-proj-factor 2 \
  --encoder-normalize-before --decoder-normalize-before \
  --share-all-embeddings \
  --save-dir $SAVE_DIR \
  --update-freq 4 --max-tokens 2000000 --max-tokens-text 8192 --batch-size 144 --max-update 100000 \
  --max-source-positions-audio 1000000 --max-source-positions 6000 \
  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 5.0 \
  --lr-scheduler inverse_sqrt --warmup-updates 10000 \
  --lr 0.0001 --criterion chimera_st \
  --label-smoothing 0.1 --mlm-weight 1.0 --align-weight 1.0 --ctc-weight 1.0 --type contrastive \
  --update-mix-data \
  --load-pretrained-mt-model-from $CHIMERA_MT_SAVE_DIR/checkpoint_best.pt \
  --weight-decay 0.0 \
  --no-progress-bar \
  --save-interval-updates 1000 \
  --save-interval 999999 \
  --validate-interval 999999 \
  --keep-interval-updates 5 \
  --seed 1 --report-accuracy \
  --log-interval 100 \
  --ddp-backend=legacy_ddp \
  --fp16 > $SAVE_DIR/log.txt


# average checkpoints
python3 egs/utils/average_checkpoints.py --inputs=$SAVE_DIR --output=$SAVE_DIR/checkpoint_avg.pt --num-update-checkpoints 5

fairseq-generate ${SPEECH_DATA_ROOT}/en-$target \
	--config-yaml config_wave.yaml --gen-subset tst-COMMON_wave_triple \
	--user-dir zero_shot_wave_to_text \
	--task zero_shot_wave_to_text --generate \
	--source-lang en --target-lang $target \
	--path $SAVE_DIR/checkpoint_avg.pt \
	--max-tokens 2000000 --beam 5 --scoring sacrebleu > $SAVE_DIR/res.txt