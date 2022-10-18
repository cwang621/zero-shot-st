#!/bin/bash
export ST_SAVE_DIR="$SAVE_ROOT/en-$target/zero_shot_st${suffix}"
export SAVE_DIR=$ST_SAVE_DIR
mkdir -p $SAVE_DIR


fairseq-train $SPEECH_DATA_ROOT/en-$target \
  --source-lang en --target-lang $target \
  --config-yaml config_wave.yaml \
  --train-subset train_wave_en_asr \
  --valid-subset dev_wave_triple \
  --parallel-text-data $SPEECH_DATA_ROOT/en-$target/mt_data${suffix}/bin \
  --user-dir zero_shot_wave_to_text \
  --task zero_shot_wave_to_text \
  --arch wave_transformer_vq_memory --memory-num 64 --vq-vars 50 --vq-groups 128 \
  --use-mlm --weight-proj-depth 2 --weight-proj-factor 2 \
  --encoder-normalize-before --decoder-normalize-before \
  --share-all-embeddings \
  --save-dir $SAVE_DIR \
  --update-freq 4 --max-tokens 2000000 --max-tokens-text 8192 --batch-size 144 --max-update 150000 \
  --max-source-positions-audio 1000000 --max-source-positions 6000 \
  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 5.0 \
  --lr-scheduler inverse_sqrt --warmup-updates 10000 \
  --lr 0.0001 --criterion zero_shot_st \
  --label-smoothing 0.1 --mlm-weight 1.0 --align-weight 1.0 --ctc-weight 1.0 --teacher-mode \
  --update-mix-data \
  --load-pretrained-mt-model-from $MT_SAVE_DIR/checkpoint_best.pt \
  --weight-decay 0.0 \
  --no-progress-bar \
  --save-interval-updates 1000 \
  --save-interval 999999 \
  --validate-interval 999999 \
  --keep-interval-updates 5 \
  --seed 1 --report-accuracy \
  --log-interval 100 \
  --ddp-backend=legacy_ddp \
  --skip-invalid-size-inputs-valid-test \
  --fp16 > $SAVE_DIR/log.txt
