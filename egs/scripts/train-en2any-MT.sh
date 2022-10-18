#!/bin/bash
export suffix=""
export SAVE_ROOT=~/checkpoints

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --save-root) export SAVE_ROOT="$2"; shift ;;
        --expand) export suffix="_expand"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export MT_SAVE_DIR="$SAVE_ROOT/en-$target/mt${suffix}"
export SAVE_DIR=$MT_SAVE_DIR
mkdir -p $MT_SAVE_DIR


# Train on WMT data
fairseq-train $SPEECH_DATA_ROOT/en-$target/mt_data${suffix}/bin \
  --source-lang en --target-lang $target \
  --user-dir zero_shot_wave_to_text \
  --task translation_mlm \
  --arch transformer_vq_memory --memory-num 64 --vq-vars 50 --vq-groups 128 \
  --use-mlm \
  --weight-proj-depth 2 --weight-proj-factor 2 \
  --encoder-normalize-before --decoder-normalize-before \
  --share-all-embeddings \
  --save-dir $SAVE_DIR \
  --update-freq 4 --batch-size 144 --max-tokens 8196 --max-update 150000 \
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
  --skip-invalid-size-inputs-valid-test \
  --fp16 > $SAVE_DIR/log.txt

fairseq-generate $SPEECH_DATA_ROOT/en-$target/mt_data${suffix}/bin \
  --source-lang en --target-lang $target \
  --user-dir zero_shot_wave_to_text \
  --task translation_mlm \
  --path $SAVE_DIR/checkpoint_best.pt \
  --max-tokens 8192 --beam 5 --scoring sacrebleu \
  --remove-bpe sentencepiece > $SAVE_DIR/res.txt