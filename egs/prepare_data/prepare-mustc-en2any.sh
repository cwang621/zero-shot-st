#!/bin/bash
export subword=unigram
export subword_tokens=10000
export SPEECH_DATA_ROOT=~/data/MUSTC

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --speech-data-root) export SPEECH_DATA_ROOT="$2"; shift ;;
        --subword) export subword="$2"; shift ;;
        --subword-tokens) export subword_tokens="$2"; shift ;;
        --target) export target="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "arguments"
echo "SPEECH_DATA_ROOT: $SPEECH_DATA_ROOT"
echo "target language: $target"
echo "subword: $subword"

src="en"
tgt=$target
BPE_TOKENS=$subword_tokens
SCRIPTPATH=$(pwd)
resource=$SCRIPTPATH/egs/resources

python3 egs/prepare_data/prep_mustc_data.py --data-root $SPEECH_DATA_ROOT --vocab-type $subword \
  --vocab-size $subword_tokens --languages $target

# extract asr tsv from st tsv
CUR_ROOT=$SPEECH_DATA_ROOT/en-$target
for split in train dev tst-COMMON tst-HE; do
	cut -f1-4 $CUR_ROOT/${split}_wave_triple.tsv > $CUR_ROOT/tmp1.tsv
	cut -f4 $CUR_ROOT/${split}_wave_triple.tsv > $CUR_ROOT/tmp2.tsv
	tail -n +2 $CUR_ROOT/tmp2.tsv | sed "1i tgt_text" > $CUR_ROOT/tmp3.tsv
	cut -f6 $CUR_ROOT/${split}_wave_triple.tsv > $CUR_ROOT/tmp4.tsv
	paste $CUR_ROOT/tmp1.tsv $CUR_ROOT/tmp3.tsv $CUR_ROOT/tmp4.tsv > $CUR_ROOT/${split}_wave_en_asr.tsv
	rm $CUR_ROOT/tmp*
done

# train spm model
BPE_CODE=$SPEECH_DATA_ROOT/en-$tgt/spm_${subword}${subword_tokens}_wave_joint
resource_spm=$resource/en-$tgt-spm
para_text=$SPEECH_DATA_ROOT/en-$tgt/para_text
mkdir -p $para_text

for split in train dev tst-COMMON; do
  cut -f4 $SPEECH_DATA_ROOT/en-$tgt/${split}_wave_triple.tsv | tail -n +2 > $para_text/${split}.en
  cut -f5 $SPEECH_DATA_ROOT/en-$tgt/${split}_wave_triple.tsv | tail -n +2 > $para_text/${split}.$tgt
done
cat $para_text/train.en $para_text/train.$tgt > $para_text/train.en-$tgt
if [[ -d $resource_spm ]]; then
    echo "Existing spm dictionary $resource_spm detected. Copying..."
    cp $resource_spm/* $SPEECH_DATA_ROOT/en-$tgt/
else
    echo "No existing spm detected. Training..."
    mkdir -p $resource_spm
    python3 egs/utils/spm_train.py --input=$para_text/train.en-$tgt --model_prefix=$BPE_CODE --vocab_size=$subword_tokens \
        --character_coverage=1.0 --model_type=$subword --unk_id=3 --bos_id=0 --eos_id=2 --pad_id=1
    cut -f1 $BPE_CODE.vocab | tail -n +5 | sed "s/$/ 100/g" > $BPE_CODE.txt
    cp $BPE_CODE.* $resource_spm/
fi

for split in train dev tst-COMMON; do
  python3 egs/utils/spm_encode.py --model=$BPE_CODE.model < $para_text/$split.en > $para_text/spm.$split.en
  python3 egs/utils/spm_encode.py --model=$BPE_CODE.model < $para_text/$split.$tgt > $para_text/spm.$split.$tgt
done

fairseq-preprocess \
  --source-lang $src --target-lang $tgt \
  --trainpref $para_text/spm.train \
  --validpref $para_text/spm.dev \
  --testpref $para_text/spm.tst-COMMON \
  --destdir $para_text/bin \
  --srcdict $BPE_CODE.txt --tgtdict $BPE_CODE.txt \
  --workers $(nproc)
