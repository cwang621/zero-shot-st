#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
echo "arguments"
echo "OPUS_DATA_ROOT: $OPUS_DATA_ROOT"
echo "target language: $target"
echo "subword: $subword"
echo "subword-tokens: $subword_tokens"
echo

SCRIPTPATH=$(pwd)
SCRIPTS=mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPE_TOKENS=$subword_tokens
src=en
tgt=$target
version=opus100
OUTDIR=${version}_${src}_${tgt}
mkdir -p $OPUS_DATA_ROOT
cd $OPUS_DATA_ROOT


CORPORA=(\
    "opus-100-corpus/v1.0/supervised/$src-$tgt/opus.${src}-${tgt}-train"
)
DEV_CORPORA="opus-100-corpus/v1.0/supervised/$src-$tgt/opus.${src}-${tgt}-dev"
TEST_CORPORA="opus-100-corpus/v1.0/supervised/$src-$tgt/opus.${src}-${tgt}-test"


# processing from scratch
if [[ ! -d "$SCRIPTS" ]]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi
lang=$src-$tgt
prep=$OUTDIR
tmp=$prep/tmp
orig=orig
dev=$DEV_CORPORA
mkdir -p $orig $tmp $prep


echo "pre-processing train data..."
for l in $src $tgt; do
    train_file=$tmp/train.tags.$lang.tok.$l
    if [[ -f $train_file ]]; then
        rm $train_file
    fi
    for f in "${CORPORA[@]}"; do
        if [[ $f != "null" ]]; then
            train_raw=$orig/$f.$l
            echo "containing: $train_raw"
            cat $orig/$f.$l  >> $train_file
        fi
    done
done

echo "pre-processing dev data..."
for l in $src $tgt; do
    dev_file=$tmp/valid.tags.$lang.tok.$l
    dev_raw=$orig/$dev.$l
    echo "containing: $dev_raw"
    cat $orig/$dev.$l > $dev_file
done

echo "pre-processing test data..."
for l in $src $tgt; do
    test_file=$tmp/test.$l
    test_raw=$orig/$TEST_CORPORA.$l
    echo "containing $test_raw"
    cat $orig/$TEST_CORPORA.$l | \
        sed -e "s/\’/\'/g"  > $test_file
done

wc -l $tmp/*

echo "using original valid set"
for l in $src $tgt; do
      cp $tmp/valid.tags.$lang.tok.$l $tmp/valid.$l
      cp $tmp/train.tags.$lang.tok.$l $tmp/train.$l
done

# learning BPE
TRAIN=$tmp/train.${tgt}-en
if [[ -f $TRAIN ]]; then
    rm -f $TRAIN
fi
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done


spm=$prep/spm
mkdir -p $spm
BPE_CODE=$spm/spm_unigram10000_wave_joint

# learning spm or copying an existing one
resource_spm=egs/resources/en-$tgt-spm
if [[ -d $SCRIPTPATH/$resource_spm ]]; then
    echo "Existing spm dictionary $resource_spm detected. Copying..."
    cp $SCRIPTPATH/$resource_spm/* $spm/
else
    echo "No existing spm detected."
fi

# applying BPE
for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
    echo "apply_spm.py to ${f}..."
        python3 $SCRIPTPATH/egs/utils/spm_encode.py --model=$BPE_CODE.model < $tmp/$f > $tmp/spm.$f
    done
done

# cleaning train and valid, moving all to $prep
perl $CLEAN -ratio 1.5 $tmp/spm.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/spm.valid $src $tgt $prep/valid 1 250
for L in $src $tgt; do
    cp $tmp/spm.test.$L $prep/test.$L
done

fairseq-preprocess \
  --source-lang $src --target-lang $tgt \
  --trainpref $prep/train \
  --validpref $prep/valid \
  --testpref $prep/test \
  --destdir $prep/bin \
  --srcdict $BPE_CODE.txt --tgtdict $BPE_CODE.txt \
  --workers $(nproc)

cd $SCRIPTPATH

# append wmt to must_c
mt_root=$SPEECH_DATA_ROOT/en-$tgt/mt_data
mkdir -p $mt_root/bin
cp $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/dict.* $mt_root/bin
for lang in en $tgt; do
  ln -s $OPUS_DATA_ROOT/${version}_en_$target/bin/train.en-$tgt.$lang.idx $mt_root/bin/train.en-$tgt.$lang.idx
  ln -s $OPUS_DATA_ROOT/${version}_en_$target/bin/train.en-$tgt.$lang.bin $mt_root/bin/train.en-$tgt.$lang.bin
  ln -s $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/valid.en-$tgt.$lang.idx $mt_root/bin/valid.en-$tgt.$lang.idx
  ln -s $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/valid.en-$tgt.$lang.bin $mt_root/bin/valid.en-$tgt.$lang.bin
  ln -s $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/test.en-$tgt.$lang.idx $mt_root/bin/test.en-$tgt.$lang.idx
  ln -s $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/test.en-$tgt.$lang.bin $mt_root/bin/test.en-$tgt.$lang.bin
done


mt_expand_root=$SPEECH_DATA_ROOT/en-$tgt/mt_data_expand
mkdir -p $mt_expand_root/bin
cp $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/dict.* $mt_expand_root/bin
for lang in en $tgt; do
  ln -s $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/train.en-$tgt.$lang.idx $mt_expand_root/bin/train.en-$tgt.$lang.idx
  ln -s $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/train.en-$tgt.$lang.bin $mt_expand_root/bin/train.en-$tgt.$lang.bin
  ln -s $OPUS_DATA_ROOT/${version}_en_$target/bin/train.en-$tgt.$lang.idx $mt_expand_root/bin/train1.en-$tgt.$lang.idx
  ln -s $OPUS_DATA_ROOT/${version}_en_$target/bin/train.en-$tgt.$lang.bin $mt_expand_root/bin/train1.en-$tgt.$lang.bin
  ln -s $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/valid.en-$tgt.$lang.idx $mt_expand_root/bin/valid.en-$tgt.$lang.idx
  ln -s $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/valid.en-$tgt.$lang.bin $mt_expand_root/bin/valid.en-$tgt.$lang.bin
  ln -s $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/test.en-$tgt.$lang.idx $mt_expand_root/bin/test.en-$tgt.$lang.idx
  ln -s $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/test.en-$tgt.$lang.bin $mt_expand_root/bin/test.en-$tgt.$lang.bin
done