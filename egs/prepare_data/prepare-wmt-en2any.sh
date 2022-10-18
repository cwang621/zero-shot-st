#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
MOSES_DIR="mosesdecoder"

echo "arguments"
echo "DATA_ROOT: $WMT_DATA_ROOT"
echo "MOSES_DIR: $MOSES_DIR"
echo "version: $version"
echo "target language: $target"
echo "subword: $subword"
echo


BPE_TOKENS=$subword_tokens
SCRIPTPATH=$(pwd)
cd $WMT_DATA_ROOT

if [[ $target == "de" ]]; then
    CORPORA=(
        "training/europarl-v7.de-en"
        "commoncrawl.de-en"
        "training/news-commentary-v9.de-en"
    )
elif [[ $target == "fr" ]]; then
    CORPORA=(
        "training/europarl-v7.fr-en"
        "commoncrawl.fr-en"
        "training/news-commentary-v9.fr-en"
    )
elif [[ $target == "ru" ]]; then
    CORPORA=(
        "null"
        "commoncrawl.ru-en"
        "training/news-commentary-v9.ru-en"
    )
elif [[ $target == "es" ]]; then
    CORPORA=(
        "training/europarl-v7.es-en"
        "commoncrawl.es-en"
        "null"
    )
elif [[ $target == "ro" ]]; then
    CORPORA=(
        "training-parallel-ep-v8/europarl-v8.ro-en"
        "SETIMES/en-ro/setimes.en-ro"
    )
else
    echo "target language not supported"
    exit 1
fi

OUTDIR=${version}_en_$target

if [[ ! -d "$MOSES_DIR" ]]; then
    echo "Please set MOSES_DIR variable correctly to point to MosesDecoder."
    exit
fi
src=en
tgt=$target
lang=$src-$tgt
prep=$OUTDIR
tmp=$prep/tmp
orig=orig
dev=dev/newstest2013
mkdir -p $orig $tmp $prep

echo "pre-processing train data..."
for l in $src $tgt; do
    train_file=$tmp/train.tags.$lang.tok.$l
    if [[ -f $train_file ]]; then
        rm $train_file
    fi
    for f in "${CORPORA[@]}"; do
        if [[ $f != "null" ]]; then
            echo "containing: $orig/$f.$l"
            cat $orig/$f.$l >> $train_file
        fi
    done
done


# echo "pre-processing test data..."
# for l in $src $tgt; do
#     if [[ "$l" == "$src" ]]; then
#         t="src"
#     else
#         t="ref"
#     fi
#     if [[ $tgt != "es" ]]; then
#         test_file=$orig/test-full/newstest2014-${tgt}en-$t.$l.sgm
#     else
#         test_file=$orig/dev/newstest2012-$t.$l.sgm
#     fi
#     grep '<seg id' $test_file | \
#         sed -e 's/<seg id="[0-9]*">\s*//g' | \
#         sed -e 's/\s*<\/seg>\s*//g' | \
#         sed -e "s/\’/\'/g" > $tmp/test.$l
#     echo ""
# done

wc -l $tmp/*

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done


spm=$prep/spm
mkdir -p $spm
BPE_CODE=$spm/spm_${subword}${subword_tokens}_wave_joint

# learning spm or copying an existing one
resource_spm=egs/resources/$src-$tgt-spm
if [[ -d $SCRIPTPATH/$resource_spm ]]; then
    echo "Existing spm dictionary $resource_spm detected. Copying..."
    cp $SCRIPTPATH/$resource_spm/* $spm/
else
    echo "No existing spm detected. Please set the correct path"
fi

# applying SPM algorithm
for L in $src $tgt; do
    for f in train.$L valid.$L; do
        echo "apply_spm to ${f}..."
        python3 $SCRIPTPATH/egs/utils/spm_encode.py --model=$BPE_CODE.model < $tmp/$f > $tmp/spm.$f
    done
done


# cleaning train and valid, moving all to $prep
SCRIPTS=$MOSES_DIR/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

perl $CLEAN -ratio 1.5 $tmp/spm.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/spm.valid $src $tgt $prep/valid 1 250
# for L in $src $tgt; do
#     cp $tmp/spm.test.$L $prep/test.$L
# done

fairseq-preprocess \
  --source-lang $src --target-lang $tgt \
  --trainpref $prep/train \
  --validpref $prep/valid \
  --destdir $prep/bin \
  --srcdict $BPE_CODE.txt --tgtdict $BPE_CODE.txt \
  --workers $(nproc)

cd $SCRIPTPATH

# append wmt to must_c
mt_root=$SPEECH_DATA_ROOT/en-$tgt/mt_data
mkdir -p $mt_root/bin
cp $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/dict.* $mt_root/bin
for lang in en $tgt; do
  ln -s $WMT_DATA_ROOT/${version}_en_$target/bin/train.en-$tgt.$lang.idx $mt_root/bin/train.en-$tgt.$lang.idx
  ln -s $WMT_DATA_ROOT/${version}_en_$target/bin/train.en-$tgt.$lang.bin $mt_root/bin/train.en-$tgt.$lang.bin
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
  ln -s $WMT_DATA_ROOT/${version}_en_$target/bin/train.en-$tgt.$lang.idx $mt_expand_root/bin/train1.en-$tgt.$lang.idx
  ln -s $WMT_DATA_ROOT/${version}_en_$target/bin/train.en-$tgt.$lang.bin $mt_expand_root/bin/train1.en-$tgt.$lang.bin
  ln -s $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/valid.en-$tgt.$lang.idx $mt_expand_root/bin/valid.en-$tgt.$lang.idx
  ln -s $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/valid.en-$tgt.$lang.bin $mt_expand_root/bin/valid.en-$tgt.$lang.bin
  ln -s $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/test.en-$tgt.$lang.idx $mt_expand_root/bin/test.en-$tgt.$lang.idx
  ln -s $SPEECH_DATA_ROOT/en-$tgt/para_text/bin/test.en-$tgt.$lang.bin $mt_expand_root/bin/test.en-$tgt.$lang.bin
done
