#!/bin/bash

export OPUS_DATA_ROOT=~/data/OPUS
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data-dir) export OPUS_DATA_ROOT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
echo "arguments"
echo "OPUS_DATA_ROOT: $OPUS_DATA_ROOT"
echo

SCRIPTPATH=$(pwd)

mkdir -p $OPUS_DATA_ROOT/orig
cd $OPUS_DATA_ROOT/orig
tarfile=opus-100-corpus-v1.0.tar.gz

wget -P $DATA_ROOT https://object.pouta.csc.fi/OPUS-100/v1.0/$tarfile
tar xzvf $tarfile


cd $OPUS_DATA_ROOT
# cloning github repository
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

cd $SCRIPTPATH