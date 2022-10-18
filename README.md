# DCMA: Discrete Cross-Modal Alignment Enables Zero-Shot Speech Translation

The Code for the EMNLP 2022 main conference paper **Discrete Cross-Modal Alignment Enables Zero-Shot Speech Translation**, which aims to train an end-to-end speech translation model in a zero-shot fashion (only ASR and MT data are available).

# Training a Model on MuST-C

Training an En-De model as an example.

## Installation

The following environments are required.

* Python == 3.7
* torch == 1.8, torchaudio == 0.8.0, cuda==10.1
* python library

``` bash
pip install pandas sentencepiece editdistance PyYAML tqdm soudfile
```

* fairseq == 1.0.0a0+741fd13

``` bash
cd fairseq
pip install --editable ./
cd ..
```

**NOTE:** fairseq == 1.0.0a0 is not a stable release. Our codes are not compatible with the current fairseq version. Please install the corresponding version we provided, or you will need to modify the model codes.

### Data Preparation

0. set configuration

Please set the global variables of `WMT_DATA_ROOT`, `SPEECH_DATA_ROOT` and `SAVE_ROOT`. These will be where to put the WMT datasets, MUST-C dataset and checkpoints, respectively. Set the global variables `target` to specify the translation direction.
For example:

``` bash
export WMT_DATA_ROOT=~/WMT
export SPEECH_DATA_ROOT=~/MUSTC
export SAVE_ROOT=~/checkpoints
export target=de
mkdir -p $MUSTC_ROOT $WMT_ROOT $SAVE_ROOT
```


1. Download and uncompress the En-De MuST-C dataset to `$SPEECH_DATA_ROOT/en-$target`.


2. Download the WMT to `$WMT_ROOT/orig` via:

``` bash
bash egs/prepare_data/download-wmt.sh --wmt14 --data-dir $WMT_DATA_ROOT --target $target
```


3. Prepare the MUST-C datasets and produce a joint spm dictionary:

``` bash
bash egs/prepare_data/prepare-mustc-en2any.sh \
    --speech-data-root $SPEECH_DATA_ROOT --subword unigram --subword-tokens 10000
```

After this step, the directory `$SPEECH_DATA_ROOT` should look like:

```
├── en-de
│   ├── config_wave.yaml
│   ├── data
│   ├── para_text
│   ├── spm_unigram10000_wave_joint.model
│   ├── spm_unigram10000_wave_joint.txt
│   ├── spm_unigram10000_wave_joint.vocab
│   ├── train_wave_triple.tsv
│   ├── train_wave_en_asr.tsv
│   ├── dev_wave_triple.tsv
│   ├── dev_wave_en_asr.tsv
│   ├── tst-COMMON_wave_triple.tsv
│   ├── tst-COMMON_wave_en_asr.tsv
│   ├── tst-HE_wave_triple.tsv
│   ├── tst-HE_wave_triple.tsv
└── MUSTC_v1.0_en-de.tar.gz
```

Each `.tsv` file is formed as:

```
id	audio	n_frames	src_text	tgt_text	speaker
```

`tgt_text` contains source transcriptions in `XXX_en_asr.tsv`, and contains target translations in `XXX_wave_triple.tsv`.

​	4. Prepare the WMT datasets:

``` bash
bash egs/prepare_data/prepare-wmt-en2any.sh
```

This step will produce two folders `mt_data` and `mt_data_expand` in `$SPEECH_DATA_ROOT/en-$target`. The former only contains parallel text from WMT datasets, and the latter contains WMT and in-domain MUST-C text data.

## Training

5. Pre-training on MT data:

``` bash
export CUDA_VISIBLE_DEVICES=0,1
bash egs/scripts/train-en2any-MT.sh --save-root $SAVE_ROOT
```

Our experiment is carried out on 2 V100 GPUs. If you want to use more or less GPUs, please modify the `update-freq` in the training scripts.
Also, if you want to leverage the in-domain parallel text data from MUST-C, just add argument `--expand` like this:

```
bash egs/scripts/train-en2any-MT.sh --save-root $SAVE_ROOT --expand
```

6. Zero-shot fine-tuning:

``` bash
bash egs/scripts/train-en2any-zero-shot-ST.sh
```

**Note**: This step will use the same MT data as step 5 automatically. 

## Evaluation

7. Averaging Checkpoints and Evaluate It

We average the last 5 checkpoints and evaluate it.

``` bash
bash egs/scripts/eval-en2any-ST.sh
```

# License

The codes are dependent on fairseq code base, therefore carrying the MIT License of the original codes.

# Contact

If you have any questions, please feel free to contact us by sending an email to wangchen2020@ia.ac.cn