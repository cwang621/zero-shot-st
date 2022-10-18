# average checkpoints
python3 egs/utils/average_checkpoints.py --inputs=$SAVE_DIR --output=$SAVE_DIR/checkpoint_avg.pt --num-update-checkpoints 5

fairseq-generate ${SPEECH_DATA_ROOT}/en-$target \
	--config-yaml config_wave.yaml --gen-subset tst-COMMON_wave_triple \
	--user-dir zero_shot_wave_to_text \
	--task zero_shot_wave_to_text --generate \
	--source-lang en --target-lang $target \
	--path $SAVE_DIR/checkpoint_avg.pt \
	--max-tokens 4000000 --beam 5 --scoring sacrebleu > $SAVE_DIR/res.txt

tail -1 $SAVE_DIR/res.txt