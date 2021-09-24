python3 run_wav2vec_clf.py \
    --pooling_mode="mean" \
    --model_name_or_path="lighteternal/wav2vec2-large-xlsr-53-greek" \
    --model_mode="wav2vec"\
    --output_dir="/home/aad13432ni/github/soxan/models" \
    --train_file=/home/aad13432ni/github/soxan/content/data/train.csv\
    --validation_file=/home/aad13432ni/github/soxan/content/data/test.csv \
    --test_file=/home/aad13432ni/github/soxan/content/data/test.csv \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-4 \
    --num_train_epochs=5.0 \
    --evaluation_strategy="steps"\
    --save_steps=100 \
    --eval_steps=100 \
    --logging_steps=100 \
    --save_total_limit=2 \
    --do_eval \
    --do_train \
    --freeze_feature_extractor