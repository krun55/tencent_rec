#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}


# Step X: Precompute global CTRs (user & item)
if [ -z "${TRAIN_DATA_PATH}" ]; then
  echo "TRAIN_DATA_PATH is not set" >&2
  exit 1
fi
CTR_DIR=${USER_CACHE_PATH:-${TRAIN_DATA_PATH}}/ctr
if [ ! -f "${CTR_DIR}/user_ctr.json" ]; then
  echo "Precomputing CTR..."
  mkdir -p ${CTR_DIR}
  python -u precompute_ctr.py \
    --seq_path ${TRAIN_DATA_PATH}/seq.jsonl \
    --output_dir ${CTR_DIR} \
    --prior auto \
    --k_clip 10,10000
  echo "CTR precomputed at ${CTR_DIR}."
else
  echo "CTR artifacts exist at ${CTR_DIR}, skipping."
fi

# Step 2: Train with ranking loss
echo "Starting training with ranking loss..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u main.py \
    --use_ctr --use_userctr --use_itemctr \
    --ctr_dir ${CTR_DIR} \
    --use_rankloss=0 \
    --rankloss_mode=pairwise \
    --lambda_rank=0.3 \
    --rankloss_tau=0.6 \
    --rankloss_weights=1.0,0.6,0.4,0.2 \
    --click_actions=1 \
    --expo_actions=0 \
    --head_ratio=0.2 \
    --use_rel_bias=1 \
    --use_timefeat \
    --norm_first \
    --norm_type=rmsnorm \
    --encoder_type=hstu \
    --dqk=32 --dv=64 \
    --num_blocks=10 --num_heads=8 \
    --dropout_rate=0.2 \
    --batch_size=64 \
    --lr=0.002 \
    --warmup_steps=2666 \
    --clip_norm=1.0 \
    --temperature=0.03 \
    --num_epochs=8 \
    --pin_memory \
    --gradient_accumulation_steps=4