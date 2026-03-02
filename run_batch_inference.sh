#!/bin/bash

# 설정할 이미지 경로들 (실제 테스트할 이미지로 변경하세요)
TARGET="dataset3/unbraid/img/test/wavy_766.png"
SKETCH="dataset3/braid/sketch/test/braid_2534.png"
MATTE="dataset3/braid/matte/test/braid_2534.png"

# 다른 파라미터들
SCALE=1.1
X=0
Y=0
GUIDANCE=20.0
STEPS=50
BG_RATIO=0.5

# 결과물이 저장될 폴더 생성
mkdir -p results_stage2_compare

echo "Starting Batch Inference for Stage 2 Checkpoints..."

# 15부터 40까지 5단위로 반복
for CP in 15 20 25 30 35 40; do
    CHECKPOINT_PATH="sd35_sketch_hair_lora/stage2_checkpoint-${CP}"
    OUTPUT_PATH="results/0302/cp${CP}.png"

    echo "=================================================="
    echo "Running Inference explicitly for: ${CHECKPOINT_PATH}"
    echo "=================================================="

    python run_transfer.py \
      --target="${TARGET}" \
      --sketch="${SKETCH}" \
      --matte="${MATTE}" \
      --scale=${SCALE} \
      --x=${X} \
      --y=${Y} \
      --checkpoint="${CHECKPOINT_PATH}" \
      --guidance=${GUIDANCE} \
      --steps=${STEPS} \
      --bg_start_ratio=${BG_RATIO} \
      --output="${OUTPUT_PATH}"
      
    echo "Saved: ${OUTPUT_PATH}"
    echo ""
done

echo "🎉 All inference tasks completed! Check the 'results_stage2_compare' folder."
