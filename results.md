# 추론 결과 비교

## 데이터 설명
| 유형 | 설명 |
| :--- | :--- |
| **Original** | **목표 및 배경 소스**. <br> 1. **학습(Training)**: loss 계산을 위한 Ground Truth로 사용 <br> 2. **추론(Inference)**: 마스크 외부의 얼굴, 몸, 배경을 제공하며, 이 배경은 픽셀 단위로 완벽하게 보존됩니다. |
| **Matte** | **인페인팅 영역 정의 및 노이즈 제어 (Inpainting Scope & Noise Injection)**. <br> **1. 흰색 영역 (Noise Injection)**: <br> &nbsp;&nbsp;- **생성 및 노이즈 주입**: 오직 이 영역에만 노이즈가 주입되고, 모델은 sketch 조건에 맞춰 이 노이즈를 새로운 머리카락으로 Denoising을 진행 <br> **2. 검은색 영역 (Preservation)**: <br> &nbsp;&nbsp;- **원본 보존**: 노이즈가 전혀 섞이지 않고, 원본 이미지의 픽셀 값이 100% 그대로 유지됩니다. |
| **Sketch** | **구조적 조건 (Structural Condition)**. <br> 주요 입력 제어 요소. 모델은 이 선들을 보고 생성할 헤어스타일의 흐름, 형태, 구조(예: 땋은 머리의 밀도, 방향)를 결정합니다. |
| **생성 결과** | **최종 결과물** : 배경과 스케치를 바탕으로 모델이 생성한 새로운 헤어가 결합된 결과 |

## Training Data
| ID | Original | Matte | Sketch | 생성 결과 |
| :---: | :---: | :---: | :---: | :---: |
| **Braid 1** | <img src="dataset/braid/img/train/braid_1.png" width="200" /> | <img src="dataset/braid/matte/train/braid_1.png" width="200" /> | <img src="dataset/braid/sketch/train/braid_1.png" width="200" /> | <img src="results/train/braid_1.png" width="200" /> |
| **Braid 4** | <img src="dataset/braid/img/train/braid_4.png" width="200" /> | <img src="dataset/braid/matte/train/braid_4.png" width="200" /> | <img src="dataset/braid/sketch/train/braid_4.png" width="200" /> | <img src="results/train/braid_4.png" width="200" /> |
| **Braid 5** | <img src="dataset/braid/img/train/braid_5.png" width="200" /> | <img src="dataset/braid/matte/train/braid_5.png" width="200" /> | <img src="dataset/braid/sketch/train/braid_5.png" width="200" /> | <img src="results/train/braid_5.png" width="200" /> |
| **Braid 6** | <img src="dataset/braid/img/train/braid_6.png" width="200" /> | <img src="dataset/braid/matte/train/braid_6.png" width="200" /> | <img src="dataset/braid/sketch/train/braid_6.png" width="200" /> | <img src="results/train/braid_6.png" width="200" /> |
| **Braid 7** | <img src="dataset/braid/img/train/braid_7.png" width="200" /> | <img src="dataset/braid/matte/train/braid_7.png" width="200" /> | <img src="dataset/braid/sketch/train/braid_7.png" width="200" /> | <img src="results/train/braid_7.png" width="200" /> |
| **Braid 8** | <img src="dataset/braid/img/train/braid_8.png" width="200" /> | <img src="dataset/braid/matte/train/braid_8.png" width="200" /> | <img src="dataset/braid/sketch/train/braid_8.png" width="200" /> | <img src="results/train/braid_8.png" width="200" /> |

## Test Data
| ID | Original | Matte | Sketch | 생성 결과 |
| :---: | :---: | :---: | :---: | :---: |
| **Braid 2534** | <img src="dataset/braid/img/test/braid_2534.png" width="200" /> | <img src="dataset/braid/matte/test/braid_2534.png" width="200" /> | <img src="dataset/braid/sketch/test/braid_2534.png" width="200" /> | <img src="results/test/braid_2534.png" width="200" /> |
| **Braid 2537** | <img src="dataset/braid/img/test/braid_2537.png" width="200" /> | <img src="dataset/braid/matte/test/braid_2537.png" width="200" /> | <img src="dataset/braid/sketch/test/braid_2537.png" width="200" /> | <img src="results/test/braid_2537.png" width="200" /> |
| **Braid 2539** | <img src="dataset/braid/img/test/braid_2539.png" width="200" /> | <img src="dataset/braid/matte/test/braid_2539.png" width="200" /> | <img src="dataset/braid/sketch/test/braid_2539.png" width="200" /> | <img src="results/test/braid_2539.png" width="200" /> |
| **Braid 2548** | <img src="dataset/braid/img/test/braid_2548.png" width="200" /> | <img src="dataset/braid/matte/test/braid_2548.png" width="200" /> | <img src="dataset/braid/sketch/test/braid_2548.png" width="200" /> | <img src="results/test/braid_2548.png" width="200" /> |
| **Braid 2562** | <img src="dataset/braid/img/test/braid_2562.png" width="200" /> | <img src="dataset/braid/matte/test/braid_2562.png" width="200" /> | <img src="dataset/braid/sketch/test/braid_2562.png" width="200" /> | <img src="results/test/braid_2562.png" width="200" /> |
| **Braid 2572** | <img src="dataset/braid/img/test/braid_2572.png" width="200" /> | <img src="dataset/braid/matte/test/braid_2572.png" width="200" /> | <img src="dataset/braid/sketch/test/braid_2572.png" width="200" /> | <img src="results/test/braid_2572.png" width="200" /> |
| **Braid 2574** | <img src="dataset/braid/img/test/braid_2574.png" width="200" /> | <img src="dataset/braid/matte/test/braid_2574.png" width="200" /> | <img src="dataset/braid/sketch/test/braid_2574.png" width="200" /> | <img src="results/test/braid_2574.png" width="200" /> |
| **Braid 2576** | <img src="dataset/braid/img/test/braid_2576.png" width="200" /> | <img src="dataset/braid/matte/test/braid_2576.png" width="200" /> | <img src="dataset/braid/sketch/test/braid_2576.png" width="200" /> | <img src="results/test/braid_2576.png" width="200" /> |
| **Braid 2590** | <img src="dataset/braid/img/test/braid_2590.png" width="200" /> | <img src="dataset/braid/matte/test/braid_2590.png" width="200" /> | <img src="dataset/braid/sketch/test/braid_2590.png" width="200" /> | <img src="results/test/braid_2590.png" width="200" /> |
| **Braid 2592** | <img src="dataset/braid/img/test/braid_2592.png" width="200" /> | <img src="dataset/braid/matte/test/braid_2592.png" width="200" /> | <img src="dataset/braid/sketch/test/braid_2592.png" width="200" /> | <img src="results/test/braid_2592.png" width="200" /> |

## Model Comparison: Orig vs GAN vs DiT
| 파일명 (File) | Original | Sketch | GAN-based | DiT-based |
| :---: | :---: | :---: | :---: | :---: |
| **braid_14.png** | <img src="compare/img/braid_14.png" width="200" /> | <img src="compare/sketch/braid_14.png" width="200" /> | <img src="compare/gan-based/braid_14.png" width="200" /> | <img src="compare/dit-based/braid_14.png" width="200" /> |
| **braid_30.png** | <img src="compare/img/braid_30.png" width="200" /> | <img src="compare/sketch/braid_30.png" width="200" /> | <img src="compare/gan-based/braid_30.png" width="200" /> | <img src="compare/dit-based/braid_30.png" width="200" /> |
| **braid_49.png** | <img src="compare/img/braid_49.png" width="200" /> | <img src="compare/sketch/braid_49.png" width="200" /> | <img src="compare/gan-based/braid_49.png" width="200" /> | <img src="compare/dit-based/braid_49.png" width="200" /> |
| **bun_3.png** | <img src="compare/img/bun_3.png" width="200" /> | <img src="compare/sketch/bun_3.png" width="200" /> | <img src="compare/gan-based/bun_3.png" width="200" /> | <img src="compare/dit-based/bun_3.png" width="200" /> |

## 실험 환경 (Experimental Setup)
본 결과는 다음 설정으로 생성되었습니다.
- **Model**: Stable Diffusion 3.5 Medium
- **Checkpoint**: `stage2_braid/stage2_checkpoint-10` (Braid Specialization)
- **LoRA Rank**: 128
- **Inference Steps**: 30
- **Guidance Scale**: 7.5
- **Resolution**: 1024x1024

## 학습 loss 그래프
![Stage 2 Loss](results/loss_graph_stage2.png)
> **Note**: Epoch 0-9 (총 10 Epoch). 초반에는 Loss가 불안정했으나, 후반부로 갈수록 Shape Loss와 MSE가 조금은 안정화 되는 경향

> **Note**: 학습을 더 진행하면 loss가 더 안정화 될까? overfitting이 발생할 수도 있을 것 같다.

