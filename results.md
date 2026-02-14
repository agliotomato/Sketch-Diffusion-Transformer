# 추론 결과 비교 (Inference Results Comparison)

## 데이터 범례 및 역할 (Data Legend & Roles)
| 유형 (Type) | 역할 및 설명 (Role & Description) |
| :--- | :--- |
| **원본 이미지 (Original Image)** | **목표 및 배경 소스**. <br> 1. **학습(Training)**: 손실(Loss) 계산을 위한 정답지(Ground Truth)로 사용됩니다. <br> 2. **추론(Inference)**: 마스크 외부의 얼굴, 몸, 배경을 제공하며, 이 "깨끗한 배경"은 픽셀 단위로 완벽하게 보존됩니다. |
| **매트/마스크 (Matte)** | **인페인팅 영역 정의**. <br> - **흰색 영역**: 모델이 새로운 머리카락을 생성하는 "헤어 영역"입니다. <br> - **검은색 영역**: 원본 이미지가 그대로 유지되는 "보호 영역"입니다. |
| **스케치 (Sketch)** | **구조적 조건 (Structural Condition)**. <br> 주요 입력 제어 요소입니다. 모델은 이 선들을 보고 생성할 헤어스타일의 흐름, 형태, 구조(예: 땋은 머리의 밀도, 방향)를 결정합니다. |
| **생성 결과 (Generated Result)** | **최종 결과물**. <br> 원본 이미지의 "깨끗한 배경"과 스케치를 바탕으로 모델이 생성한 "새로운 헤어"가 결합된 결과입니다. |

## 학습 샘플 (Training Samples Braid 1-8)
| ID | 원본 이미지 | 매트 (Mask) | 스케치 | 생성 결과 |
| :---: | :---: | :---: | :---: | :---: |
| **Braid 1** | <img src="dataset/braid/img/train/braid_1.png" width="200" /> | <img src="dataset/braid/matte/train/braid_1.png" width="200" /> | <img src="dataset/braid/sketch/train/braid_1.png" width="200" /> | <img src="results/train/braid_1.png" width="200" /> |
| **Braid 4** | <img src="dataset/braid/img/train/braid_4.png" width="200" /> | <img src="dataset/braid/matte/train/braid_4.png" width="200" /> | <img src="dataset/braid/sketch/train/braid_4.png" width="200" /> | <img src="results/train/braid_4.png" width="200" /> |
| **Braid 5** | <img src="dataset/braid/img/train/braid_5.png" width="200" /> | <img src="dataset/braid/matte/train/braid_5.png" width="200" /> | <img src="dataset/braid/sketch/train/braid_5.png" width="200" /> | <img src="results/train/braid_5.png" width="200" /> |
| **Braid 6** | <img src="dataset/braid/img/train/braid_6.png" width="200" /> | <img src="dataset/braid/matte/train/braid_6.png" width="200" /> | <img src="dataset/braid/sketch/train/braid_6.png" width="200" /> | <img src="results/train/braid_6.png" width="200" /> |
| **Braid 7** | <img src="dataset/braid/img/train/braid_7.png" width="200" /> | <img src="dataset/braid/matte/train/braid_7.png" width="200" /> | <img src="dataset/braid/sketch/train/braid_7.png" width="200" /> | <img src="results/train/braid_7.png" width="200" /> |
| **Braid 8** | <img src="dataset/braid/img/train/braid_8.png" width="200" /> | <img src="dataset/braid/matte/train/braid_8.png" width="200" /> | <img src="dataset/braid/sketch/train/braid_8.png" width="200" /> | <img src="results/train/braid_8.png" width="200" /> |

## 테스트 세트 결과 샘플 (Test Set Sample 10/107)
| ID | 원본 이미지 | 매트 (Mask) | 스케치 | 생성 결과 |
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

## 모델 비교 (Model Comparison: Orig vs GAN vs DiT)
| 파일명 (File) | 원본 (Original) | 스케치 (Sketch) | GAN-based | DiT-based (Ours) |
| :---: | :---: | :---: | :---: | :---: |
| **braid_14.png** | <img src="compare/img/braid_14.png" width="200" /> | <img src="compare/sketch/braid_14.png" width="200" /> | <img src="compare/gan-based/braid_14.png" width="200" /> | <img src="compare/dit-based/braid_14.png" width="200" /> |
| **braid_30.png** | <img src="compare/img/braid_30.png" width="200" /> | <img src="compare/sketch/braid_30.png" width="200" /> | <img src="compare/gan-based/braid_30.png" width="200" /> | <img src="compare/dit-based/braid_30.png" width="200" /> |
| **braid_49.png** | <img src="compare/img/braid_49.png" width="200" /> | <img src="compare/sketch/braid_49.png" width="200" /> | <img src="compare/gan-based/braid_49.png" width="200" /> | <img src="compare/dit-based/braid_49.png" width="200" /> |
| **bun_3.png** | <img src="compare/img/bun_3.png" width="200" /> | <img src="compare/sketch/bun_3.png" width="200" /> | <img src="compare/gan-based/bun_3.png" width="200" /> | <img src="compare/dit-based/bun_3.png" width="200" /> |
