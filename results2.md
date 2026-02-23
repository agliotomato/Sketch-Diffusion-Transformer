# Matte를 다른 이미지에 맞추는 과정

| 20\_-50.png | 40\_10.png | 30\_10.png | 10\_10.png |
| :---: | :---: | :---: | :---: |
| <img src="results/cross_identity_test/20_-50.png" width="512"> | <img src="results/cross_identity_test/40_10.png" width="512"> | <img src="results/cross_identity_test/30_10.png" width="512"> | <img src="results/cross_identity_test/10_10_matte.png" width="512"> |

## 최종 결과 (Final Results)

| 10\_10\_v2.png | result\_transfer\_ckpt30.png |
| :---: | :---: |
| <img src="results/cross_identity_test/10_10_v2.png" width="512"> | <img src="results/cross_identity_test/10_10.png" width="512"> |

| 781\_matte.png | 781\_0\_0.png |
| :---: | :---: |
| <img src="results/cross_identity_test/781_matte.png" width="512"> | <img src="results/cross_identity_test/781_0_0.png" width="512"> |

---

##  재학습 계획

추론 시의 배경 주입 스케줄링(Background Scheduling)만으로 해결되지 않는 근본적인 'Identity Prior 문제를 해결하기 위한 재학습 전략이 필요.

### 1. 목표: Identity-Style 분리
모델이 "얼굴 모양이 이러니 머리는 이래야 해"라고 판단하는 대신, "얼굴이 누구든 상관없이 스케치 선이 있는 곳에 머리를 그려야 한다"는 절대 규칙을 학습

### 2. 핵심 구현: 랜덤 아핀 변형
`dataset_sd35.py`를 수정하여 학습 시마다 다음 변형을 적용.
- **배경(얼굴)**: 변형 없이 그대로 유지.
- **스케치 & 마스크**: 동일한 랜덤 변형(회전, 이동)을 세트로 적용.
  - **회전(Rotation)**: $-15^{\circ} \sim 15^{\circ}$
  - **평행 이동(Translation)**: 상하좌우 최대 50픽셀
- **결과**: 얼굴과 스케치의 위치 관계가 매번 어긋나게 되어, 모델이 스케치에만 의존하여 형태를 잡는 법을 학습

### 3. 학습 프로세스
1. **`dataset_sd35.py` 수정**: `torchvision.transforms.functional`을 이용한 실시간 변형 로직 추가.
2. **Stage 1 (Unbraid)**: 일반 머리 데이터로 위치 강인성(Robustness) 기초 학습.
3. **Stage 2 (Braid)**: 복잡한 땋은 머리 데이터로 정교한 구조 학습 (Gradient Loss 병행).
