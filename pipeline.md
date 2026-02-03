# SD3.5 헤어 생성 파이프라인 문서 (SD3.5 Hair Generation Pipeline)

본 문서는 Stable Diffusion 3.5 Large 모델을 기반으로 한 스케치 기반 헤어 생성 모델의 아키텍처, 학습 원리, 및 운영 방식을 상세히 기술합니다.

## 1. 개요 (Overview)
- **목표**: 스케치(Sketch)를 입력받아 실사 같은 머리카락(Hair)을 생성하는 모델 개발.
- **기반 모델**: `stabilityai/stable-diffusion-3.5-large` (80억 파라미터).
- **제어 전략**:
    - **형태(Structure)**: 사전 학습된 Canny ControlNet 사용.
    - **질감/디테일(Texture)**: 독자 데이터셋으로 미세 조정(Fine-tuning)된 **LoRA 어댑터** 사용.
- **하드웨어 최적화**: NVIDIA A100 (40GB/80GB) 환경에 최적화.

---

## 2. 학습 구성 요소 요약 (Training Configuration Summary)

| 구성 요소 (Component) | 모델 종류 | 상태 (Status) | 설명 (Description) |
| :--- | :--- | :--- | :--- |
| **Transformer (SD3.5)** | MM-DiT Large (8B) | ❄️ **Frozen** (동결) | 모델의 지식(본체)은 건드리지 않음 |
| **LoRA Adapter** | Peft / Low-Rank Adaptation | 🔥 **Train Only** (학습) | `to_k, to_q, to_v, to_out` 레이어만 학습 |
| **ControlNet** | Canny ControlNet (SD3.5) | ❄️ **Frozen** (동결) | 스케치를 해석하는 능력은 이미 배운 거 씀 |
| **VAE** | AutoencoderKL | ❄️ **Frozen** (동결) | 이미지를 압축/복원하는 기능만 사용 |
| **Text Encoder (CLIP/T5)** | (Not Loaded) | ❌ **Unused** (미사용) | 메모리 절약을 위해 더미(Dummy) 값으로 대체 |

---

## 3. 핵심 아키텍처 (Architecture Components)

### A. SD3.5 Transformer (with LoRA)
- **역할**: 실제 이미지를 생성하는 "두뇌" 역할을 합니다.
- **학습 방식**: 80억 개의 파라미터를 전부 다시 학습하는 것은 비효율적이므로, **LoRA (Low-Rank Adaptation)** 레이어만 삽입하여 학습합니다.
- **타겟 모듈**: `to_k` (Key), `to_q` (Query), `to_v` (Value), `to_out` (Output Projections) - 어텐션 연산의 핵심 부분.
- **Rank**: 16 (경량화 설정).
- **최적화**: `gradient_checkpointing`을 활성화하여 VRAM 사용량을 대폭 절감합니다.

### B. ControlNet (Frozen - 얼림)
- **모델**: `stabilityai/stable-diffusion-3.5-large-controlnet-canny`.
- **역할**: 입력된 스케치를 해석하여 전체적인 "형태 가이드"를 제공합니다.
- **선정 이유**: 우리의 입력 데이터인 "스케치(선 그림)"는 컴퓨터가 보는 Canny Edge(외곽선) 맵과 의미적으로 완벽하게 일치합니다. 따라서 별도의 학습 없이 기존 ControlNet을 그대로 사용해도 최고의 성능을 냅니다.

### C. 텍스트 인코더 (Dummy Strategy - 텍스트 무시 전략)
- **역할**: 보통은 프롬프트(글자)를 이해하는 역할입니다.
- **학습 전략**: **사용 안 함 (Dummy 값 전달)**.
    - 무거운 T5/CLIP 인코더 대신 **"0으로 채워진 빈 텐서(Zero Tensor)"**를 입력으로 넣습니다.
    - **이유 1 (VRAM 절약)**: 텍스트 인코더만 10GB 이상의 메모리를 차지하므로 이를 제거하여 학습 효율을 높입니다.
    - **이유 2 (시각 집중)**: 모델이 텍스트에 의존하지 않고, 오직 **스케치(ControlNet)와 마스크 영역**만 보고 그림을 그리도록 강제합니다.
    - **주의사항 (Pooled Projections)**: 텍스트 인코더는 꺼도 되지만, **`pooled_projections` (압축된 텍스트 벡터)**는 반드시 `dummy` 값이라도 전달해야 합니다. SD3 ControlNet이 **시간(Timestep) 정보를 계산할 때 이 벡터를 필수적으로 참조**하기 때문입니다.

---

## 3. 학습 과정: 마스크드 디퓨전 (Masked Diffusion)

### 데이터셋 구조 (`dataset/braid`)
| 폴더명 | 내용물 | 역할 |
|:---:|:---:|:---:|
| `img/` | 원본 사진 (RGB) | 정답지 (Ground Truth) |
| `matte/` | 헤어 마스크 (Binary, 흑백) | 채점 범위 (이 부분만 학습함) |
| `sketch/` | 컬러/선 스케치 | 문제지 (입력 조건) |

### 학습 루프 (The Training Loop) 상세
1.  **입력 처리 (Input Processing)**:
    - **Latents ($z_0$)**: 원본 이미지를 VAE로 압축한 잠재 표현.
    - **Noisy Latents ($z_t$)**: 여기에 랜덤 노이즈를 섞어 이미지를 망가뜨립니다.
    - **Condition**: 스케치 이미지를 **VAE로 인코딩하여(16채널 Latent)** ControlNet에 입력합니다. (단순 RGB 입력은 안됨)
    - **Mask**: 마스크 이미지를 Latent 크기로 줄입니다.

2.  **모델 예측 (Prediction)**:
    - 모델은 `망가진 이미지` + `스케치 정보` + `빈 텍스트`를 받습니다.
    - 모델의 임무는 노이즈를 제거하여 원래 이미지를 복구하는 것(Noise Prediction / Flow Matching)입니다.

3.  **마스크드 손실 계산 (Masked Loss Calculation) - ⭐ 핵심**:
    - 일반적인 학습은 이미지 전체의 틀린 그림 찾기를 합니다.
    - **우리는 오직 "머리카락 마스크" 영역에서만 틀린 점을 찾습니다.**
    - **수식**:
      $$ L = \text{MSE}( \text{예측값} - \text{정답값} ) \times \text{Mask}_{hair} $$
    - **효과**: 모델이 얼굴이나 배경을 엉뚱하게 바꾸는 것을 방지하고, **오직 머리카락을 그리는 법만 집중적으로 학습**하게 됩니다.

---

## 4. 하드웨어 및 최적화 (Hardware & Optimization)
- **정밀도**: FP16 (Mixed Precision) 사용하여 속도 UP, 메모리 DOWN.
- **Gradient Checkpointing**: **켜짐 (ON)**.
    - 중간 연산 결과를 저장하지 않고 필요할 때 다시 계산하여 메모리를 아끼는 기술.
    - 이 기술 덕분에 A100 뿐만 아니라 24GB VRAM(A10, 3090)에서도 학습이 가능합니다.

## 5. 추론 전략 (Inference Strategy - 학습 후)
학습이 끝난 후 `inference_sd3_5.py`를 실행할 때의 동작:
1.  **LoRA 로드**: 우리가 가르친 "머리카락 질감 노하우"를 장착합니다.
2.  **텍스트 인코더 ON**: 학습 때는 껐지만, 추론 때는 켭니다. (예: "Red hair", "Blonde hair" 등 색상 조절을 위해)
3.  **최종 결과 합성**:
    - **형태**: 스케치를 따라감 (ControlNet 담당).
    - **질감**: 실사 같은 머릿결 (LoRA 담당).
    - **색상**: 프롬프트 명령어 (Text Encoder 담당).

## 6. 심화 기술: 왜 Reference Attention인가? (Why Reference Attention vs Inpainting?)
우리는 단순 인페인팅(Inpainting) 대신 **Reference Attention** 기법을 최종 목표로 합니다.

| 기술 | 방식 | 단점 |
| :--- | :--- | :--- |
| **Simple Inpainting** | 구멍난 부분(Mask)을 주변 픽셀을 보고 메꿈 | 얼굴과 머리카락 경계가 어색할 수 있음. 전신 조명/맥락을 놓칠 수 있음. |
| **Reference Attention** | **원본 이미지를 계속 쳐다보면서(Attend)** 머리카락을 그림 | **얼굴 톤, 조명, 분위기가 완벽히 일치**함. 경계선이 사라지고 자연스러움. |

*   **핵심 이유**: 헤어스타일 변경은 얼굴과의 조화가 생명입니다. Reference Attention은 생성된 머리카락이 **"원본 얼굴의 조명과 피부톤"을 실시간으로 참고**하게 만들어, 합성 티가 나지 않는 결과를 만듭니다.
