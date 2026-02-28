# S2I-Net (Sketch-to-Image Network) 학습 및 추론 파이프라인 최종 보고서
**Branch**: `feature/curriculum-training-restore`
**Base Model**: Stable Diffusion 3.5 Medium (SD3.5)

---

## 1. 개요 (Overview)
본 파이프라인은 «SketchHairSalon» 논문에서 제시된 **2단계 커리큘럼 학습(Curriculum Learning) 전략**과 **가우시안 기반 형태 손실(Gaussian Shape Loss)** 설계 철학을 SD3.5 아키텍처에 완벽히 이식한 최종 버전입니다. 

기존 단일 학습(Joint Training) 방식에서 발생하던 "스케치를 무시하는 현상"과 "특정 인종/얼굴(Prior)에 과적합되는 현상"을 근본적으로 해결하기 위해, 데이터 증강의 보간법(Interpolation) 이원화부터 손실 함수(Loss function)의 재설계까지 전 구간(A to Z)을 논문의 사양에 맞추어 재구축하였습니다.

---

## 2. 데이터 처리 및 증강 전략 (Dataset & Preprocessing)

### 2.1. 데이터 속성에 따른 보간법(Interpolation) 이원화
모델이 입력 스케치의 지시(Stroke)와 마스크의 경계(Boundary)를 오해 없이 받아들이도록 보간법을 분리 적용했습니다.
*   **Target Image (RGB)**: 연속적인 색상 값을 가지므로 `Bilinear` 보간법을 사용하여 시각적 자연스러움과 안티앨리어싱(Anti-aliasing)을 유지합니다.
*   **Sketch & Mask (Categorical Data)**: 학습 시 조건 입력(mask/sketch)은 픽셀에 할당된 범주적 의미(Categorical) 보존이 필수적이므로 변환 시 오직 **`Nearest`(최근접 이웃) 보간법**만을 고정밀도로 사용합니다. (반면, 최종 추론 합성 단계의 블렌딩 마스크는 경계의 시각적 이음새를 줄이는 것이 목적이므로 Bilinear 기반의 soft edge를 사용합니다.)

### 2.2. 전면적 데이터 증강 (Global Data Augmentation)
모델이 특정 스케치 위치(Spatial Prior)에 의존하는 것을 막기 위해, 1단계와 2단계 학습 전체에 걸쳐 강력한 동적 데이터 증강을 가합니다.
*   무작위 좌우 반전 (50% 확률)
*   무작위 평행 이동 (X, Y축 ±50px)
*   무작위 회전 (±15도)
조건 입력(Sketch/Mask) 변환 시에도 `Nearest`를 유지하여 왜곡 없는 강건한(Robust) 형태 대응 능력을 부여합니다.

#### 🚨 핵심 고려 사항: 3중 매칭 공간 정렬 (3-way Spatial Alignment)
기하학적 증강(회전, 이동, 반전)을 가할 때, **타겟 사진(Target Image), 스케치(Cond), 매트(Mask) 세 장의 이미지에 반드시 100% 동일한 난수(각도, 이동거리)를 일괄 적용**해야만 합니다. 하나라도 따로 회전하게 되면 모델이 스케치의 공간적 지시를 무시하게 되는 치명적 오작동(과적합)을 유발하며, 현재 본 파이프라인은 이 1:1 공간 매칭을 완벽히 강제하도록 구현되었습니다.

---

## 3. 2단계 커리큘럼 학습 전략 (Two-Stage Curriculum Training)
복잡한 땋은 머리 구조를 한 번에 학습하지 못하는 문제를 해결하기 위해, 논문의 순차적 학습 방식을 채택했습니다.

### Stage 1: 일반화 학습 (Generalization Phase)
*   **목적**: 다양한 형태의 "일반 머리"를 통해 기본적인 머리카락 텍스처 생성 능력, 스케치 추종 능력, 그리고 얼굴/배경과의 자연스러운 ब्ल렌딩(Blending) 능력을 체화합니다.
*   **데이터**: `unbraid` 카테고리 (3,000장) + 동적 데이터 증강
*   **손실 함수**: 기본 MSE Loss (마스크 영역 한정)
*   **결과**: 스케치가 어떤 각도나 크기로 주어져도 유연하게 머리카락 텍스처를 씌울 수 있는 "기초 체력"이 완성됩니다.

### Stage 2: 특화 파인튜닝 (Specialization Phase)
*   **목적**: 1단계의 범용적 가중치를 이어받아, 땋은 머리(Braid) 특유의 복잡한 밧줄 구조와 매듭(Knot) 형태를 집중적으로 각인시킵니다.
*   **데이터**: `braid` 카테고리 (1,000장) + 동적 데이터 증강
*   **손실 함수**: MSE Loss + **Shape Reconstruction Loss (`lambda_shape=0.1`)**

#### 💡 혁신 포인트: Gaussian Shape Loss (형태 복원 손실)
*   **문제**: 모델이 픽셀 단위(L1/MSE, Sobel) 오차만 최소화하려 들면, 땋은 머리의 세세한 텍스처(잔물결)에 집착하느라 전체적인 매듭 덩어리(Volume)를 놓칩니다.
*   **해결 (Gaussian L1 Loss)**: 예측 이미지(Prediction)와 정답 이미지(Target) 양쪽에 **강력한 가우시안 블러(Kernel 11, Sigma 10)**를 통과시켜 잔물결을 고의로 뭉개버립니다. 이후 뿌옇게 된 "덩어리(Volume)" 간의 L1 절대 오차를 계산합니다. 블러 이후 저주파 성분에서는 L1이 MSE 대비 큰 구조 오차에 선형적으로 반응해, 과도한 평균화(oversmoothing)보다 실루엣/볼륨 정합을 더 안정적으로 유도합니다.
*   **효과**: 모델은 텍스처 페인팅을 멈추고 스케치가 지시한 땋은 머리의 "형태적 실루엣"과 "굵은 매듭 구조"를 우선적으로 복원하도록 강제됩니다.

#### 💡 심화 이론: Rectified Flow(RF)와 Shape Loss의 완벽한 앙상블 원리
*   **RF의 본질**: 기존 디퓨전(SD 1.5)은 꼬불꼬불한 노이즈 예측(Epsilon)을 썼지만, SD3.5의 Rectified Flow는 완전 노이즈 상태에서 실제 사진으로 향하는 **'직선 경로의 속도(Velocity, `pred`)'**를 직접 예측합니다.
*   **속도(Velocity) vs 실제 형태(Volume)**: 모델이 예측해낸 값(`pred`) 자체는 그저 추상적인 속도 화살표 뭉치(Velocity Vector)일 뿐입니다. 여기에 대고 가우시안 블러를 때려봤자 형태 오차가 제대로 계산될 리 없습니다.
*   **RF 역구동 (Euler 역산)**: 코드 내 `pred_x0 = z_t - sigmas * pred`를 통해, 현재의 노이즈 낀 이미지(`z_t`)에서 모델이 예측한 속도(`pred`)만큼 시간을 되감아(`sigmas`) **"이 속도대로 쭉 전진하면 도착할 모델의 최종 예상 픽셀 그림($x_0$)"을 훈련 시점에 즉석에서 시뮬레이션으로 역산**해냅니다.
*   **대망의 L1 Shape Loss 타격**: 이렇게 RF 공식을 통해 역산되어 나온 **최종 도착 예정 이미지(`pred_x0`)**와 **진짜 정답 목표 사진(`target_x0`)** 양쪽에 강력한 가우시안 블러를 씌우고, 땋은 매듭(Knot) 덩어리가 똑같이 재현되었는지 L1(절대 오차)으로 무자비하게 비교해서 틀린 만큼 회초리(Gradient)를 칩니다.
*   **📝 요약**: SD3.5는 Rectified Flow라는 직선 가속도를 예측하지만, 우리의 Shape Loss는 그 가속도의 도착 지점($x_0$)을 RF 수학 공식으로 실시간으로 뽑아내서 형태적 오차를 계산하기 때문에 일말의 충돌 없이 최적의 시너지를 발휘합니다.

---

## 4. 네트워크 구조 및 노이즈 주입 (Architecture & Noise Injection)

### 4.1. 입력 차원 확장 (Input Channel Expansion)
사전학습 가중치를 최대한 보존하기 위해, 조건(Sketch Latent) 결합은 첫 입력 임베딩 레이어(`pos_embed`)에서의 입력 차원 확장(16 -> 32)으로 처리하였습니다. 새로 추가된 조건 입력 부분은 zero-init으로 초기화하여, 학습 초기에 사전학습 분포를 깨지 않도록 설계되었습니다.

### 4.2. 배경 블렌딩 모듈 (Background Blending via Soft Masking)
네트워크 내부에서 머리카락(Foreground)과 얼굴(Background)이 자연스럽게 이어지도록 처리합니다.
*   입력된 선명한 형태의 헤어 마스크(Matte)를 잠재 공간(Latent Space, 128x128)으로 `Nearest` 다운샘플링합니다.
*   이 마스크에 가우시안 필터(`sigma=10`)를 덧씌워 경계를 부드럽게 녹인 **소프트 마스크(Soft Mask)**를 생성합니다.
*   soft mask를 통해 forward process에서 머리카락 영역에만 충분한 noise를 부여하고, non-mask 영역은 원본 정보를 더 많이 유지시켜 배경(얼굴, 피부) 손상 위험을 현저히 억제하였습니다.

---

## 5. 추론 전략 (Inference Strategy)

### 5.1. Reference Attention 미사용 구조 (In-Context Reference)
별도의 복잡한 K, V 주입식 Reference Attention 모듈을 추가하지 않았습니다. 대신 non-mask 영역의 latent를 보존한 상태로 디노이징을 수행하면, 모델은 동일 샘플 내 배경 토큰을 지속적으로 관측하게 되어 self-attention이 배경의 색/조명 통계에 조건화된 생성을 수행하게 됩니다 (추가 모듈 없이 in-context reference 역할 수행).

### 5.2. Classifier-Free Guidance (CFG) 적용
스케치 구조를 강제하기 위해, 입력 스케치 조건이 존재하는 경우(`cond_img`)와 없는 경우(`Zero-tensor`)를 동시에 추론(Batch=2)하여 방향성을 외삽합니다.
*   **CFG Scale (3.5 ~ 4.5)**: 모델이 기존에 가지고 있던 Prior(얼굴 생김새에 따른 기본 머리 모양)를 억누르고, 사용자가 제공한 스케치의 형태와 결을 절대적으로 따르도록 만듭니다.

### 5.3. 잠재 공간 소프트 블렌딩 (Latent Soft Blending)
디노이징 스텝이 모두 끝난 직후, VAE를 통해 픽셀 공간으로 나가기 전에 **잠재 공간(Latent Space) 내에서 최종 알파 블렌딩**을 수행합니다. 
*   이때 사용되는 마스크는 `Bilinear`를 통해 경계의 안티앨리어싱이 유지된 상태로 적용되며, 생성된 복잡한 머리카락 텍스처가 원본 사진의 배경 픽셀과 이질감 없이 융합(Seamless Mix)되도록 보장합니다.
