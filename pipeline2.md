
본 문서는 **Stable Diffusion 3.5 Medium** 모델을 기반으로, 입력 채널 확장** 방식을 통해 스케치 입력을 받아 실사 퀄리티의 헤어를 생성하는 파이프라인을 기술합니다.

## 1. 아키텍처 

ControlNet을 사용하지 않고, SD3.5의 입력 레이어를 수정하여 스케치 정보를 입력하는 구조를 사용합니다.

### A. 모델 구성 요소
| 컴포넌트 | 모델명 | 상 태 | 역할 |
| :--- | :--- | :--- | :--- |
| **Base Model** | SD3.5 Medium | ❄️| 이미지 생성 능력 제공 |
| **Input Layer** |  | 🔥 | 입력 채널 16 $\to$ 32 확장 (Noisy Latent + Sketch Latent) |
| **LoRA Adapter** |  | 🔥 | **질감(Texture) 및 구조(Structure)** 동시 학습 |

---

## 2. 입력 데이터 전처리
데이터셋(`dataset_sd35.py`)에서 로드된 각 데이터는 다음과 같은 전처리 과정을 거쳐 모델에 입력됩니다.

### A. Target Image (원본 이미지)
*   **파일 형식**: RGB 이미지 (`img/train/*.png`)
*   **전처리**:
    1.  **Resize**: 1024x1024 (Bilinear Interpolation)
    2.  **Normalize**: `[0, 1]` $\to$ `[-1, 1]` 범위로 정규화
    3.  **VAE Encoding**: VAE를 통해 `Latent Space`로 압축 ($1024 \times 1024 \times 3 \to 128 \times 128 \times 16$)
    4.  **Scaling**: VAE Scaling Factor를 곱해 스케일 조정

### B. Sketch Image (입력 조건)
*   **파일 형식**: RGB 컬러 스케치 (`sketch/train/*.png`)
*   **역할**: 생성될 헤어의 형태(Shape)와 색상(Color) 가이드
*   **전처리**:
    1.  **Resize**: 1024x1024 (Bilinear Interpolation)
    2.  **Normalize**: `[0, 1]` $\to$ `[-1, 1]` 범위로 정규화
    3.  **VAE Encoding**: Target Image와 **동일한 VAE**를 사용하여 `Latent Space`로 인코딩 ($128 \times 128 \times 16$)
    4.  **Concatenation**: 노이즈가 섞인 Target Latent와 채널 방향으로 결합되어 모델의 입력이 됨 (총 32채널)

### C. Mask Image (학습 영역)
*   **파일 형식**: Grayscale 마스크 (`matte/train/*.png`)
*   **역할**: 손실 함수(Loss) 계산 시 **헤어 영역**에만 가중치를 부여
*   **전처리 (Soft Masking)**:
    1.  **Threshold**: 127 기준으로 이진화(Binary)
    2.  **Dilation**: `kernel(15x15)`로 영역 확장 $\to$ 헤어 라인과 피부의 경계 포함
    3.  **Gaussian Blur**: `sigma=10`으로 블러링 $\to$ 경계면을 부드럽게 만들어 자연스러운 블렌딩 유도
    4.  **Resize**: `Latent Space` 크기($128 \times 128$)로 축소 (Nearest Neighbor)

---

## 3. 학습 전략

학습은 **Curriculum Learning** 방식을 채택

### 1단계: 일반화
*   **목표**: "머리카락"이라는 소재의 일반적인 **질감**과 **블렌딩** 능력 학습.
*   **데이터**: `unbraid` (일반 생머리, 웨이브 등 다양한 스타일).
*   **제어 방식**: 스케치는 대략적인 가이드로만 작용하고, 자연스러운 헤어 텍스처 생성에 집중.
*   **손실 함수**: 단순 **MSE Loss** (픽셀 간 차이 최소화).

### 2단계: 복잡한 머리
*   **목표**: 땋은 머리(Braid)와 같은 복잡한 **구조적 디테일** 완벽 재현.
*   **데이터**: `braid` 
*   **손실 함수**: **MSE + Gradient Loss**.
    *   **Gradient Loss**: 이미지의 변화량(기울기)을 미분하여 비교. 흐릿하게 뭉개지는 것을 방지하고 **머리카락 한 올 한 올의 선명도**를 높임.
*   **효과**: 복잡한 패턴도 뭉개지지 않고 칼같이 선명하게 생성.

---

## 4. 손실 함수

### A. Normalized Masked MSE Loss
ROI(Hair)에만 집중하여 학습 효율을 극대화합니다.
$$
L_{mse} = \frac{\sum (\| \mathbf{v}_{pred} - \mathbf{v}_{gt} \|^2 \odot \mathbf{M}_{soft})}{\sum \mathbf{M}_{soft} + \epsilon}
$$
*   **Soft Mask ($\mathbf{M}_{soft}$)**: 경계면 가중치를 부드럽게 적용하여 이질감 없는 합성 유도.

### B. Pixel-Space Gradient Loss (Structural)
픽셀 간의 밝기 변화율(Gradient)을 비교하여 구조적 선명도를 학습합니다. (2단계에서 사용)
$$
L_{grad} = \| \nabla(\mathbf{I}_{pred}) - \nabla(\mathbf{I}_{gt}) \|_1
$$
*   Latent Space가 아닌 **Pixel Space**에서 계산하여 인간의 시각적 인지(Perceptual Quality)와 더 유사한 결과 도출.

---

## 5. 추론

ControlNet 없이 단일 Transformer 모델이 형태와 질감을 모두 처리합니다.

1.  **입력 데이터 준비**:
    *   **Original Image**: 전체 이미지 (배경 유지용)
    *   **Matte (Mask)**: 수정할 헤어 영역
    *   **Sketch**: 생성할 헤어의 가이드

2.  **Clean Background Injection (Inference Loop)**:
    *   **학습과 동일한 조건**을 추론 시에도 유지해야 합니다.
    *   매 Denoising Step $t$마다:
    $$
    z_{t}^{input} = \mathbf{M}_{soft} \odot z_{t}^{noise} + (1-\mathbf{M}_{soft}) \odot z_{0}^{clean}
    $$
    *   즉, **배경 영역**은 노이즈가 없는 **깨끗한 원본 Latent($z_{0}^{clean}$)**를 강제로 주입하고, **마스크 영역**만 노이즈($z_{t}^{noise}$)를 제거해 나갑니다.
    *   이를 통해 배경은 완벽하게 원본을 유지하면서, 마스크 영역만 자연스럽게 생성됩니다.

3.  **Concatenation**:
    *   최종 입력: $Concat([z_{t}^{input}, \text{Sketch Latent}], dim=1)$

---

## 6. 기술적 세부 사항 (Technical Details)
*   **Model**: StabilityAI Stable Diffusion 3.5 Medium
*   **Optimizer**: 8-bit AdamW (Memory Efficient)

## 7. 참고 문헌 (References)
*   [ArXiv:2407.15886](https://arxiv.org/pdf/2407.15886)

