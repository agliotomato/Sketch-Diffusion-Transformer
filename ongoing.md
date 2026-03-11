# 시도해본 두 가지 학습방식

## 0. VAE 복원 테스트

![VAE 복원 테스트 결과](./vae_reconstruction_test.png)

 VAE 복원은 정상 . 문제는 **손실 함수(Loss)**

## 1. 손실 함수 (Loss Function) 차이

### Sobel Gradient Loss

*   **원리:** `GradientLoss` 클래스를 사용하여 이미지에 Sobel 필터(x, y축)를 씌운 뒤, 1차 미분값 즉 윤곽선(엣지)을 추출합니다. 이 윤곽선 간의 L1 오차를 최소화하는 효과.


| 원본 사진 | 스케치 | 정렬 | 결과  |
| :---: | :---: | :---: | :---: |
| <img src="dataset/braid/img/test/wavy_766.png" width="200"> | <img src="dataset/braid/sketch/test/braid_2534.png" width="200"> | <img src="results/0228_2/766_2537_matte.png" width="200">| <img src="results/0228_2/2.png" width="200"> |
| <img src="dataset/braid/img/test/wavy_781.png" width="200"> | <img src="dataset/braid/sketch/test/braid_2537.png" width="200"> | <img src="results/0228_2/781_2537_matte.png" width="200">| <img src="results/0228_2/5.png" width="200"> |



*   **장점:** Sobel 필터는 고주파 성분(얇은 선, 질감)을 강하게 잡아냅니다. 따라서 모델이 원본 스케치의 세밀한 땋은 선을 무시하지 못하고 강제로 따라가게 만듭니다.
*   **단점:** 볼륨감이 뭉개지는 경향이 있음.

### Gaussian Shape Loss 기반 -> 기존 논문에서 제시된 방법
*   **원리:** `ShapeLoss` 클래스를 사용하여 예측값과 정답에 아주 강한 **가우시안 블러($\sigma=10.0$)**를 먹인 뒤, 그 뭉개진 덩어리(Volume) 간의 L1 오차를 최소화합니다.

| 원본 사진 | 스케치 | 정렬 | 결과  |
| :---: | :---: | :---: | :---: |
| <img src="dataset/braid/img/test/wavy_766.png" width="200"> | <img src="dataset/braid/sketch/test/braid_2534.png" width="200"> | <img src="results/0228_2/766_2537_matte.png" width="200">| <img src="results/0302/40_20_30.png" width="200"> |
| <img src="dataset/braid/img/test/wavy_781.png" width="200"> | <img src="dataset/braid/sketch/test/braid_2537.png" width="200"> | <img src="results/0228_2/781_2537_matte.png" width="200">| <img src="results/0302/3.png" width="200"> |




*   **장점:** 전체적인 머리의 실루엣이나 부피감을 따라가는 능력이 탁월합니다.
*   **단점:** 얇은 땋은 선(고주파) 오차가 블러로 인해 모두 뭉개짐. 따라서 모델은 스케치의 선을 무시하고 자신의 학습된 머리로 마음대로 채우게 됩니다. **(현재 스케치를 무시하는 가장 큰 이유)**

### GAN에서는 충분했는데 SD 3.5에서는 안 통하는 이유 분석
SketchHairSalon은 GAN구조를 사용 

GAN 구조에서는 Generator가 뭉개진 Shape Loss만 보고 형태를 그릴 때,  **Discriminator와 Perceptual Loss**가 "진짜 머리카락 사진의 고주파 질감(Texture)과 디테일"을 강제로 학습하도록 유도하였음

하지만 SD3.5 구조에서는 판별자나 Perceptual Loss가 부재함

오직 "MSE(노이즈 예측 오차) + 뭉개진 Shape Loss" 두 가지만 존재합니다. SD 3.5는 텍스처를 스스로 알아서 그리는 능력이 있음(강력한 Prior). 따라서 얇은 선(고주파) 오차가 Shape Loss에서 블러로 깎여나가 증발해버리면, SD 3.5는
얇은 선은 무시하고 학습된 머리결로 채워넣으며 스케치 선을 무시해 버리는 것입니다.

---

## 2. 향후 훈련 전략: 3단계 타임스텝 연동 하이브리드 손실 (Time-guided Hybrid Loss)

단순히 가중치를 고정하지 않고, 디퓨전의 노이즈 단계($t$)에 따라 모델의 '멱살'을 잡는 **Time-dependent Guidance** 전략을 사용합니다.

*   **Step 1: Layout ($t=1000 \to 700$)**
    *   **목표:** 대략적인 구도와 덩어리 배치.
    *   **핵심:** `L_shape` (Gaussian, Kernels [11, 21, 31]) 가중치를 1.5배 높여 위치를 고정합니다.
*   **Step 2: Structure ($t=700 \to 300$) - 결정적 구간**
    *   **목표:** 땋은 머리의 선과 구조적 디테일 확정.
    *   **핵심:** `L_gradient` (Sobel) 가중치를 **3.0배**, `L_lpips`를 **1.5배** 폭증시켜 스케치를 강제로 따르게 합니다.
*   **Step 3: Texture ($t=300 \to 0$)**
    *   **목표:** 잔머리 및 머릿결 질감 마무리.
    *   **핵심:** `L_lpips` (지수적 폭증) 공식을 적용하여 질감을 극대화합니다 ($\lambda = 5 \times \exp(-4s)$). 노이즈가 없는 깨끗한 이미지에 가까워질수록 질감 제약을 강하게 줍니다.

이와 함께 스케치 이미지의 GAN 아티팩트를 제거하는 **이진화(Binarization)** 전처리를 상시 적용하여 훈련 안정성을 높였습니다. 
스케치 전처리의 핵심은 **검은색 배경(안정적인 0값)에 하얀색 스트로크(명확한 신호 255값)**가 들어오도록 통일한 것입니다. 

현재 `checkpoints2` 내부에는 이 전략이 적용된 두 가지의 하이브리드 학습 폴더가 존재합니다:
*   **`stage2_braid_hybrid`**: 1단계(Generalization)에서는 기존 스케치를 그대로 사용하고, 2단계(Specialization)에서만 하얀색 스트로크 이진화를 적용하여 디테일을 살린 모델.
*   **`stage2_braid_hybrid2`**: 1단계와 2단계 **모두** 하얀색 스트로크 이진화를 공통 적용하여 처음부터 일관된 스케치 Prior를 학습시킨 모델.

앙상블은 배제하고 이 단일 모델의 다이내믹 손실 제어에 집중합니다.

## 3. 참고 : 원본 논문(SketchHairSalon)의 최종 손실 함수

원본 논문이 디테일한 선을 잃지 않고 머리털을 생성할 수 있었던 비결은 여러 Loss를 하이브리드로 사용하였기 때문입니다. 논문의 2단계(S2I-Net: 스케치 $\to$ 실제 형태) 최종 손실 식은 다음과 같습니다. 

$$L_{S2I} = \lambda_1 L_1 + \lambda_{adv} L_{cGAN} + \lambda_{per} L_{per} + \lambda_{shape} L_{shape}$$

1.  $L_1$ : 픽셀 단위 일치(Pixel-wise Loss)
2.  $L_{cGAN}$ : 적대적 훈련을 통한 고주파/디테일 보강 (Discriminator)
3.  $L_{per}$ : VGG 네트워크를 통한 텍스처/스타일 일치 (Perceptual Loss)
4.  $L_{shape}$ : 가우시안 블러($\sigma=10.0$)를 씌워 전체 덩어리/실루엣을 맞춤 (저주파 보강)

**결론:** 결과적으로 원본 논문은 $L_{cGAN}$과 $L_{per}$라는 제약 조건을 통해 국소적인 고주파 영역을 강건하게 보존하는 동시에, 파라미터 $\sigma=10.0$의 Gaussian Blur가 적용된 $L_{shape}$을 통해 저주파 영역의 전반적인 구조와 부피감을 안정적으로 유지해냄. 이는 얇은 엣지의 특징 선택(Feature Selection)과 전체적인 매끄러운 가중치 분산 효과를 동시에 취하는 **상호 보완적인 하이브리드 손실(Hybrid Loss) 모델링 매커니즘**이다.

### GAN loss 식과 Fine-tuning 시 SD 3.5 loss 식 비교

|    분류    |    GAN 기반    |    SD 3.5 기반    |    설명    |
| :---: | :---: | :---: | :--- |
| **기본 생성 성능** | $L_1$ | **$L_{MSE}$** | • **기본 원리:** 기존 GAN은 이미지 픽셀 간의 L1 Loss를 계산했지만, Diffusion 모델 구조에 맞춰 노이즈 예측 오차인 MSE Loss로 대체함. |
| **전체적인 형태** | $L_{shape}$ | **$L_{shape}$ (Gaussian Blur)** | • **동일 적용:** 기존 논문과 동일하게 $\sigma=10.0$ 수준의 강한 가우시안 블러를 적용함. 전체적인 머리 실루엣과 볼륨감을 학습시켜 augmentation으로 인한 위치 변화에도 강건 하게 대응함. |
| **디테일한 땋은 선** | $L_{cGAN}$ | **$L_{gradient}$ (Multi-scale Sobel)** | • **대체 기법:** SD 모델에는 Discriminator가 없기 때문에 얇은 선이 뭉개지는 현상이 발생함. 이를 해결하기 위해 여러 커널 크기(3x3, 5x5 등)의 Sobel 필터로 엣지를 추출하여 세밀한 디테일 손실을 방지함. |
| **질감 및 스타일** | $L_{per}$ (VGG) | **$L_{LPIPS}$** | • **기능 보완:** 단순한 픽셀 매칭만으로는 '땋은 머리' 특유의 텍스처 패턴을 살리기 어려움. 따라서 사전 학습된 LPIPS를 도입해, 모델의 기본 Prior(생머리 등)를 억제하고 스케치와 유사한 질감을 확정적으로 생성하도록 유도함. |

### 기하학적 증강
*   회전($\pm15^\circ$), 이동($\pm10\%$), 좌우 반전을 Target, Mask, Sketch 이미지 3장에 완벽히 동기화하여 동일하게 적용합니다.
*   덕분에 모델이 스케치의 위치 변화나 구도 변화에 매우 튼튼하게 대응할 수 있습니다.

---

**결론:** 
앙상블 방식은 배제하고, 향후 학습은 **데이터 전처리(스케치 이진화)**와 지정된 타임스텝 구간에서 **Shape Loss를 부스팅하는 하이브리드 손실(Hybrid Loss)** 훈련 구조만을 반영하여 단일 모델의 성능 극대화에 집중한다.
