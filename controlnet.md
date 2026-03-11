# ControlNet-style Conditioning for SD3.5 DiT

## 프로젝트 목표

SketchHairSalon Stage-2 철학을 SD3.5 Medium DiT + ControlNet-style conditioning으로 옮긴다.

**입력 조건**
- `sketch_rgb` (3ch): colored stroke — RGB 유지, grayscale 변환 금지
- `soft_matte` (1ch): hair region prior — continuous soft value 유지, binary threshold 금지

**목표**
- Hair structure / strand flow / braid-like geometry / silhouette를 sketch와 matte에 따라 제어
- 타겟 인물에 matte + sketch를 주었을 때, matte 영역 안에서 sketch를 따르는 헤어 생성

**개발 원칙**
- 반드시 V1 → V1.5 → V2 순서로 진행
- V1이 실제로 conditioning을 먹는 것이 확인된 뒤에만 V2 진행
- region-aware noise는 V1.5 실험으로만 추가, V1 본체와 섞지 말 것

---

## SD3.5 Medium 아키텍처 (실측값)

> `inspect_sd35.py` 실행 결과. 구현 시 이 표 기준으로 설계할 것.

### 모델 구조

| 항목 | 값 |
|------|-----|
| MM-DiT blocks (N) | **24** |
| Single-DiT blocks | 0 (Large 전용) |
| image hidden_dim | **1536** (24 heads × 64) |
| joint_attention_dim | 4096 (text context — injection 대상 아님) |
| caption_projection_dim | 1536 |

### VAE

| 항목 | 값 |
|------|-----|
| spatial compression | 8× |
| latent shape (1024 입력) | (16, 128, 128) |
| scaling_factor | 1.5305 |
| patch_size | 2 |
| token grid | **64 × 64 = 4096 tokens** |

### Selected DiT Block 표

| Block | Relative position | Token grid | Control level | α (alpha) | 의도 |
|-------|-------------------|------------|---------------|-----------|------|
| block[6]  | ~25% | 64×64 (4096) | F4 (8×8, 256ch)  | 1.0 | silhouette / volume / matte prior |
| block[10] | ~40% | 64×64 (4096) | F3 (16×16, 256ch) | 0.7 | sketch + matte 균형 |
| block[13] | ~55% | 64×64 (4096) | F3 (16×16, 256ch) | 0.4 | sketch structure / flow / braid refinement |

> late 1/3 구간 (block[14] 이후)은 injection 금지 — texture prior 보존

### Control Encoder Pyramid vs Token Grid

| Level | Feature size | Channels | → Token grid projection |
|-------|-------------|----------|------------------------|
| F1 | 64×64 | 64  | bilinear → 64×64, Linear(64→1536), zero-init |
| F2 | 32×32 | 128 | bilinear → 64×64, Linear(128→1536), zero-init |
| F3 | 16×16 | 256 | bilinear → 64×64, Linear(256→1536), zero-init |
| F4 | 8×8   | 256 | bilinear → 64×64, Linear(256→1536), zero-init |

---

## V1 — Conditioning 검증 Baseline

### 목적

**딱 하나**: RGB sketch + soft matte가 SD3.5 DiT generation을 실제로 제어하는지 확인.

V1은 최종 모델이 아니라 conditioning 검증용 baseline

### 입력 채널

```python
control_input = torch.cat([sketch_rgb, soft_matte], dim=1)  # (B, 4, H, W)
```

- sketch_rgb: (B, 3, H, W) — RGB 유지
- soft_matte: (B, 1, H, W) — continuous [0,1] 유지

### Noise Policy

standard flow matching, latent 전체에 균일하게 noise:

```
GT image → VAE encode → z (clean latent)
         → sample t, noise ε
         → z_t = (1 - σ_t) * z + σ_t * ε  (flow matching)

target velocity = ε - z

condition (sketch, matte): clean 유지, noise 없음
```

### 학습 대상

| 모듈 | 학습 여부 |
|------|----------|
| ControlEncoder | ✅ trainable |
| ControlProjection × 3 | ✅ trainable |
| SD3.5 DiT transformer | ❌ frozen |
| VAE | ❌ frozen |
| Text encoders | ❌ frozen |

**금지**: LoRA, full fine-tuning, token conditioning, extra loss

### Control Encoder 구조

```
Input: (B, 4, H, W)

Stem:
  Conv3×3(4→32, stride=1) + GN + SiLU
  Conv3×3(32→32, stride=1) + GN + SiLU

Down1: Conv3×3(32→64, stride=2) + ResBlock(64)   → F1: (B, 64,  H/2,  W/2)
Down2: Conv3×3(64→128, stride=2) + ResBlock(128)  → F2: (B, 128, H/4,  W/4)
Down3: Conv3×3(128→256, stride=2) + ResBlock(256) → F3: (B, 256, H/8,  W/8)
Down4: Conv3×3(256→256, stride=2) + ResBlock(256) → F4: (B, 256, H/16, W/16)
```

원칙: image를 매번 naive resize해서 block마다 넣지 말고, encoder pyramid로 multi-scale feature를 만들 것.

### Injection 방식

```python
# 각 selected block마다 ControlProjection P_i (zero-init)
# P_i: (B, C, h, w) → bilinear(64×64) → flatten → Linear(C→1536) → (B, 4096, 1536)

x_i = x_i + alpha_i * P_i(F_i)
```

hook 방식으로 구현 (diffusers 코드 수정 없이):

```python
def hook(module, input, output):
    hs, enc_hs = output
    return (hs + alpha * ctrl_feat, enc_hs)

transformer.transformer_blocks[block_idx].register_forward_hook(hook)
```

### Loss

```
L = MSE(pred_velocity, target_velocity)   # flow matching loss only
```

금지: shape loss / gradient loss / LPIPS / perceptual / GAN

V1 목적은 "조건이 먹는가" 확인이지, 성능 최대화가 아니다.

### 평가 기준

반드시 baseline (no-control) 과 나란히 비교할 것.

| 체크 항목 | 확인 방법 |
|----------|----------|
| matte adherence | hair region이 matte 영역을 벗어나는지 |
| hair leak 감소 | matte 외부에 hair 생성 여부 |
| sketch direction/flow | 생성된 strand가 sketch 방향을 따르는지 |
| braid / crossing / strand grouping | 복잡한 구조 반응성 |
| RGB sketch color | sketch 색상이 생성에 반영되는지 |

---

## V1.5 — Region-Aware Noise

V1이 끝난 뒤에만 추가. V1 본체와 동시에 섞지 말 것.

### 목적

matte 영역에 더 강하게 noise를 주고, background는 덜 noisy하게 만들어 DiT의 장점을 활용한다.
→ region-aware forward diffusion 실험.

### 핵심 수식

```
z_t = α_t * z + σ_t * s(m) ⊙ ε

s(m) = λ_bg + (1 - λ_bg) * m

  m       : latent-resolution soft matte in [0, 1]
  λ_bg    : background noise scale, 초기값 0.1 권장
```

의미:
- hair 내부: 거의 full noise (s ≈ 1.0)
- 경계: 중간 noise
- background: 약한 noise (s ≈ 0.1)

### 구현 예시

```python
z = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
noise = torch.randn_like(z)

matte_latent = F.interpolate(
    soft_matte, size=z.shape[-2:], mode="bilinear", align_corners=False
).clamp(0.0, 1.0)

lambda_bg = 0.1
scale = lambda_bg + (1.0 - lambda_bg) * matte_latent  # (B,1,H,W)

z_t = alpha_t * z + sigma_t * (scale * noise)
```

binary threshold 금지 — soft matte 그대로 유지.

### 비교 실험 

| 실험 | 설명 | 우선순위 |
|------|------|----------|
| Full latent noise | standard flow matching | baseline |
| Soft matte + low bg noise (λ_bg=0.1) | 주력 실험 |  주력 |
| Binary matte noise | silhouette 강하나 artifact 위험 | ablation |
| λ_bg = 0 (background freeze) | 선택적 ablation | 선택 |

예상 tradeoff:
- full noise: 가장 표준적이나 구조 집중 약함
- binary matte noise: silhouette 강하지만 경계 artifact 위험
- soft matte noise: 가장 균형적일 가능성 높음

---

## V2 — Dual-Branch (V1 성공 후 진행 예정)

### 목적

single-branch를 dual-branch 구조로 확장해서 sketch 역할과 matte 역할을 명시적으로 분리한다.

### 구조

```python
# V1: 단일 encoder
control_input = cat([sketch_rgb, soft_matte], dim=1)  # 4ch

# V2: branch 분리
F_sketch = SketchEncoder(sketch_rgb)   # 3ch input
F_matte  = MatteEncoder(soft_matte)    # 1ch input

# 각 scale마다 fusion
F_fused = Conv1x1(cat([F_sketch, F_matte], dim=1))
# 초기에는 concat + 1×1 conv만 사용, attention/gated fusion은 이후
```

### Injection Policy

injection 위치는 V1과 동일 (block[6], [10], [13]).
단, 의미를 더 명확히 분리:

| Block | 의도 |
|-------|------|
| block[6]  | matte heavy (silhouette / region) |
| block[10] | sketch + matte 균형 |
| block[13] | sketch heavy (strand flow / braid) |

### Loss

V2부터 보조 loss 허용:

```
L_total = L_diff + λ_shape * L_shape + λ_grad * L_grad

초기 권장: λ_shape = 0.05~0.1,  λ_grad = 0.02~0.05
```

---

## 절대 금지 사항

### V1 단계

- 모든 DiT block에 injection
- late block (block[14] 이후) injection
- sketch grayscale화
- matte binary threshold
- LoRA 동시 적용
- token conditioning 동시 적용
- shape / gradient / perceptual loss 투입
- V1과 V1.5를 한 번에 섞는 것

---

## 버전별 한 줄 요약

| 버전 | 요약 |
|------|------|
| **V1** | 4ch single-branch ControlNet, block[6/10/13] injection (α=1.0/0.7/0.4), full latent noise, diffusion loss only, SD3.5 freeze |
| **V1.5** | V1 구조 위에 soft-matte region-aware noise (λ_bg≈0.1) 추가, 3-way ablation |
| **V2** | sketch/matte dual-branch + 동일 injection policy + optional shape/gradient auxiliary loss |
