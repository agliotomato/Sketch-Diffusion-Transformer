# Sketch → Hair Generation Project  
## Meeting Summary & Implementation Plan

---

# 1. Project Objective

Train a model that generates **hair appearance** conditioned on **hair structure input**.

Core task:

```
(sketch + matte) → hair
```

Inputs:
- **Sketch (RGB)**: encodes hair structure (braid strands, flow lines)
- **Matte (1 channel)**: defines spatial region where hair should exist

Target:

```
hair_target = image × matte
```

Only the **hair region** is used as supervision.

---

# 2. Problem Reframing

The original idea attempted:

```
sketch → full image generation
```

This approach is impractical because the model must learn:

- face generation
- skin appearance
- lighting
- pose
- background
- hair structure

With a dataset of ~4000 images this task is too large and leads to **conditioning collapse**, where the model ignores the sketch and produces generic hair.

Therefore the problem is reduced to:

```
(sketch + matte) → hair only
```

This isolates the learning objective to:

- hair topology
- strand geometry
- hair texture
- shading

---

# 3. Dataset

| Category | Count |
|--------|------|
| braid | 1000 |
| unbraid | 3000 |
| total | 4000 |

Characteristics:

**unbraid**
- simple structure
- good for learning hair texture and shading

**braid**
- complex topology
- critical for structure learning

---

# 4. Training Strategy

Training proceeds in two stages.

## Stage 1 — Pretraining (unbraid)

Purpose:

- learn hair appearance prior
- learn shading and texture

Training objective:

```
(sketch + matte) → hair
dataset: unbraid
```

---

## Stage 2 — Fine-tuning (braid)

Purpose:

- learn braid topology
- enforce sketch-controlled structure

Training objective:

```
(sketch + matte) → braid hair
dataset: braid
```

Important:

- braid samples must not be overwhelmed by unbraid prior
- braid-focused sampling is recommended

---

# 5. Model Objective

Diffusion training predicts noise conditioned on sketch and matte:

```
εθ(z_t | sketch, matte)
```

Target:

```
hair_target
```

Initial loss:

```
L = L1 + LPIPS
```

More complex losses can be introduced after the baseline works.

---

# 6. Data Augmentation Strategy

All augmentations must preserve **sketch–structure correspondence**.

## 6.1 Geometry Augmentation

Apply identical transforms to:

- sketch
- matte
- hair target

Operations:

- horizontal flip
- rotation ±10°
- scale 0.9–1.1
- small translation

Purpose:

- improve spatial generalization
- avoid positional overfitting

---

## 6.2 Sketch Color Randomization

Colored strokes represent **strand separation**, not hair color.

Risk:

If colors are fixed, the model may learn shortcuts:

```
specific color → specific appearance
```

Solution:

- randomly reassign stroke colors per sample
- maintain distinct colors between strands

Result:

```
color becomes a structural cue
not an appearance signal
```

---

## 6.3 Sketch Thickness Jitter

Problem:

Thin strokes disappear during downsampling.

Augmentations:

- random dilation
- random erosion
- slight blur

Effect:

- robustness to line thickness
- stable topology learning

---

## 6.4 Matte Boundary Perturbation

Problem:

Models overfit to perfectly aligned masks.

Augmentations:

- small dilation
- small erosion
- edge blur

Effect:

- boundary robustness
- smoother blending later

---

## 6.5 Hair Appearance Augmentation

Applied only to the hair target image.

Operations:

- brightness jitter
- contrast jitter
- saturation jitter
- color temperature shift

Purpose:

- decouple structure from appearance
- increase diversity

---

# 7. Augmentations to Avoid

These operations break topology labels and must not be used:

- large rotations
- perspective warp
- elastic deformation
- random cutout across braid
- independent transforms of sketch and matte

---

# 8. Post-Generation Pipeline

Hair generation is only the first stage.

Step 1 — Hair Generation

```
(sketch + matte) → generated hair
```

Step 2 — Composite

```
face image + generated hair + mask
```

Step 3 — Diffusion Refinement

```
diffusion inpainting
```

Purpose:

- correct boundaries
- harmonize lighting
- blend texture

---

# 9. Evaluation Objective

The primary metric is **structure fidelity**:

```
Does the generated hair follow the sketch topology?
```

Not:

```
Does the image look generally realistic?
```

Success means:

- braid crossings preserved
- strand separation maintained
- structure controlled by sketch

---

# 10. Immediate Implementation Tasks

1. Dataset preprocessing

```
hair_target = image × matte
```

2. Implement augmentation pipeline

- geometry transforms
- sketch color randomization
- sketch thickness jitter
- matte boundary perturbation
- hair appearance jitter

3. Train **unbraid pretraining model**

4. Fine-tune on **braid dataset**

5. Evaluate **structure fidelity**

---

# Final Summary

The project focuses on **structure-conditioned hair generation**.

```
(sketch + matte) → hair
```

Key principles:

- reduce problem complexity
- isolate hair generation task
- preserve sketch–structure mapping
- enforce topology learning before realism