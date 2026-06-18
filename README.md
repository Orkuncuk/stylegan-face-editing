# StyleGAN2-ADA Face Attribute Editing

Two-part project exploring semantic face editing in the latent space of a pre-trained StyleGAN2-ADA generator. Part 1 uses classical direction-finding methods (SVM, mean difference). Part 2 trains a conditional diffusion model on the same latent space with Classifier-Free Guidance.

Both notebooks run on Kaggle with a T4 GPU.

**Kaggle notebooks:** [Part 1 — SVM](Yhttps://www.kaggle.com/code/orkunakta/stylegan2-latent-direction-svm) · [Part 2 — Diffusion](https://www.kaggle.com/code/orkunakta/stylegan2-latent-diffusion-cfg)

---

## Repository Structure

```
.
├── 01_svm_direction_finding.ipynb   # Part 1: SVM-based latent direction finding
├── 02_diffusion_cfg_editing.ipynb   # Part 2: Conditional latent diffusion + CFG
├── requirements.txt
└── README.md
```

---

## Part 1 — SVM Direction Finding

**Notebook:** `01_svm_direction_finding.ipynb`

### What it does

Trains a ResNet18 multi-label classifier on 30K CelebA images to detect facial attributes (Smiling, Male, Eyeglasses). Generates 5K faces from the StyleGAN2-ADA generator, labels them with the classifier, and computes semantic editing directions in W-space using two methods:

- **Mean Difference** — subtracts the centroid of the negative class from the positive class centroid
- **LinearSVC** — finds the hyperplane that best separates the two classes; uses its normal vector as the direction

Editing is applied as `w_new = w + alpha * direction` in both Z-space and W-space.

### Pipeline

```
Random z  →  G.mapping()  →  W-space
                                 │
                      G.synthesis()  →  image  →  ResNet18  →  label
                                 │
                         SVM / Mean Diff  →  direction vector n̂
                                 │
                      w_new = w + α · n̂  →  edited face
```

### Key results

- W-space manipulation preserves identity better than Z-space
- SVM directions are slightly more disentangled than mean difference
- Score-weighted direction (using classifier probabilities instead of binary labels) gives cleaner edits

---

## Part 2 — Conditional Latent Diffusion + CFG

**Notebook:** `02_diffusion_cfg_editing.ipynb`

### What it does

Trains an MLP-based diffusion model directly on 20K W-space latent vectors. The model learns the conditional distribution of W given facial attributes (Male, Eyeglasses, Smiling). At inference time, Classifier-Free Guidance is used to extract attribute directions without retraining.

### Why diffusion instead of SVM

SVM finds a linear boundary between two classes. The diffusion model learns the full distribution of W-space and can represent non-linear attribute structure. It also supports multi-attribute conditioning simultaneously.

### Architecture

```
Input: (noisy_w, timestep t, condition c)

noisy_w  →  Linear(512 → 2048)  ─┐
t        →  Sinusoidal embeddings  ├──  sum  →  3× Residual blocks  →  Linear(2048 → 512)
c        →  MLP(3 → 2048)         ─┘
```

Condition `c` is a 3-dim binary vector `[Male, Eyeglasses, Smiling]`. During training, 10% of samples use a null token (`c = -1`) to enable Classifier-Free Guidance at inference.

### Direction extraction

```python
# CFG-guided direction averaged over multiple timesteps
guided_pos = eps_uncond + scale * (eps_pos - eps_uncond)
guided_neg = eps_uncond + scale * (eps_neg - eps_uncond)
direction  = mean(guided_neg - guided_pos)  # over t = [100, 300, 500, 700, 900]
```

Using multiple timesteps instead of only `t=999` (pure noise) gives more stable directions.

### Training details

| Setting | Value |
|---|---|
| Dataset | 20K StyleGAN2-ADA W vectors (FFHQ) |
| Hidden dim | 2048 |
| Timesteps | 1000 (linear schedule) |
| Epochs | 150 |
| Optimizer | Adam, lr=1e-4 |
| LR schedule | Cosine annealing with warm restarts (T_0=50, T_mult=2) |
| EMA decay | 0.9999 |
| Grad clipping | max_norm=1.0 |
| CFG dropout | 10% |

---

## Setup

These notebooks are designed to run on **Kaggle** with the CelebA dataset.

```bash
kaggle datasets download -d jessicali9530/celeba-dataset
```

To run locally:

```bash
pip install -r requirements.txt
git clone https://github.com/NVlabs/stylegan2-ada-pytorch
```

Pre-trained FFHQ model is downloaded automatically at runtime:

```python
network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
```

---

## Tech Stack

- **Generator:** StyleGAN2-ADA (NVIDIA, pre-trained on FFHQ)
- **Classifier:** ResNet18 fine-tuned on CelebA
- **Direction finding:** Scikit-learn LinearSVC, mean-difference, score-weighted averaging
- **Diffusion model:** Custom MLP (hidden_dim=2048) with sinusoidal timestep embeddings
- **Guidance:** Classifier-Free Guidance (CFG)
- **Framework:** PyTorch
- **Platform:** Kaggle (NVIDIA T4 GPU)