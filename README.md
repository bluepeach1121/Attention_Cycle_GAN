# README: Attention CycleGAN

This repository implements an **Attention-based CycleGAN** for unpaired image-to-image translation, building upon the original [CycleGAN paper](https://arxiv.org/abs/1703.10593) 
The core idea remains to learn two mapping functions (G: X -> Y) and (F: Y -> X) that can translate between two image domains without paired training data.
However, this version introduces several changes and improvements over the original work, most notably the addition of **Self-Attention** layers and **WGAN-GP** style training with **spectral normalization**.

---

## Overview of the Original CycleGAN

The original CycleGAN paper proposed:
1. **Adversarial Loss:** Two discriminators (D_X) and (D_Y) encourage each generator to produce outputs indistinguishable from the target domain.
2. **Cycle Consistency Loss:** A forward cycle (x -> G(x) -> F(G(x))) and a backward cycle (y -> F(y) -> G(F(y))) penalize inconsistencies (L1 difference) to ensure that translating to the other domain and back reconstructs the original image.
3. **Identity Loss (optional):** Enforces (G(y) ≈ y) when feeding domain Y images into generator G, to preserve color composition (and vice versa for F).

In the original paper’s codebase:
- **Generator** used a ResNet-based or U-Net-based architecture (e.g., reflection padding, instance normalization, residual blocks).
- **Discriminator** used a PatchGAN architecture, typically with least-squares GAN (LSGAN) or vanilla GAN losses.
- **No Self-Attention** was used.
- **No Spectral Normalization** or **WGAN-GP** was included; gradient penalty was not part of the original design.

---

## Differences in This Repository

1. **Self-Attention Layers**  
   - Both the Generator (`GeneratorResNet`) and the Discriminator (`Discriminator`) include a `SelfAttention` module.  
   - This allows the model to capture **long-range dependencies** and focus on specific parts of the image, which can help improve translations in more complex scenes.  
   - The `SelfAttention` module follows the typical query-key-value mechanism, computing an attention map and then reweighting feature maps accordingly.

2. **WGAN-GP Style Adversarial Loss**  
   - Instead of LSGAN or vanilla GAN loss, the code uses a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** objective.  
   - The training script (`cycle_gan.py`) includes a `gradient_penalty` function that computes the gradient norm penalty as introduced in the WGAN-GP paper, helping stabilize the training and encourage Lipschitz continuity.  
   - As part of WGAN training, the **Generator** uses a negative sign in the loss (maximizing -D(·) corresponds to minimizing the Wasserstein distance), and the **Discriminator** is penalized for deviations from a unit gradient norm.

3. **Spectral Normalization**  
   - Spectral normalization helps stabilize training by constraining the Lipschitz constant of each layer, often improving the quality of generated images.
  
     **it was fun learning about Lipschitz constants and the role of eigenvalues in spectral_norm**

4. **Architecture & Layer Choices**  
   - **Reflection Padding** is used in the generator’s residual blocks (consistent with Johnson et al. style architectures).  
   - **Instance Normalization** is used, as in the original CycleGAN, but combined here with spectral normalization in the convolutional layers.  
   - **Dropout** is added within the Discriminator residual blocks for additional regularization.

---

## File-by-File Summary

- **`models.py`**  
  - **`GeneratorResNet`**:  
    - Builds upon a ResNet architecture with reflection padding and instance normalization.  
    - Includes a `SelfAttention` layer after the residual blocks, introducing the global attention mechanism.  
    - Uses **spectral_norm** on all convolutions.  

  - **`Discriminator`**:  
    - Patch-based architecture but includes more layers, dropout, and a **SelfAttention** block for capturing cross-region relations.  
    - Uses **spectral_norm** and an additional final convolution for 1D output.

  - **`SelfAttention`**:  
    - Standard attention with `query`, `key`, `value` convolutions of dimension (in_dim//8) (for query/key) and `in_dim` (for value).  
    - Output is weighted by a learnable parameter (gamma).

  - **`weights_init_normal`**:  
    - Initializes convolution weights with a normal distribution (µ=0, σ=0.02) following the DCGAN-style initialization.

- **`cycle_gan.py`**  
  - Main training script.  
  - Implements **WGAN-GP** by computing the gradient penalty in `gradient_penalty()`.  
  - Maintains **two discriminators** (`D_A` and `D_B`) and two generators (`G_AB` and `G_BA`), exactly as in CycleGAN.  

---

## Usage



2. **Training, (i havent trained yet)**  
   - Adjust your parameters in `cycle_gan.py` via command line or script arguments, e.g.:

     ```bash
     python cycle_gan.py \
       --dataset_name monet2photo \
       --img_height 128 \
       --img_width 128 \
       --n_epochs 20 \
       --batch_size 1 \
       --sample_interval 100 \
       --checkpoint_interval 5
     ```

   - Make sure to prepare your unpaired datasets under the folder structure:
     ```
       monet2photo/
         trainA/
         trainB/
         testA/
         testB/
     ```
   - The script expects data in `trainA`, `trainB` for domain A and B, respectively, and `testA`, `testB` for test data.

3. **Testing / Generating Samples**  
   - The script automatically generates samples every `--sample_interval` iterations to an `images/<dataset_name>` folder.  
   - Checkpoints are saved in `saved_models/<dataset_name>`, which you can load (e.g., `--epoch <N>`) to resume training.

---


---

## References

- **Original CycleGAN Paper**  
  [*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*](https://arxiv.org/abs/1703.10593),  
  *Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros*, ICCV 2017.

- **Self-Attention GAN**  
  [*Self-Attention Generative Adversarial Networks*](https://arxiv.org/abs/1805.08318),  
  *Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena*, ICML 2019.

- **WGAN-GP**  
  [*Improved Training of Wasserstein GANs*](https://arxiv.org/abs/1704.00028),  
  *Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville*, NIPS 2017.

---

**Enjoy using Attention CycleGAN!** Feel free to email me or ask questions...

