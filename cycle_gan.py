import os
import argparse
import numpy as np
import itertools
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.autograd import grad, Variable
from tqdm import tqdm

# ------------------------------------------------------------------
#  Import your local modules
# ------------------------------------------------------------------
from models import GeneratorResNet, Discriminator, weights_init_normal
from datasets import ImageDataset
from utils import ReplayBuffer, LambdaLR


# ------------------------------------------------------------------
#  Gradient Penalty for WGAN-GP
# ------------------------------------------------------------------

def gradient_penalty(discriminator, real_imgs, fake_imgs, device, lambda_gp):
    """
    Summary of gradient_penalty Function
    Interpolates between real and fake images using random weights.
    Computes discriminator output for the interpolated images.
    Computes the gradient of the discriminator's output with respect to the interpolated images.
    Computes the L2 norm of the gradient.
    Penalizes deviations from a norm of 1, enforcing the Lipschitz constraint.
    Returns the gradient penalty, which is added to the discriminator loss.
    """
    alpha = torch.rand(real_imgs.size(0), 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_imgs)
    interpolates = alpha * real_imgs + ((1 - alpha) * fake_imgs)
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates).mean()
    grad_outputs = torch.ones_like(d_interpolates, device=device)

    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gp


# ------------------------------------------------------------------
#  Function to sample test images
# ------------------------------------------------------------------
def generate_test_examples(step, test_loader, G_AB, G_BA, device, dataset_name):
    """
    Saves a grid of test samples:
      real_A -> fake_B
      real_B -> fake_A
    """
    G_AB.eval()
    G_BA.eval()

    try:
        batch = next(iter(test_loader))
    except StopIteration:
        return  # no data in test set

    real_A = batch["A"].to(device)
    real_B = batch["B"].to(device)

    with torch.no_grad():
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

    # Grid them up
    real_A_grid = make_grid(real_A, nrow=4, normalize=True)
    fake_B_grid = make_grid(fake_B, nrow=4, normalize=True)
    real_B_grid = make_grid(real_B, nrow=4, normalize=True)
    fake_A_grid = make_grid(fake_A, nrow=4, normalize=True)

    final_grid = torch.cat([real_A_grid, fake_B_grid, real_B_grid, fake_A_grid], dim=1)

    os.makedirs(f"images/{dataset_name}", exist_ok=True)
    save_image(final_grid, f"images/{dataset_name}/test_{step}.png", normalize=False)


# ------------------------------------------------------------------
#  Main: CycleGAN
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CycleGAN in a different style")

    # Basic training params
    parser.add_argument("--epoch", type=int, default=0, help="Epoch to start from if loading a checkpoint")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of total epochs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default="monet2photo", help="Folder with trainA, trainB, testA, testB")
    parser.add_argument("--img_height", type=int, default=128)
    parser.add_argument("--img_width", type=int, default=128)
    parser.add_argument("--channels", type=int, default=3)

    # TTUR Two Time-scale Update Rule
    parser.add_argument("--g_lr", type=float, default=0.0002)
    parser.add_argument("--d_lr", type=float, default=0.0004)

    # Optim
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--decay_epoch", type=int, default=10, help="When to start decaying LR linearly")

    # Weights
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="Cycle consistency weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="Identity loss weight")
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="WGAN-GP gradient penalty weight")

    # Logging
    parser.add_argument("--sample_interval", type=int, default=100, help="Interval for saving sample test images")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="Save models every X epochs (-1 to disable)")

    # System
    parser.add_argument("--n_cpu", type=int, default=4, help="Number of data loader threads")

    opt = parser.parse_args()
    print(opt)

    # -------------------------------------------
    #  Prepare
    # -------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (opt.channels, opt.img_height, opt.img_width)

    # Generators & Discriminators
    G_AB = GeneratorResNet(input_shape, num_residual_blocks=3).to(device)
    G_BA = GeneratorResNet(input_shape, num_residual_blocks=3).to(device)
    D_A = Discriminator(input_shape).to(device)
    D_B = Discriminator(input_shape).to(device)

    # If loading checkpoint
    os.makedirs(f"saved_models/{opt.dataset_name}", exist_ok=True)
    if opt.epoch != 0:
        G_AB.load_state_dict(torch.load(f"saved_models/{opt.dataset_name}/G_AB_{opt.epoch}.pth"))
        G_BA.load_state_dict(torch.load(f"saved_models/{opt.dataset_name}/G_BA_{opt.epoch}.pth"))
        D_A.load_state_dict(torch.load(f"saved_models/{opt.dataset_name}/D_A_{opt.epoch}.pth"))
        D_B.load_state_dict(torch.load(f"saved_models/{opt.dataset_name}/D_B_{opt.epoch}.pth"))
    else:
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # Losses
    cycle_criterion = nn.L1Loss()
    identity_criterion = nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        params=itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=opt.g_lr,
        betas=(opt.b1, opt.b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))

    # LR schedulers (custom LambdaLR from utils.py)
    lr_scheduler_G = LambdaLR(optimizer_G, n_epochs=opt.n_epochs, offset=opt.epoch, decay_start_epoch=opt.decay_epoch)
    lr_scheduler_D_A = LambdaLR(optimizer_D_A, n_epochs=opt.n_epochs, offset=opt.epoch, decay_start_epoch=opt.decay_epoch)
    lr_scheduler_D_B = LambdaLR(optimizer_D_B, n_epochs=opt.n_epochs, offset=opt.epoch, decay_start_epoch=opt.decay_epoch)

    # Replay buffers
    fake_A_buffer = ReplayBuffer(device=device)
    fake_B_buffer = ReplayBuffer(device=device)

    # -------------------------------------------
    #  Datasets & Dataloaders
    # -------------------------------------------
    # Training set
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * opt.channels, (0.5,) * opt.channels),
    ]
    train_dataset = ImageDataset(root=opt.dataset_name, transforms_=transforms_, unaligned=True, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    # Test set (for generating sample images occasionally)
    test_dataset = ImageDataset(root=opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    # -------------------------------------------
    #  Training
    # -------------------------------------------
    total_steps = 0
    for epoch in range(opt.epoch, opt.n_epochs):
        G_AB.train()
        G_BA.train()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{opt.n_epochs}", unit="batch") as pbar:
            for i, batch in enumerate(train_loader):
                total_steps += 1

                real_A = batch["A"].to(device)
                real_B = batch["B"].to(device)

                # ---------------------
                #  Train Generators
                # ---------------------
                optimizer_G.zero_grad()

                # Identity losses
                id_A = G_BA(real_A)
                id_B = G_AB(real_B)
                loss_id_A = identity_criterion(id_A, real_A)
                loss_id_B = identity_criterion(id_B, real_B)
                loss_identity = (loss_id_A + loss_id_B) / 2 * opt.lambda_id

                # GAN losses (WGAN)
                fake_B = G_AB(real_A)
                loss_GAN_AB = -torch.mean(D_B(fake_B))
                fake_A = G_BA(real_B)
                loss_GAN_BA = -torch.mean(D_A(fake_A))
                loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

                # Cycle consistency
                recov_A = G_BA(fake_B)
                recov_B = G_AB(fake_A)
                loss_cycle_A = cycle_criterion(recov_A, real_A)
                loss_cycle_B = cycle_criterion(recov_B, real_B)
                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2 * opt.lambda_cyc

                # Total generator loss
                g_loss = loss_GAN + loss_cycle + loss_identity
                g_loss.backward()
                optimizer_G.step()

                # ------------------------
                #  Train Discriminator A
                # ------------------------
                optimizer_D_A.zero_grad()
                fake_A_ = fake_A_buffer.push_and_pop(fake_A)
                gp_A = gradient_penalty(D_A, real_A, fake_A_, device, opt.lambda_gp)
                d_loss_A = (torch.mean(D_A(fake_A_)) - torch.mean(D_A(real_A))) + gp_A
                d_loss_A.backward()
                optimizer_D_A.step()

                # ------------------------
                #  Train Discriminator B
                # ------------------------
                optimizer_D_B.zero_grad()
                fake_B_ = fake_B_buffer.push_and_pop(fake_B)
                gp_B = gradient_penalty(D_B, real_B, fake_B_, device, opt.lambda_gp)
                d_loss_B = (torch.mean(D_B(fake_B_)) - torch.mean(D_B(real_B))) + gp_B
                d_loss_B.backward()
                optimizer_D_B.step()

                d_loss = (d_loss_A + d_loss_B) / 2

                # ------------------------
                #  Logging
                # ------------------------
                pbar.set_postfix({
                    "D_loss": f"{d_loss.item():.4f}",
                    "G_loss": f"{g_loss.item():.4f}"
                })
                pbar.update(1)

                # Save test images periodically
                if total_steps % opt.sample_interval == 0:
                    generate_test_examples(total_steps, test_loader, G_AB, G_BA, device, opt.dataset_name)

            # Decay learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

        # Save model checkpoints if needed
        if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
            torch.save(G_AB.state_dict(), f"saved_models/{opt.dataset_name}/G_AB_{epoch+1}.pth")
            torch.save(G_BA.state_dict(), f"saved_models/{opt.dataset_name}/G_BA_{epoch+1}.pth")
            torch.save(D_A.state_dict(), f"saved_models/{opt.dataset_name}/D_A_{epoch+1}.pth")
            torch.save(D_B.state_dict(), f"saved_models/{opt.dataset_name}/D_B_{epoch+1}.pth")


if __name__ == "__main__":
    main()
