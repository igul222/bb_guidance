"""
CIFAR-10 DDPM sampling code.
All notation follows VDM (Kingma et al.) unless otherwise specified.
"""

import fire
import numpy as np
import lib.utils
import lib.unet
import os
import pytorch_fid_wrapper as pfw
import torch
import torch.nn.functional as F
import torchvision
import tqdm
from torch import nn

def main(**args):
    args = lib.utils.AttributeDict(args)
    args.setdefault('n_samples', 10_000)
    args.setdefault('n_timesteps', 100)
    lib.utils.print_args(args)

    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp64 by default and explicitly switch to fp16 where
    # appropriate.
    torch.set_default_dtype(torch.float64)

    # Load model with pretrained weights
    model, T = lib.unet.load_cifar10_unet_and_T()
    model = model.float().cuda().eval()
    lib.utils.print_model(model)

    # Calculate cosine schedule variables following "Improved Diffusion".
    unclamped_alpha_squared = torch.cos(
        float(np.pi/2) * (torch.linspace(0., 1., T+1).cuda() + 0.008) / 1.008
    ).pow(2)
    beta = torch.clamp(
        1 - (unclamped_alpha_squared[1:] / unclamped_alpha_squared[:-1]),
        max=0.999
    )
    alpha_squared = torch.cumprod(1 - beta, dim=0)
    gamma = torch.log1p(-alpha_squared) - torch.log(alpha_squared)

    def generate_samples(n_samples):
        # Implementation follows Appendix A.4 eqn 33 of VDM
        with torch.no_grad():
            z = torch.randn((n_samples, 3, 32, 32)).cuda()
            for t in range(T-1, -1, -T//args.n_timesteps):
                s = t - (T // args.n_timesteps)
                t_batch = torch.tensor([1000.*t/T]*n_samples).cuda()
                with torch.cuda.amp.autocast():
                    epsilon_pred = model(z.float(), t_batch)[:,:3]
                x0_pred = (
                    (z - epsilon_pred.double() * (1 - alpha_squared[t]).sqrt())
                    / alpha_squared[t].sqrt()
                ).clamp(-1, 1)
                if s >= 0:
                    c = -torch.expm1(gamma[s] - gamma[t])
                    z *= (1 - c) * (alpha_squared[s] / alpha_squared[t]).sqrt()
                    z += c * alpha_squared[s].sqrt() * x0_pred.double()
                    z += (c * (1-alpha_squared[s])).sqrt() * torch.randn_like(z)
            return x0_pred

    print('Generating samples for viewing...')
    samples = generate_samples(64)
    samples_uint8 = ((samples + 1) * 127.5).clamp(0, 255).byte()
    lib.utils.save_image_grid(samples_uint8.permute(0,2,3,1), 'samples.png')

    # Compute FID
    with torch.no_grad():
        print('Generating samples for FID...')
        bs = 256
        n_batches = int(np.ceil(args.n_samples / bs))
        samples = [generate_samples(bs) for _ in tqdm.tqdm(range(n_batches))]
        samples = torch.cat(samples, dim=0)[:args.n_samples]
        samples = (samples.float() + 1) / 2.
        print('Computing FID...')
        torch.set_default_dtype(torch.float32)
        cifar10 = torchvision.datasets.CIFAR10(
            os.environ['DATA_DIR'], train=True, download=True)
        X_train = (torch.tensor(cifar10.data).clone() / 255.).permute(0,3,1,2)
        pfw.set_config(device='cuda:0')
        fid = pfw.fid(samples, X_train)
        print(fid)

if __name__ == '__main__':
    fire.Fire(main)