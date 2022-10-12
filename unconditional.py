"""
CIFAR-10 DDPM sampling code.
All notation follows VDM (Kingma et al.) unless otherwise specified.
"""

import fire
import numpy as np
import lib.utils
import lib.unet
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

def main(**args):
    args = lib.utils.AttributeDict(args)
    args.setdefault('n_samples', 32)
    args.setdefault('sampling_timesteps', 4000)
    lib.utils.print_args(args)

    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp64 by default and explicitly switch to fp16 where
    # appropriate.
    torch.set_default_dtype(torch.float64)

    # Load model with pretrained weights
    model, T = lib.unet.load_cifar10_unet_and_T()
    model = model.float().cuda()
    lib.utils.print_model(model)

    # Calculate diffusion schedule variables. `beta' is from the notation of
    # DDPM (Ho et al.)
    unclamped_alpha_squared = torch.cos(
        float(np.pi/2) * (torch.linspace(0., 1., T+1).cuda() + 0.008) / 1.008
    ).pow(2)
    beta = torch.clamp(
        1 - (unclamped_alpha_squared[1:] / unclamped_alpha_squared[:-1]),
        max=0.999
    )
    alpha_squared = torch.cumprod(1 - beta, dim=0)
    gamma = torch.log1p(-alpha_squared) - torch.log(alpha_squared)

    # Generate samples. Code follows Appendix A.4 eqn 33 of VDM.
    with torch.no_grad():
        z = torch.randn((args.n_samples, 3, 32, 32)).cuda()
        for t in tqdm.tqdm(range(T-1, -1, -T//args.sampling_timesteps)):
            s = t - (T // args.sampling_timesteps)
            t_batch = torch.tensor([1000.*t/T]*args.n_samples).cuda()
            with torch.cuda.amp.autocast():
                epsilon_pred = model(z.float(), t_batch)[:,:3]
            x0_pred = (
                (z - epsilon_pred.double() * (1 - alpha_squared[t]).sqrt())
                / alpha_squared[t].sqrt()
            ).clamp(-1, 1)
            if s >= 0:
                c = -torch.expm1(gamma[s] - gamma[t])
                z *= (1 - c) * alpha_squared[s].sqrt() / alpha_squared[t].sqrt()
                z += c * (alpha_squared[s].sqrt() * x0_pred.double())
                z += (c * (1 - alpha_squared[s])).sqrt() * torch.randn_like(z)

    x0_pred = ((x0_pred + 1) * 127.5).clamp(0, 255).byte()
    lib.utils.save_image_grid(x0_pred.permute(0,2,3,1), 'samples.png')

if __name__ == '__main__':
    fire.Fire(main)