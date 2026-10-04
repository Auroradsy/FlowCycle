#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FlowCycle — CycleGAN rewired so that a single dense flow IS the bridge.

An earlier attempt tapped a 16-D vector off the bottleneck and
re-injected it by FiLM.  A16 killed it: `feat` (256x28x28) still reached the
decoder untouched, so the latent was decoration (u-shuffle changed SSIM by
0.000) and the flow contributed nothing to the translation --- `G_A2B` was
already a complete T1->FA translator on its own.

Here the two ResnetGenerators are *split* instead of augmented:

      E_A --> z (A's rep, 256x28x28) --> D_A --> T1        [self]
                    |  ^
                    f  f^-1
                    v  |
      E_B --> u (B's rep, 256x28x28) --> D_B --> FA        [self]

      T1->FA :  T1 -E_A-> z -f-> u -D_B-> FA
      FA->T1 :  FA -E_B-> u -f^-1-> z -D_A-> T1

Three properties this buys, all of them checkable:

  1. Parameter-neutral.  E_A+D_B is exactly G_A2B and E_B+D_A is exactly G_B2A,
     so the host costs the same as plain CycleGAN; the flow adds 1.84 M (12 %).
  2. `f` is the EXACT identity at init (SpatialActNorm log_scale=bias=0,
     SpatialCoupling last conv zero-init), so a freshly warm-started model
     reproduces plain CycleGAN bit-for-bit on both cross directions.  See
     `FlowCycle.load_cyclegan` + train.py --check_init.
  3. No bypass.  D_B only ever sees E_B(FA) or f(E_A(T1)); there is no spatial
     path around the flow, so u-shuffle must destroy the output by construction.

The bottleneck is tapped BEFORE the last ReLU of `down`, so the code is
sign-free --- otherwise `feat >= 0` and the flow's (signed) output would be
off-distribution for the decoder no matter how well it is trained.
"""
import torch
import torch.nn as nn

from .backbone import ResnetGenerator, PatchDiscriminator, init_weights
from .flow import SpatialFlow


# ---------------------------------------------------------------------------
# the two halves of a ResnetGenerator
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """`head` + `down` of a ResnetGenerator -> (B, ngf*4, 28, 28).

    Submodule names match the host so a plain-CycleGAN state_dict loads by
    prefix with strict=True.  `pre_relu` drops the trailing ReLU of `down`
    (indices of the parameterised layers are unchanged: InstanceNorm here is
    affine=False, ReLU has no parameters).
    """

    def __init__(self, in_ch=1, ngf=64, pre_relu=True):
        super().__init__()
        g = ResnetGenerator(in_ch=in_ch, out_ch=1, ngf=ngf, n_blocks=1)
        self.head = g.head
        self.down = nn.Sequential(*list(g.down.children())[:-1]) if pre_relu else g.down

    def forward(self, x):
        return self.down(self.head(x))


class Decoder(nn.Module):
    """`res` + `up` + `tail` of a ResnetGenerator -> image in [-1, 1]."""

    def __init__(self, out_ch=1, ngf=64, n_blocks=6):
        super().__init__()
        g = ResnetGenerator(in_ch=1, out_ch=out_ch, ngf=ngf, n_blocks=n_blocks)
        self.res, self.up, self.tail = g.res, g.up, g.tail

    def forward(self, z):
        return self.tail(self.up(self.res(z)))


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class FlowCycle(nn.Module):
    """Two encoder/decoder pairs from CycleGAN, bridged by ONE dense conv flow."""

    def __init__(self, ngf=64, n_blocks=6, n_flow=4, flow_hidden=128, pre_relu=True,
                 img_ch=1, shared_dec=False):
        super().__init__()
        ch = ngf * 4
        self.img_ch = img_ch
        self.shared_dec = bool(shared_dec)
        self.enc_A = init_weights(Encoder(img_ch, ngf, pre_relu))
        self.enc_B = init_weights(Encoder(img_ch, ngf, pre_relu))
        self.dec_A = init_weights(Decoder(img_ch, ngf, n_blocks))
        # shared_dec: ONE decoder for both modalities. dec_B is dec_A, so every path
        # that names either of them is unchanged, but the state dict carries the same
        # tensors under both prefixes and a reader that rebuilds two decoders from it
        # gets two identical copies, i.e. the same function.
        self.dec_B = self.dec_A if self.shared_dec else init_weights(Decoder(img_ch, ngf, n_blocks))
        # NOTE: init_weights must NOT touch the flow — it would overwrite the
        # zero-init that makes f the exact identity at step 0.
        self.flow = SpatialFlow(n_flow, ch, flow_hidden)
        self.ch, self.n_blocks = ch, n_flow

    # -- the four paths ----------------------------------------------------
    def a_to_b(self, z_A):
        """A-space -> B-space (one forward application of f)."""
        u, _ = self.flow(z_A)
        return u

    def b_to_a(self, u_B):
        z, _ = self.flow.inverse(u_B)
        return z

    def self_A(self, x_A):
        return self.dec_A(self.enc_A(x_A))

    def self_B(self, x_B):
        return self.dec_B(self.enc_B(x_B))

    def cross_A2B(self, x_A):
        return self.dec_B(self.a_to_b(self.enc_A(x_A)))

    def cross_B2A(self, x_B):
        return self.dec_A(self.b_to_a(self.enc_B(x_B)))

    # -- the morph path ----------------------------------------------------
    def walk(self, z, inverse=False):
        """All L+1 states along the flow: [z, b1, ..., bL] (bL == f(z))."""
        states, x = [z], z
        rng = reversed(range(self.n_blocks)) if inverse else range(self.n_blocks)
        for b in rng:
            if inverse:
                x, _ = self.flow.layers[2 * b + 1].inverse(x)
                x, _ = self.flow.layers[2 * b].inverse(x)
            else:
                x, _ = self.flow.layers[2 * b](x)
                x, _ = self.flow.layers[2 * b + 1](x)
            states.append(x)
        return states

    # -- warm start --------------------------------------------------------
    def load_cyclegan(self, path, map_location="cpu"):
        """Initialise the four halves from a trained plain-CycleGAN checkpoint.

        E_A + D_B are the two halves of G_T1toFA, E_B + D_A of G_FAtoT1, so with
        f still at its identity init the cross paths reproduce the host exactly.
        """
        if self.shared_dec:
            raise RuntimeError('a shared decoder cannot be warm-started from a CycleGAN: '
                               'the host has one decoder per direction and the second load '
                               'would silently overwrite the first')
        ck = torch.load(path, map_location=map_location)
        A2B, B2A = ck["G_T1toFA"], ck["G_FAtoT1"]

        def take(sd, prefixes, dst):
            sub = {k: v for k, v in sd.items() if k.split(".")[0] in prefixes}
            dst.load_state_dict(sub, strict=True)

        take(A2B, {"head", "down"}, self.enc_A)
        take(A2B, {"res", "up", "tail"}, self.dec_B)
        take(B2A, {"head", "down"}, self.enc_B)
        take(B2A, {"res", "up", "tail"}, self.dec_A)
        return ck.get("epoch", None)

    # -- diagnostics -------------------------------------------------------
    @torch.no_grad()
    def flow_work(self, z):
        """||f(z) - z|| / ||z||.  ~0 means the flow collapsed to the identity."""
        u = self.a_to_b(z)
        return float(((u - z).flatten(1).norm(dim=1) /
                      (z.flatten(1).norm(dim=1) + 1e-8)).mean())

    @torch.no_grad()
    def latent_gap(self, z_A, u_B):
        """||f(z_A) - E_B(x_B)|| / ||E_B(x_B)||.  Reported, never optimised."""
        u = self.a_to_b(z_A)
        return float(((u - u_B).flatten(1).norm(dim=1) /
                      (u_B.flatten(1).norm(dim=1) + 1e-8)).mean())


def make_discriminators(ndf=64, mix=False, mix_b=False, img_ch=1):
    """D_FA / D_T1 (host, unchanged) and optionally D_mix for the morph path.

    D_mix is trained on real T1 UNION real FA, so "realistic" for it means
    "a real brain slice of either modality" — exactly the supervision an
    intermediate flow state needs, and it requires no ground-truth mid-frames.
    """
    d = {"FA": init_weights(PatchDiscriminator(img_ch, ndf)),
         "T1": init_weights(PatchDiscriminator(img_ch, ndf))}
    if mix:
        d["mix"] = init_weights(PatchDiscriminator(img_ch, ndf))
    if mix_b:
        # A second critic for the B->A leg.  Sharing one critic across both
        # legs makes the generator satisfy two OPPOSITE endpoint targets at
        # once, and the cheapest joint solution is a path that barely moves
        # (measured: forward 0.183 -> 0.042, backward 0.120 -> 0.023).
        d["mix_b"] = init_weights(PatchDiscriminator(img_ch, ndf))
    return d


if __name__ == "__main__":
    m = FlowCycle()
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    host = n(m.enc_A) + n(m.enc_B) + n(m.dec_A) + n(m.dec_B)
    print(f"host (= 2 x ResnetGenerator): {host/1e6:.2f} M")
    print(f"flow                        : {n(m.flow)/1e6:.2f} M "
          f"({100*n(m.flow)/host:.0f} % of host)")
    x = torch.randn(2, 1, 112, 112)
    z = m.enc_A(x)
    print("bottleneck:", tuple(z.shape))
    print("f == identity at init:", torch.allclose(m.a_to_b(z), z, atol=1e-6))
    print("f invertible          :", torch.allclose(m.b_to_a(m.a_to_b(z)), z, atol=1e-4))
    print("walk states           :", len(m.walk(z)))
