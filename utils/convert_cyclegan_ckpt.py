#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert an official-layout CycleGAN generator pair into the host format.

`FlowCycle.load_cyclegan` wants a single file holding

    {"G_T1toFA": <state_dict>, "G_FAtoT1": <state_dict>, "epoch": ...}

whose keys are those of `model.backbone.ResnetGenerator` — i.e. grouped by
`head. / down. / res. / up. / tail.`.  The reference CycleGAN implementation
(and every re-implementation that follows it) instead stores ONE flat
`nn.Sequential` called `model`, so the same tensors arrive under `model.<N>`.

The index map below is fixed by the reference architecture and does not depend
on n_blocks; only the residual range moves:

    model.0   ReflectionPad2d(3)          -- no params
    model.1   Conv2d(in, ngf, 7)          -> head.1
    model.2   InstanceNorm2d              -- affine=False, no params
    model.3   ReLU
    model.4   Conv2d(ngf, ngf*2, 3, 2)    -> down.0
    model.7   Conv2d(ngf*2, ngf*4, 3, 2)  -> down.3
    model.10 .. model.{10+n_blocks-1}     -> res.0 .. res.{n_blocks-1}
              each carries .block.1 and .block.5 (the two 3x3 convs)
    model.{B+0} ConvTranspose2d           -> up.0
    model.{B+3} ConvTranspose2d           -> up.3
    model.{B+7} Conv2d(ngf, out, 7)       -> tail.1

Residual blocks are matched by NAME (`.block.` or `.conv_block.`) rather than by
index, so a re-implementation that renames the inner Sequential still converts.

    python -m utils.convert_cyclegan_ckpt --a2b G_AB.pt --b2a G_BA.pt \
        --out exps/checkpoints/h2z_host/last.pth
"""
import argparse
import os
import re
import sys

import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model.backbone import ResnetGenerator                               # noqa: E402


def _unwrap(obj):
    """Accept a state_dict, a wrapped checkpoint, or a pickled Module."""
    if hasattr(obj, "state_dict") and not isinstance(obj, dict):
        return obj.state_dict()
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "net", "g", "G"):
            v = obj.get(k)
            if isinstance(v, dict) and any(hasattr(t, "shape") for t in v.values()):
                return v
        return obj
    raise TypeError(f"cannot read a state_dict out of {type(obj)}")


def flat_to_grouped(sd):
    """`model.<N>.*`  ->  `head./down./res./up./tail.`  Returns (dict, n_blocks)."""
    res_ids = sorted({int(m.group(1)) for k in sd
                      if (m := re.match(r"model\.(\d+)\.(conv_)?block\.", k))})
    if not res_ids:
        raise ValueError("no residual blocks found — is this an official-layout "
                         f"CycleGAN generator?  keys start with {list(sd)[:4]}")
    if res_ids != list(range(res_ids[0], res_ids[0] + len(res_ids))):
        raise ValueError(f"residual block indices are not contiguous: {res_ids}")
    n_blocks, first, last = len(res_ids), res_ids[0], res_ids[-1]
    if first != 10:
        raise ValueError(f"expected the first residual block at model.10, got {first}")

    # up/tail sit immediately after the residual stack
    b = last + 1
    direct = {"model.1": "head.1", "model.4": "down.0", "model.7": "down.3",
              f"model.{b}": "up.0", f"model.{b+3}": "up.3", f"model.{b+7}": "tail.1"}

    out, unmapped = {}, []
    for k, v in sd.items():
        stem, _, leaf = k.rpartition(".")            # "model.1", "weight"
        if stem in direct:
            out[f"{direct[stem]}.{leaf}"] = v
            continue
        m = re.match(r"model\.(\d+)\.(?:conv_)?block\.(\d+)$", stem)
        if m:
            out[f"res.{int(m.group(1)) - first}.block.{m.group(2)}.{leaf}"] = v
            continue
        unmapped.append(k)
    if unmapped:
        raise ValueError(f"unmapped keys: {unmapped}")
    return out, n_blocks


def convert(path):
    sd = _unwrap(torch.load(path, map_location="cpu"))
    if any(k.startswith(("head.", "down.", "res.")) for k in sd):
        n_blocks = len({int(m.group(1)) for k in sd
                        if (m := re.match(r"res\.(\d+)\.", k))})
        return sd, n_blocks
    return flat_to_grouped(sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2b", required=True, help="generator A->B (horse->zebra)")
    ap.add_argument("--b2a", required=True, help="generator B->A (zebra->horse)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img_ch", type=int, default=3)
    ap.add_argument("--ngf", type=int, default=64)
    a = ap.parse_args()

    g_ab, nb_ab = convert(a.a2b)
    g_ba, nb_ba = convert(a.b2a)
    if nb_ab != nb_ba:
        raise SystemExit(f"the two generators disagree on n_blocks: {nb_ab} vs {nb_ba}")

    # Load into the real module with strict=True.  This is the actual test: it
    # catches a wrong ngf, a wrong channel count and any key we failed to map,
    # which a bare dict comparison would not.
    for name, sd in [("A->B", g_ab), ("B->A", g_ba)]:
        g = ResnetGenerator(a.img_ch, a.img_ch, a.ngf, nb_ab)
        g.load_state_dict(sd, strict=True)
        print(f"  {name}: loaded strict=True into ResnetGenerator("
              f"in={a.img_ch}, ngf={a.ngf}, n_blocks={nb_ab})")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"G_T1toFA": g_ab, "G_FAtoT1": g_ba, "epoch": -1,
                "n_blocks": nb_ab, "img_ch": a.img_ch, "ngf": a.ngf,
                "source": [os.path.abspath(a.a2b), os.path.abspath(a.b2a)]}, a.out)
    print(f"wrote {a.out}  (n_blocks={nb_ab}, img_ch={a.img_ch})")


if __name__ == "__main__":
    main()
