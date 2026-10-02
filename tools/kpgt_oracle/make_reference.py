"""Dump KPGT reference tensors for the pure-PyTorch port to be verified against.

Runs in the CPU-only DGL environment (``C:/bindra/envs/kpgt``). Writes an .npz of
graph structure, model weights and staged intermediate activations, so the port
can be checked layer by layer rather than only at the output -- a single
end-to-end mismatch tells you nothing about where it went wrong.

Usage (from the isolated DGL environment; see envs/kpgt/README.md):

    KPGT_REF=../KPGT python tools/kpgt_oracle/make_reference.py reference.npz small
    KPGT_REF=../KPGT python tools/kpgt_oracle/make_reference.py reference_base.npz base
"""

import os
import sys
from pathlib import Path

# The KPGT reference checkout is not vendored: it is a separate repository with
# its own licence. Clone it and point KPGT_REF at it.
_REF = os.environ.get("KPGT_REF", "../KPGT")
if not (Path(_REF) / "src" / "model" / "light.py").exists():
    raise SystemExit(
        f"KPGT reference not found at {_REF!r}. Clone https://github.com/lihan97/KPGT "
        "and set KPGT_REF to its path."
    )
sys.path.insert(0, str(Path(_REF).resolve()))

import dgl
import numpy as np
import torch
from torch import nn

from src.data.collator import preprocess_batch_light
from src.data.featurizer import smiles_to_graph_tune
from src.model.light import LiGhTPredictor

# Small on purpose: the port is verified on architecture, not on scale. A base
# config run is added separately once this passes.
CONFIG = dict(
    d_node_feats=137,
    d_edge_feats=14,
    d_g_feats=48,
    d_fp_feats=512,
    d_md_feats=200,
    d_hpath_ratio=4,
    n_mol_layers=2,
    path_length=5,
    n_heads=4,
    n_ffn_dense_layers=2,
    input_drop=0.0,
    feat_drop=0.0,
    attn_drop=0.0,
    n_node_types=1,
)

MOLECULES = [
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccccc1",
    "C",                      # single unbonded atom: the virtual-atom path
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "N[C@@H](C)C(=O)O",
    "CC(C)(C)c1ccccc1",
    "O",                      # another bondless case
]

MAX_LENGTH = 5
N_VIRTUAL_NODES = 2


def main(out_path: str, preset: str = "small") -> None:
    global CONFIG
    if preset == "base":
        CONFIG = dict(CONFIG, d_g_feats=768, d_hpath_ratio=12, n_mol_layers=12, n_heads=12)
    torch.manual_seed(20261001)
    np.random.seed(20261001)

    model = LiGhTPredictor(**CONFIG)
    model.eval()

    graphs = [
        smiles_to_graph_tune(
            smiles,
            max_length=MAX_LENGTH,
            n_virtual_nodes=N_VIRTUAL_NODES,
            add_self_loop=True,
        )
        for smiles in MOLECULES
    ]
    batched = dgl.batch(graphs)
    batched.edata["path"][:, :] = preprocess_batch_light(
        batched.batch_num_nodes(), batched.batch_num_edges(), batched.edata["path"][:, :]
    )

    # Captured BEFORE the forward pass: LiGhT._init_path rewrites the -1e6
    # sentinel to -1 in place, so dumping afterwards records the mutated tensor
    # and makes a correct port look wrong.
    path_before_forward = batched.edata["path"].clone().numpy()

    n_graphs = len(graphs)
    fp = torch.rand(n_graphs, CONFIG["d_fp_feats"])
    md = torch.rand(n_graphs, CONFIG["d_md_feats"])

    with torch.no_grad():
        indicators = batched.ndata["vavn"]
        node_h = model.node_emb(batched.ndata["begin_end"], indicators)
        edge_h = model.edge_emb(batched.ndata["edge"], indicators)
        triplet_h0 = model.triplet_emb(node_h, edge_h, fp, md, indicators)

        # Staged inside LiGhT so a mismatch is localisable.
        light = model.model
        path_indices = batched.edata["path"]
        dist_h = light._featurize_path(batched, path_indices)
        path_h = light._init_path(batched, triplet_h0, path_indices)
        dist_attn = light.dist_attn_layer(dist_h)
        path_attn = light.path_attn_layer(path_h)

        per_layer = []
        triplet_h = triplet_h0
        for layer in light.mol_T_layers:
            triplet_h = layer(batched, triplet_h, dist_attn, path_attn)
            per_layer.append(triplet_h.clone())

        # Readout exactly as generate_fps does: virtual nodes removed, mean over
        # what remains, concatenated with the two virtual-node states.
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        readout_graph = dgl.batch(graphs)
        readout_graph.ndata["ht"] = triplet_h
        readout_graph.remove_nodes(
            np.where(indicators.detach().cpu().numpy() >= 1)[0]
        )
        readout = dgl.readout_nodes(readout_graph, "ht", op="mean")
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)

    source, destination = batched.edges()
    arrays = {
        "smiles": np.array(MOLECULES, dtype=object),
        "num_nodes_per_graph": batched.batch_num_nodes().numpy(),
        "num_edges_per_graph": batched.batch_num_edges().numpy(),
        "edge_src": source.numpy(),
        "edge_dst": destination.numpy(),
        "begin_end": batched.ndata["begin_end"].numpy(),
        "edge_feats": batched.ndata["edge"].numpy(),
        "vavn": batched.ndata["vavn"].numpy(),
        "path": path_before_forward,
        "path_after_forward": batched.edata["path"].numpy(),
        "lgp": batched.edata["lgp"].numpy(),
        "mgp": batched.edata["mgp"].numpy(),
        "vp": batched.edata["vp"].numpy(),
        "sl": batched.edata["sl"].numpy(),
        "fp": fp.numpy(),
        "md": md.numpy(),
        # staged activations
        "node_h": node_h.numpy(),
        "edge_h": edge_h.numpy(),
        "triplet_h0": triplet_h0.numpy(),
        "dist_h": dist_h.numpy(),
        "path_h": path_h.numpy(),
        "dist_attn": dist_attn.numpy(),
        "path_attn": path_attn.numpy(),
        "g_feats": g_feats.numpy(),
        "readout": readout.numpy(),
    }
    for index, states in enumerate(per_layer):
        arrays[f"triplet_h_layer{index}"] = states.numpy()
    for name, tensor in model.state_dict().items():
        arrays[f"param::{name}"] = tensor.numpy()

    np.savez_compressed(out_path, **arrays)
    print(f"wrote {out_path}")
    print(f"  {n_graphs} molecules, {batched.num_nodes()} triplet nodes, "
          f"{batched.num_edges()} edges")
    print(f"  g_feats {tuple(g_feats.shape)}  (3 x d_g_feats = {3 * CONFIG['d_g_feats']})")
    print(f"  {len(model.state_dict())} parameter tensors")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reference.npz",
         sys.argv[2] if len(sys.argv) > 2 else "small")
