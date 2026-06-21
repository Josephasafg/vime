"""Convert gpt-oss-20b per-expert BF16 → fused HF format.

Reads: gpt-oss-20b-bf16/ (per-expert split: experts.{e}.gate_proj.weight etc.)
Writes: gpt-oss-20b-bf16-v2/ (fused: experts.gate_up_proj [E, hidden, 2*ffn] etc.)

The fused format matches vLLM _load_weights_other and transformers GptOssExperts.
"""
import argparse, json, os, shutil
from collections import defaultdict
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Copy config and non-weight files
    for fname in os.listdir(input_dir):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        src, dst = os.path.join(input_dir, fname), os.path.join(output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Build full key→file map
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shards = sorted(set(weight_map.values()))

    # Bucket per-expert keys by layer, accumulate rest directly
    gate_proj_w  = defaultdict(dict)   # [layer_prefix][e] = tensor
    up_proj_w    = defaultdict(dict)
    down_proj_w  = defaultdict(dict)
    gate_proj_b  = defaultdict(dict)
    up_proj_b    = defaultdict(dict)
    down_proj_b  = defaultdict(dict)
    other = {}

    for shard in shards:
        shard_path = os.path.join(input_dir, shard)
        print(f"Reading {shard}...")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                # e.g. model.layers.0.mlp.experts.3.gate_proj.weight
                if ".mlp.experts." in key:
                    parts = key.split(".mlp.experts.")
                    prefix = parts[0]           # model.layers.0
                    rest   = parts[1]           # 3.gate_proj.weight
                    e_str, sub = rest.split(".", 1)
                    e = int(e_str)
                    if sub == "gate_proj.weight":  gate_proj_w[prefix][e] = t
                    elif sub == "up_proj.weight":  up_proj_w[prefix][e]   = t
                    elif sub == "down_proj.weight":down_proj_w[prefix][e] = t
                    elif sub == "gate_proj.bias":  gate_proj_b[prefix][e] = t
                    elif sub == "up_proj.bias":    up_proj_b[prefix][e]   = t
                    elif sub == "down_proj.bias":  down_proj_b[prefix][e] = t
                    else:
                        other[key] = t
                else:
                    other[key] = t

    output_tensors = {}

    # Fuse per-expert weights
    layers = sorted(gate_proj_w.keys())
    for prefix in layers:
        E = len(gate_proj_w[prefix])
        # gate_up_proj [E, hidden, 2*ffn]
        # Original storage: gate_proj.weight [ffn, hidden] (Linear out×in)
        # HF stores gate_up_proj as [hidden, 2*ffn] after transpose, interleaved g/u
        gate_list = [gate_proj_w[prefix][e] for e in range(E)]  # each [ffn, hidden]
        up_list   = [up_proj_w[prefix][e]   for e in range(E)]  # each [ffn, hidden]
        ffn, hidden = gate_list[0].shape

        # Interleave gate and up rows: [g0,u0,g1,u1,...] shape [2*ffn, hidden] per expert
        interleaved = []
        for e in range(E):
            gu = torch.empty(2 * ffn, hidden, dtype=gate_list[e].dtype)
            gu[0::2] = gate_list[e]
            gu[1::2] = up_list[e]
            interleaved.append(gu)
        # Stack → [E, 2*ffn, hidden], then transpose → [E, hidden, 2*ffn]
        gate_up = torch.stack(interleaved, dim=0).transpose(1, 2).contiguous()
        output_tensors[f"{prefix}.mlp.experts.gate_up_proj"] = gate_up

        # down_proj [E, ffn, hidden] (HF shape: [E, intermediate, hidden])
        # per-expert down_proj.weight is [ffn, hidden] (or [hidden, ffn] depending on storage)
        # From preprocess_gpt_oss.py comment: dequantized[e] is [hidden, intermediate]
        # → stored as experts.{e}.down_proj.weight [hidden, intermediate]
        # HF wants [E, intermediate, hidden] → transpose each expert weight
        down_list = [down_proj_w[prefix][e] for e in range(E)]  # each [2880, 2880] (square)
        # For non-square we'd need to check; since both dims are 2880, transpose is safe
        # HF down_proj[e] used as: gated_output @ down_proj[e], gated_output=[T, intermediate]
        # → down_proj[e] must be [intermediate, hidden]
        # preprocess stored dequantized[e] directly → if blocks were [E, hidden, intermediate]
        # → down_proj.weight[e] is [hidden, intermediate], must transpose to [intermediate, hidden]
        down_stacked = torch.stack([d.t().contiguous() for d in down_list], dim=0)
        output_tensors[f"{prefix}.mlp.experts.down_proj"] = down_stacked

        # gate_up_proj_bias [E, 2*ffn] interleaved
        if gate_proj_b.get(prefix):
            gb = [gate_proj_b[prefix][e] for e in range(E)]
            ub = [up_proj_b[prefix][e]   for e in range(E)]
            bias_interleaved = []
            for e in range(E):
                b = torch.empty(2 * ffn, dtype=gb[e].dtype)
                b[0::2] = gb[e]
                b[1::2] = ub[e]
                bias_interleaved.append(b)
            output_tensors[f"{prefix}.mlp.experts.gate_up_proj_bias"] = torch.stack(bias_interleaved, dim=0)

        # down_proj_bias [E, hidden]
        if down_proj_b.get(prefix):
            db = [down_proj_b[prefix][e] for e in range(E)]
            output_tensors[f"{prefix}.mlp.experts.down_proj_bias"] = torch.stack(db, dim=0)

    output_tensors.update(other)

    # Save in shards of ~5GB
    SHARD_BYTES = 5 * 1024**3
    chunks, cur, cur_size = [], {}, 0
    for k, t in output_tensors.items():
        sz = t.numel() * t.element_size()
        if cur_size + sz > SHARD_BYTES and cur:
            chunks.append(cur); cur = {}; cur_size = 0
        cur[k] = t; cur_size += sz
    if cur:
        chunks.append(cur)

    total = len(chunks)
    new_weight_map = {}
    print(f"Writing {total} shard(s) with {len(output_tensors)} tensors...")
    for i, chunk in enumerate(chunks):
        fname = f"model-{i+1:05d}-of-{total:05d}.safetensors"
        save_file(chunk, os.path.join(output_dir, fname))
        for k in chunk:
            new_weight_map[k] = fname
        print(f"  wrote {fname} ({len(chunk)} tensors)")

    total_size = sum(t.numel() * t.element_size() for t in output_tensors.values())
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": new_weight_map}, f, indent=2)

    print(f"Done! {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    convert(args.input, args.output)
