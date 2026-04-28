import argparse
import os

import yaml

from apex_plus.cluster.cluster import Cluster
from apex_plus.models.registry import get_model_ir
from apex_plus.search.engine import SearchEngine
from apex_plus.simulator.comm_profile import EnergyConfig, InterconnectConfig, SpectraConfig
from apex_plus.simulator.trace import Trace
from apex_plus.utils.dtype import DTYPE, _DTYPE_REGISTRY

# NOTE: This is not a complete list of supported models.
SHORTCUT = {
    "bloom": "bigscience/bloom",
    "llama-7b": "huggyllama/llama-7b",
    "llama-13b": "huggyllama/llama-13b",
    "llama-30b": "huggyllama/llama-30b",
    "llama-65b": "huggyllama/llama-65b",
    "gpt-j": "EleutherAI/gpt-j-6b",
    "gpt-neox": "EleutherAI/gpt-neox-20b",
    "wizardcoder": "WizardLM/WizardCoder-15B-V1.0",
    "whisper": "openai/whisper-large-v3",
    "clip": "openai/clip-vit-large-patch14",
    # HF authentication token is needed if there is local config.json file
    # Configuration file can be taken from HF model repo
    "mistral-7b-local": "./apex_plus/models/mistral_config.json",
    "mistral-7b": "teknium/OpenHermes-2.5-Mistral-7B",
    "t5": "google/flan-t5-xxl",
    "llama3-70b": "./apex_plus/models/llama3_70b_config.json",
    "mixtral-8x7b-local": "./apex_plus/models/mixtral8x7b_config.json",
    "mixtral-8x7b": "cognitivecomputations/dolphin-2.5-mixtral-8x7b",
    "mixtral-8x22b-local": "./apex_plus/models/mixtral8x22b_config.json",
    "mixtral-8x22b": "mistral-community/Mixtral-8x22B-v0.1-AWQ",
    "llama3.1-70b": "./apex_plus/models/llama3.1_70b_config.json",
    "llama3.1-405b": "./apex_plus/models/llama3.1_405b_config.json",
    "moe-64x": "./apex_plus/models/moe_64x_config.json",
    "moe-64x-large": "./apex_plus/models/moe_64x_large_config.json",
    "deepseek-v2": "./apex_plus/models/deepseek_v2_config.json",
    "deepseek-v3": "./apex_plus/models/deepseek_v3_config.json",
    "kimi-k2": "./apex_plus/models/kimi_k2_config.json",
    "qwen3-235b": "./apex_plus/models/qwen3_235b_a22b_config.json",
}


def get_model_shortcuts():
    return SHORTCUT


def main(args: argparse.Namespace):
    if args.model in SHORTCUT:
        args.model = SHORTCUT[args.model]
    print(args)

    model, model_config = get_model_ir(
        args.model, args.num_experts, args.topk, args.capacity_factor
    )

    encoder_cluster = Cluster.from_gpu(args.gpu, args.num_nodes, 1)
    cluster = Cluster.from_gpu(args.gpu, args.num_nodes, args.num_gpus_per_node)

    if args.trace_file:
        trace = Trace.from_dynamic(args.trace_file)
    else:
        trace = Trace.from_static(args.num_requests, args.prompt_len, args.output_len)

    dtype = {
        "kv": _DTYPE_REGISTRY[args.kv_dtype],
        "w": _DTYPE_REGISTRY[args.weight_dtype],
        "act": _DTYPE_REGISTRY[args.activation_dtype],
    }

    # Build interconnect config
    ic_mode = args.interconnect
    if ic_mode is None:
        ic_mode = "ib" if args.num_nodes > 1 else "nvlink"
    interconnect = InterconnectConfig(
        mode=ic_mode,
        num_rails=args.ib_rails,
    )

    # Load simulator config (energy + spectra).
    energy_config = EnergyConfig()
    spectra_config = SpectraConfig()
    config_path = args.config
    if config_path is None and os.path.exists("config.yaml"):
        config_path = "config.yaml"
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        energy_cfg = cfg.get("energy", {})
        energy_config = EnergyConfig(**{
            k: v for k, v in energy_cfg.items() if k in EnergyConfig.__dataclass_fields__
        })
        spectra_cfg = cfg.get("spectra", {})
        spectra_config = SpectraConfig(**{
            k: v for k, v in spectra_cfg.items() if k in SpectraConfig.__dataclass_fields__
        })

    if model.num_encoder_blocks == 0 and model.num_decoder_blocks == 0:
        raise RuntimeError("Number of encoders and decoders cannot both be zero.")
    if model.num_encoder_blocks > 0:
        engine = SearchEngine(
            model, encoder_cluster, trace, "encoder", dtype, interconnect, args.moe_skew, energy_config
        )  # search for encoder
        _, trace = engine.search(
            args.all,
            args.frequency,
            args.request_percentiles,
            args.token_percentiles,
            model_config,
            args.ttft_slo,
            args.tpot_slo,
            args.max_batch_size,
            args.force_ep,
        )  # updated traces by adding encode time
        trace = Trace(trace)
    if model.num_decoder_blocks > 0:
        engine = SearchEngine(
            model, cluster, trace, "decoder", dtype, interconnect, args.moe_skew, energy_config
        )  # search for decoder
        best_plans, trace = engine.search(
            args.all,
            args.frequency,
            args.request_percentiles,
            args.token_percentiles,
            model_config,
            args.ttft_slo,
            args.tpot_slo,
            args.max_batch_size,
            args.force_ep,
        )

        if args.demand_matrix and best_plans:
            from apex_plus.search.engine import get_plan_tag
            from apex_plus.simulator.demand_matrix import (
                extract_demand_matrices,
                save_demand_matrices,
            )
            from apex_plus.simulator.expert_distribution import make_expert_dist

            dm_tokens = args.dm_num_tokens or args.prompt_len
            best_plan = best_plans[0]

            # Build expert distribution from CLI args.
            expert_dist = None
            if args.moe_dist is not None:
                # Explicit --moe-dist: use it with --moe-dist-param.
                dist_name = args.moe_dist
                if args.moe_dist_param is not None:
                    dist_param = args.moe_dist_param
                elif dist_name == "zipf":
                    dist_param = args.moe_skew if args.moe_skew > 0 else 1.0
                elif dist_name == "dirichlet":
                    dist_param = 1.0
                else:
                    dist_param = 0.0
                # Extract num_experts from MoE cells in the model.
                num_experts = None
                for block in (model.decoder_block, model.encoder_block):
                    if block is None:
                        continue
                    for cell in block.cells:
                        if hasattr(cell, "num_experts"):
                            num_experts = cell.num_experts
                            break
                    if num_experts is not None:
                        break
                expert_dist = make_expert_dist(
                    dist_name, dist_param, num_experts=num_experts
                )

            matrices = extract_demand_matrices(
                best_plan, cluster, model, dtype["act"], dm_tokens,
                args.moe_skew, expert_dist,
            )
            # Extract clean model name from path or HF name.
            raw = args.model
            if "/" in raw:
                raw = raw.split("/")[-1]
            if raw.endswith(".json"):
                raw = raw.rsplit("_config.json", 1)[0].rsplit(".json", 1)[0]
            model_name = raw
            prefix = f"{model_name}.{get_plan_tag(best_plan)}"
            print(f"\n* Demand matrices ({dm_tokens} tokens):")
            save_demand_matrices(matrices, args.demand_matrix, prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Model name in HuggingFace model Hub"
    )
    # MoE config
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Number of MLP experts of the model. Default is none for models not regarded as MOE model.",
    )
    parser.add_argument(
        "--topk", type=int, default=2, help="Topk hyperparameter for MOE models."
    )
    parser.add_argument(
        "--capacity-factor",
        type=float,
        default=1.0,
        help="Capacity factor for MoE models",
    )
    # Cluster config
    parser.add_argument(
        "--num-nodes", type=int, default=1, help="Number of nodes in the cluster."
    )
    parser.add_argument(
        "--num-gpus-per-node",
        type=int,
        required=True,
        help=" Number of GPUs per node in the cluster",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        choices=["V100-PCIE-16GB", "H100-SXM-80GB","H200-SXM-141GB",],
        default="H100-SXM-80GB",
    )
    parser.add_argument("--frequency", type=int, choices=[0, 810, 1980], default=0)
    parser.add_argument(
        "--interconnect",
        type=str,
        choices=["nvlink", "ib"],
        default=None,
        help="Inter-node interconnect type. Default: nvlink if single-node, ib if multi-node.",
    )
    parser.add_argument(
        "--ib-rails",
        type=int,
        default=8,
        help="Number of IB rails (NICs) per node. DGX H100 has 8. Default: 8.",
    )
    # Workload config
    parser.add_argument("--trace-file", type=str)
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1024,
        help="Number of requests to feed into the APEX. Large number of requests increase simulation accuracy but also increases latency of simulation execution.",
    )
    # Misc
    parser.add_argument(
        "--disable-ray",
        action="store_true",
        help="Disable Ray and serialize the execution of " "simulation.",
    )
    # Quantization
    parser.add_argument(
        "--kv-dtype", type=str, choices=["float", "half", "float8"], default="half"
    )
    parser.add_argument(
        "--weight-dtype", type=str, choices=["float", "half", "float8"], default="half"
    )
    parser.add_argument(
        "--activation-dtype",
        type=str,
        choices=["float", "half", "float8"],
        default="half",
    )

    # Output config
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Output all possible execution plans. Defaults to False",
    )
    # Log Additional Percentiles
    parser.add_argument(
        "--request-percentiles",
        type=int,
        default=[],
        nargs="+",
        help="Output specified percentiles in addition to P50 and P95 for request latencies",
    )
    parser.add_argument(
        "--token-percentiles",
        type=int,
        default=[],
        nargs="+",
        help="Output specified percentiles in addition to P50 and P95 for token generation latencies",
    )
    # Define SLO in ms
    parser.add_argument(
        "--ttft-slo", 
        type=int, default=10,
        help="Define SLO Latency for TTFT in ms. Default is 10 ms"
    )
    parser.add_argument(
        "--tpot-slo", 
        type=int, 
        default=10,
        help="Define SLO Latency for TPOT in ms. Default is 10 ms"
    )
    # Define max batch size
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=0,
        help="Define max batch size. This is also known as max number of sequences."
        )
    parser.add_argument(
        "--force-ep",
        type=int,
        default=0,
        help="Force minimum expert parallelism degree with training constraint (TP x EP = total). "
             "Filters plans where MoE EP < this value or attention cell-DP != EP.",
    )
    parser.add_argument(
        "--moe-skew",
        type=float,
        default=0.0,
        help="(Legacy) Zipf exponent for MoE expert popularity. "
             "Equivalent to --moe-dist zipf --moe-dist-param <value>. "
             "Ignored when --moe-dist is set.",
    )
    parser.add_argument(
        "--moe-dist",
        type=str,
        default=None,
        choices=["uniform", "zipf", "dirichlet"],
        help="Expert routing distribution for demand matrix AllToAll traffic. "
             "uniform = equal probability. "
             "zipf = Zipf-ranked expert popularity (param = exponent s). "
             "dirichlet = per-GPU Dirichlet-sampled routing (param = alpha). "
             "Default: zipf if moe_skew > 0, else uniform.",
    )
    parser.add_argument(
        "--moe-dist-param",
        type=float,
        default=None,
        help="Parameter for --moe-dist. "
             "zipf: exponent s (0.3-0.5 typical, 1.0 = classic Zipf). "
             "dirichlet: concentration alpha (<1 spiky, 1 uniform simplex, >10 flat). "
             "uniform: ignored. Default: moe_skew value for zipf, 1.0 for dirichlet.",
    )
    # Demand matrix extraction
    parser.add_argument(
        "--demand-matrix",
        type=str,
        default=None,
        help="Output directory for demand matrices. Extracts R×R byte-traffic matrices per collective.",
    )
    parser.add_argument(
        "--dm-num-tokens",
        type=int,
        default=None,
        help="Representative token count for demand matrix (default: prompt_len).",
    )
    # Simulator config (energy + spectra)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML simulator config file. Default: config.yaml if present.",
    )
    args = parser.parse_args()

    main(args)
