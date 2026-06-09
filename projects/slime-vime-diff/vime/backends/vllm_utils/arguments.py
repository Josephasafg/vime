import argparse

from vllm.srt.server_args import ServerArgs
from vime.utils.http_utils import _wrap_ipv6


# TODO: use all vllm router arguments with `--vllm-router` prefix
def add_vllm_router_arguments(parser):
    """
    Add arguments to the parser for the vLLM router.
    """
    parser.add_argument(
        "--vllm-router-ip",
        type=str,
        default=None,
        help="IP address of the vLLM router",
    )
    parser.add_argument(
        "--vllm-router-port",
        type=int,
        default=None,
        help="Port of the vLLM router",
    )
    parser.add_argument(
        "--vllm-router-request-timeout-secs",
        type=int,
        default=14400,
        help="Timeout for requests to the vLLM router in seconds",
    )
    return parser


def add_vllm_arguments(parser):
    """
    Add arguments to the parser for the vLLM server.
    """
    parser = add_vllm_router_arguments(parser)
    parser.set_defaults(router_balance_abs_threshold=10, router_balance_rel_threshold=1.2)
    parser.add_argument("--vllm-server-concurrency", type=int, default=512)

    old_add_argument = parser.add_argument

    skipped_args = [
        "model_path",
        "config",
        "trust_remote_code",
        "random_seed",
        # memory
        "enable_memory_saver",
        # distributed
        "tp_size",
        "port",
        "nnodes",
        "node_rank",
        "dist_init_addr",
        "gpu_id_step",
        "base_gpu_id",
        "nccl_port",
        "skip_server_warmup",
        "enable_return_routed_experts",
    ]

    def new_add_argument_wrapper(*name_or_flags, **kwargs):
        """
        Add arguments to the parser, ensuring that the server arguments are prefixed and skippable.
        """
        # Determine the canonical name for skip check (e.g., "model_path")
        canonical_name_for_skip_check = None
        if "dest" in kwargs:
            canonical_name_for_skip_check = kwargs["dest"]
        else:
            for flag_name_candidate in name_or_flags:
                if isinstance(flag_name_candidate, str) and flag_name_candidate.startswith("--"):
                    # Derive from first long flag: --foo-bar -> foo_bar
                    stem = flag_name_candidate[2:]
                    canonical_name_for_skip_check = stem.replace("-", "_")
                    break
            # If no long flag and no dest, skip logic might not catch it unless short flags imply a dest.

        if canonical_name_for_skip_check and canonical_name_for_skip_check in skipped_args:
            return  # Skip this entire argument definition

        # If not skipped, proceed to prefix flags and dest
        new_name_or_flags_list = []
        for item_flag in name_or_flags:
            if isinstance(item_flag, str) and item_flag.startswith("-"):
                original_flag_stem = item_flag.lstrip("-")  # "foo-bar" from "--foo-bar", or "f" from "-f"
                prefixed_item = f"--vllm-{original_flag_stem}"
                new_name_or_flags_list.append(prefixed_item)
            else:
                # Positional arguments or non-string items
                new_name_or_flags_list.append(item_flag)

        # Prepare kwargs for the actual add_argument call.
        # Make a copy to avoid modifying the original kwargs dict.
        final_kwargs = kwargs.copy()

        # If 'dest' is explicitly provided and is a string, prefix it.
        # This ensures the attribute on the args namespace becomes, e.g., args.vllm_dest_name.
        if "dest" in final_kwargs and isinstance(final_kwargs["dest"], str):
            original_dest = final_kwargs["dest"]
            # Avoid double prefixing if dest somehow already starts with vllm_
            if not original_dest.startswith("vllm_"):
                final_kwargs["dest"] = f"vllm_{original_dest}"
        # If 'dest' is not explicitly provided (or is None/not a string),
        # argparse will derive 'dest' from the (now prefixed) flag names.
        # E.g., if the first flag is "--vllm-foo-bar", argparse sets dest to "vllm_foo_bar".

        old_add_argument(*new_name_or_flags_list, **final_kwargs)

    parser.add_argument = new_add_argument_wrapper
    ServerArgs.add_cli_args(parser)
    parser.add_argument = old_add_argument

    # PD disaggregation / multi-group config
    parser.add_argument(
        "--prefill-num-servers",
        type=int,
        default=None,
        help="Number of prefill servers for disaggregation.",
    )
    parser.add_argument(
        "--vllm-config",
        type=str,
        default=None,
        help=(
            "Path to a YAML config for vLLM engine deployment. "
            "Defines server_groups with worker_type (regular/prefill/decode/placeholder), "
            "num_gpus per group, and optional per-group 'overrides' dict of "
            "ServerArgs field names that override the base --vllm-* CLI args. "
            "Placeholder groups reserve GPU slots without creating engines. "
            "Mutually exclusive with --prefill-num-servers."
        ),
    )

    return parser


def validate_args(args):
    args.vllm_dp_size = args.vllm_data_parallel_size
    args.vllm_pp_size = args.vllm_pipeline_parallel_size
    args.vllm_ep_size = args.vllm_expert_parallel_size

    # Compute effective TP size considering PP size
    if args.vllm_pp_size > 1:
        assert args.rollout_num_gpus_per_engine % args.vllm_pp_size == 0, (
            f"rollout_num_gpus_per_engine ({args.rollout_num_gpus_per_engine}) must be divisible by "
            f"vllm_pipeline_parallel_size ({args.vllm_pp_size})"
        )
        args.vllm_tp_size = args.rollout_num_gpus_per_engine // args.vllm_pp_size
    else:
        args.vllm_tp_size = args.rollout_num_gpus_per_engine

    if args.vllm_dp_size > 1:
        assert args.vllm_enable_dp_attention

    if getattr(args, "vllm_router_ip", None):
        args.vllm_router_ip = _wrap_ipv6(args.vllm_router_ip)

    # Mutual-exclusion checks for PD disaggregation / vllm-config.
    assert not (
        getattr(args, "prefill_num_servers", None) is not None and args.rollout_external
    ), "prefill_num_servers cannot be set when rollout_external is set."

    assert not (
        getattr(args, "vllm_config", None) is not None and args.rollout_external
    ), "vllm_config cannot be set when rollout_external is set."

    assert not (
        getattr(args, "vllm_config", None) is not None and getattr(args, "prefill_num_servers", None) is not None
    ), "vllm_config and prefill_num_servers are mutually exclusive. Use server_groups in the YAML config instead."


def vllm_parse_args():
    """
    Parse vllm server arguments independently using a separate ArgumentParser.
    Uses parse_known_args() to only consume vllm-related arguments from sys.argv,
    allowing the remaining arguments to be parsed by megatron separately.

    Returns:
        argparse.Namespace: Parsed vllm arguments (all attributes prefixed with vllm_).
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_vllm_arguments(parser)

    # Compute default vllm_tensor_parallel_size from CLI args
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    temp_parser.add_argument("--vllm-pp-size", type=int, default=1)
    temp_parser.add_argument("--vllm-pipeline-parallel-size", type=int, default=1)
    temp_args, _ = temp_parser.parse_known_args()
    pp_size = temp_args.vllm_pp_size if temp_args.vllm_pp_size != 1 else temp_args.vllm_pipeline_parallel_size
    vllm_tp_size = temp_args.rollout_num_gpus_per_engine // pp_size
    parser.set_defaults(vllm_tensor_parallel_size=vllm_tp_size)

    args, _ = parser.parse_known_args()
    return args
