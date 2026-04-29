"""CLI entry point for LIBERO → starVLA dataset preprocessing.

Supports incremental processing: run once per suite, all output goes
to the same --output_dir. New samples append to data.jsonl, statistics
are recomputed over all accumulated data.

Usage:
    # Process libero_spatial
    python runners/preprocess_libero.py \
        --input_dir datasets/libero2uam/raw/libero_spatial \
        --output_dir datasets/libero2uam \
        --suite libero_spatial

    # Then process libero_object into the same output dir
    python runners/preprocess_libero.py \
        --input_dir datasets/libero2uam/raw/libero_object \
        --output_dir datasets/libero2uam \
        --suite libero_object
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path so tools/starVLA packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.preprocess.libero_preprocessor import LiberoPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LIBERO HDF5 dataset to starVLA unified format."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing LIBERO .hdf5 files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Target directory for starVLA format output.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
        help="LIBERO suite name (used in sample IDs). Default: libero_spatial",
    )
    parser.add_argument(
        "--target_object_keyword",
        type=str,
        default=None,
        help="Fallback keyword to match target object body in MuJoCo when "
             "the LLM resolver is disabled or returns no match.",
    )
    parser.add_argument(
        "--use_llm_resolver",
        action="store_true",
        help="Enable LLM-based target object resolution (DeepSeek). "
             "Requires DEEPSEEK_API_KEY env var.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="deepseek-chat",
        help="LLM model name. Default: deepseek-chat",
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="https://api.deepseek.com",
        help="OpenAI-compatible API base URL. Default: https://api.deepseek.com "
             "(SiliconFlow: https://api.siliconflow.cn/v1)",
    )
    parser.add_argument(
        "--llm_cache_path",
        type=str,
        default="cache/llm_target_resolver.json",
        help="Path to the resolver cache JSON file (relative to project root).",
    )
    parser.add_argument(
        "--on_resolve_failure",
        choices=("skip", "abort"),
        default="skip",
        help="What to do when a task's target object cannot be resolved: "
             "'skip' the sample and continue, or 'abort' the run. "
             "Only meaningful with --use_llm_resolver. Default: skip.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    target_resolver = None
    skip_on_failure = False
    if args.use_llm_resolver:
        from tools.preprocess.target_object_resolver import (
            LLMTargetResolver,
        )
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit(
                "ERROR: --use_llm_resolver requires the DEEPSEEK_API_KEY "
                "environment variable to be set."
            )
        target_resolver = LLMTargetResolver(
            api_key=api_key,
            model=args.llm_model,
            base_url=args.llm_base_url,
            cache_path=args.llm_cache_path,
        )
        skip_on_failure = args.on_resolve_failure == "skip"

    preprocessor = LiberoPreprocessor(
        suite=args.suite,
        target_object_keyword=args.target_object_keyword,
        target_resolver=target_resolver,
        skip_on_resolve_failure=skip_on_failure,
    )

    preprocessor.process(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
