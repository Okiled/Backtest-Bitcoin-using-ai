"""CLI entrypoint for the Bitcoin backtesting pipeline."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from core.optimizer import PipelineConfig, BitcoinPortfolioOptimizer
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bitcoin backtest pipeline")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Directory to store results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    config = PipelineConfig.from_files(Path(__file__).parent)
    optimizer = BitcoinPortfolioOptimizer(config=config, output_dir=args.output, device=device)
    optimizer.run()


if __name__ == "__main__":
    main()
