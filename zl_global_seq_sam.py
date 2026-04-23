import argparse
import os

import pandas as pd


DEFAULT_BASE_DIR = r"E:\code\lwCode\ZL_DCHL\logs_ex3\seed2026_20260422_211551"
DEFAULT_INPUT = os.path.join(DEFAULT_BASE_DIR, "patent_table_final.csv")
DEFAULT_OUTPUT = os.path.join(DEFAULT_BASE_DIR, "patent_table_selected_examples_global_s42.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="从总表中随机抽取专利展示样本")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="总表路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出路径")
    parser.add_argument("--sample-count", type=int, default=6, help="随机抽取样本数")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    df["top1_score"] = pd.to_numeric(df["top1_score"], errors="coerce")
    df["normalized_top1_score"] = pd.to_numeric(df["normalized_top1_score"], errors="coerce")

    selected_df = df.sample(n=min(args.sample_count, len(df)), random_state=args.seed).copy()
    selected_df = selected_df[
        [
            "sample_idx",
            "user_id",
            "prefix_poi_seq",
            "recommended_poi",
            "top1_score",
            "normalized_top1_score",
        ]
    ].copy()
    selected_df = selected_df.sort_values(by=["sample_idx"])
    selected_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"total rows: {len(df)}")
    print(f"selected rows: {len(selected_df)} -> {args.output}")


if __name__ == "__main__":
    main()
