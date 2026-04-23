import argparse
import os
import re

import pandas as pd


DEFAULT_BASE_DIR = r"E:\code\lwCode\ZL_DCHL\logs_ex3\seed2026_20260422_211551"
DEFAULT_INPUT = os.path.join(DEFAULT_BASE_DIR, "patent_table_final.csv")
def parse_args():
    parser = argparse.ArgumentParser(description="按序列长度分层并随机抽取专利展示样本")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="总表路径")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="输出目录")
    parser.add_argument("--sample-per-group", type=int, default=2, help="每个分层随机抽取样本数")
    parser.add_argument("--seed", type=int, default=2026, help="随机抽样种子")
    return parser.parse_args()


def prefix_len(prefix_text):
    if pd.isna(prefix_text):
        return 0
    return len(re.findall(r"P\d+", str(prefix_text)))


def sample_group(df, sample_count, seed):
    if df.empty:
        return df.copy()
    return df.sample(n=min(sample_count, len(df)), random_state=seed)


def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    df["top1_score"] = pd.to_numeric(df["top1_score"], errors="coerce")
    df["normalized_top1_score"] = pd.to_numeric(df["normalized_top1_score"], errors="coerce")
    df["prefix_len"] = df["prefix_poi_seq"].apply(prefix_len)

    short_df = df[(df["prefix_len"] >= 2) & (df["prefix_len"] <= 3)].copy()
    mid_df = df[(df["prefix_len"] >= 4) & (df["prefix_len"] <= 6)].copy()
    long_df = df[df["prefix_len"] >= 7].copy()

    short_df = short_df.sort_values(by=["sample_idx"])
    mid_df = mid_df.sort_values(by=["sample_idx"])
    long_df = long_df.sort_values(by=["sample_idx"])

    short_path = os.path.join(args.base_dir, "patent_table_short_seq.csv")
    mid_path = os.path.join(args.base_dir, "patent_table_mid_seq.csv")
    long_path = os.path.join(args.base_dir, "patent_table_long_seq.csv")
    selected_path = os.path.join(args.base_dir, "patent_table_selected_examples_s175.csv")

    short_df.to_csv(short_path, index=False, encoding="utf-8-sig")
    mid_df.to_csv(mid_path, index=False, encoding="utf-8-sig")
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    short_selected = sample_group(short_df, args.sample_per_group, args.seed)
    mid_selected = sample_group(mid_df, args.sample_per_group, args.seed + 1)
    long_selected = sample_group(long_df, args.sample_per_group, args.seed + 2)

    selected_df = pd.concat([short_selected, mid_selected, long_selected], ignore_index=True)
    selected_df = selected_df[
        [
            "sample_idx",
            "user_id",
            "prefix_poi_seq",
            "recommended_poi",
            "top1_score",
            "normalized_top1_score",
            "prefix_len",
        ]
    ].copy()
    selected_df = selected_df.sort_values(by=["prefix_len", "sample_idx"])
    selected_df.to_csv(selected_path, index=False, encoding="utf-8-sig")

    print(f"total rows: {len(df)}")
    print(f"short rows: {len(short_df)} -> {short_path}")
    print(f"mid rows: {len(mid_df)} -> {mid_path}")
    print(f"long rows: {len(long_df)} -> {long_path}")
    print(f"selected rows: {len(selected_df)} -> {selected_path}")


if __name__ == "__main__":
    main()
