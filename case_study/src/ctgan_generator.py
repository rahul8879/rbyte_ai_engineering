
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def train_and_sample_ctgan(input_csv, output_csv, epochs=300, sample_n=100000, random_state=42):
    try:
        from ctgan import CTGANSynthesizer
    except Exception as e:
        raise ImportError("ctgan library not installed. Install with `pip install ctgan`") from e

    print(f"[INFO] Loading {input_csv} ...")
    df = pd.read_csv(input_csv, parse_dates=['timestamp'], infer_datetime_format=True)

    # select columns and convert datatypes for CTGAN
    # CTGAN wants discrete columns marked. We'll cast categorical-like columns to string.
    cat_cols = []
    for c in ['txn_type','channel','status','payer_upi','payee_upi','payer_device_id','payee_device_id','payer_city','payee_city','bank_code']:
        if c in df.columns:
            df[c] = df[c].astype(str)
            cat_cols.append(c)

    # timestamp is continuous numeric (we'll convert to epoch seconds)
    if 'timestamp' in df.columns:
        df['timestamp_epoch'] = df['timestamp'].astype('int64') // 10**9
    # drop original timestamp for modeling (keep epoch)
    if 'timestamp' in df.columns:
        df.drop(columns=['timestamp'], inplace=True)

    print(f"[INFO] Columns: {list(df.columns)}")
    print(f"[INFO] Categorical columns: {cat_cols}")

    # CTGAN requires no NaNs in categorical columns
    df.fillna({'amount':0}, inplace=True)
    df.fillna("NA", inplace=True)

    # initialize CTGAN
    ctgan = CTGANSynthesizer(epochs=epochs, verbose=True, batch_size=500, cuda=False)
    print("[INFO] Training CTGAN ... this may take a while depending on epochs & data size")
    ctgan.fit(df, discrete_columns=cat_cols)

    print(f"[INFO] Sampling {sample_n} rows ...")
    sampled = ctgan.sample(sample_n)

    # if we created timestamp_epoch, convert back to datetime
    if 'timestamp_epoch' in sampled.columns:
        sampled['timestamp'] = pd.to_datetime(sampled['timestamp_epoch'].astype(int), unit='s')
        sampled.drop(columns=['timestamp_epoch'], inplace=True)

    # optional: ensure numeric columns correct types
    if 'amount' in sampled.columns:
        sampled['amount'] = pd.to_numeric(sampled['amount'], errors='coerce').fillna(0).round(2)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    sampled.to_csv(output_csv, index=False)
    print(f"[OK] CTGAN synthetic CSV written to {output_csv} ({len(sampled)} rows)")
    return output_csv

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input seed CSV path")
    ap.add_argument("--output", required=True, help="Output CSV path for synthetic rows")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--sample", type=int, default=100000)
    args = ap.parse_args()
    train_and_sample_ctgan(args.input, args.output, epochs=args.epochs, sample_n=args.sample)
