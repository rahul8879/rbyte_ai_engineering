
import argparse
import os
from faker import Faker
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import random

fake = Faker('en_IN')
np.random.seed(42)
random.seed(42)

TXN_TYPES = ['P2P', 'P2M', 'AutoPay']
CHANNELS = ['QR', 'Intent', 'DeepLink']
STATUSES = ['SUCCESS', 'FAILED', 'PENDING']

CITIES = {
    "Mumbai": (19.0, 19.3, 72.7, 72.95),
    "Delhi": (28.4, 28.8, 76.9, 77.4),
    "Bengaluru": (12.8, 13.2, 77.4, 77.8),
    "Hyderabad": (17.2, 17.6, 78.3, 78.6),
    "Pune": (18.4, 18.7, 73.7, 73.95),
    "Chennai": (12.9, 13.2, 80.1, 80.35),
    "Kolkata": (22.4, 22.7, 88.2, 88.5),
    "Jaipur": (26.7, 27.0, 75.6, 75.9),
}

def _random_geo(city=None):
    if city is None:
        city = np.random.choice(list(CITIES.keys()))
    lat_min, lat_max, lon_min, lon_max = CITIES[city]
    lat = np.random.uniform(lat_min, lat_max)
    lon = np.random.uniform(lon_min, lon_max)
    return city, round(lat, 6), round(lon, 6)

def _device_id():
    return f"dev_{np.random.randint(10**6, 10**8)}"

def _upi_handle(name):
    providers = ['oksbi','okhdfcbank','okaxis','okicici','okyesbank']
    uname = ''.join(e for e in name.lower() if e.isalnum())[:10]
    return f"{uname}@{np.random.choice(providers)}"

def generate_users(n_users=20000):
    users = []
    for _ in range(n_users):
        name = fake.name()
        city, lat, lon = _random_geo()
        users.append({
            "user_id": f"U{np.random.randint(10**7, 10**9)}",
            "name": name,
            "home_city": city,
            "home_lat": lat,
            "home_lon": lon,
            "device_id": _device_id(),
            "upi": _upi_handle(name),
            "bank_code": np.random.choice(["SBI","HDFC","ICICI","AXIS","YES"]),
            "typical_spend": np.random.choice([100, 300, 700, 1500, 4000], p=[0.3,0.3,0.2,0.15,0.05]),
        })
    return pd.DataFrame(users)

def generate_transactions(users_df, n_txn=50000, start_dt="2025-01-01", freq_in_seconds=60, fraud_rate=0.003):
    start = pd.to_datetime(start_dt)
    # create pools
    upis = users_df['upi'].tolist()
    devices = users_df['device_id'].tolist()
    cities = users_df['home_city'].tolist()

    timestamps = [start + timedelta(seconds=i*freq_in_seconds) for i in range(n_txn)]

    # amount: mixture to create long tail
    base = np.random.exponential(scale=800, size=n_txn)
    big_mask = np.random.rand(n_txn) < 0.05
    base[big_mask] *= np.random.uniform(4, 12, size=big_mask.sum())

    df = pd.DataFrame({
        "txn_id": [f"UPI{1000000+i}" for i in range(n_txn)],
        "timestamp": timestamps,
        "txn_type": np.random.choice(TXN_TYPES, size=n_txn, p=[0.45, 0.45, 0.10]),
        "channel": np.random.choice(CHANNELS, size=n_txn, p=[0.6, 0.3, 0.1]),
        "amount": np.round(base, 2),
        "status": np.random.choice(STATUSES, size=n_txn, p=[0.93, 0.05, 0.02]),
        "payer_upi": np.random.choice(upis, size=n_txn),
        "payee_upi": np.random.choice(upis, size=n_txn),
        "payer_device_id": np.random.choice(devices, size=n_txn),
        "payee_device_id": np.random.choice(devices, size=n_txn),
        "payer_city": np.random.choice(cities, size=n_txn),
        "payee_city": np.random.choice(cities, size=n_txn),
        "is_fraud": 0,
    })

    # ensure payer != payee
    same_mask = df['payer_upi'] == df['payee_upi']
    if same_mask.any():
        df.loc[same_mask, 'payee_upi'] = np.random.choice(upis, size=same_mask.sum())

    # lat/lon
    payer_geo = [ _random_geo(c) for c in df['payer_city'] ]
    payee_geo = [ _random_geo(c) for c in df['payee_city'] ]
    df['payer_lat'] = [x[1] for x in payer_geo]
    df['payer_lon'] = [x[2] for x in payer_geo]
    df['payee_lat'] = [x[1] for x in payee_geo]
    df['payee_lon'] = [x[2] for x in payee_geo]

    # inject initial random frauds
    n_fraud = max(1, int(fraud_rate * n_txn))
    fraud_idx = np.random.choice(df.index, size=n_fraud, replace=False)
    df.loc[fraud_idx, 'is_fraud'] = 1

    # Pattern A: high value at odd hours (2-4 AM)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    odd_mask = df['hour'].isin([2,3,4]) & (df['amount'] > 50000)
    if odd_mask.any():
        mark = df[odd_mask].sample(frac=0.4, random_state=42).index
        df.loc[mark, 'is_fraud'] = 1

    # Pattern B: burst transfers for some payers
    payer_counts = df['payer_upi'].value_counts()
    high_activity = payer_counts[payer_counts > 10].index.tolist()
    if high_activity:
        sample_payers = np.random.choice(high_activity, size=min(50, len(high_activity)), replace=False)
        for payer in sample_payers:
            idxs = df[df['payer_upi'] == payer].sample(n=min(30, len(df[df['payer_upi']==payer])), random_state=42).index
            df.loc[idxs, 'is_fraud'] = 1

    # Pattern C: new device + new city mismatches
    new_dev_mask = np.random.rand(len(df)) < 0.01
    new_city_mask = np.random.rand(len(df)) < 0.01
    idxC = df[new_dev_mask & new_city_mask].index
    df.loc[idxC, 'is_fraud'] = 1

    # Pattern D: refund loops (P2M heavy back-and-forth)
    p2m_idxs = df[df['txn_type']=='P2M'].groupby('payee_upi').filter(lambda g: len(g) > 20).index
    if len(p2m_idxs) > 0:
        df.loc[p2m_idxs.sample(frac=0.05, random_state=42).index, 'is_fraud'] = 1

    # final cleanup
    df.drop(columns=['hour'], inplace=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50000)
    ap.add_argument("--users", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--freq", type=int, default=60, help="seconds between transactions (approx)")
    ap.add_argument("--fraud_rate", type=float, default=0.003)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"[INFO] Generating {args.users} users...")
    users = generate_users(args.users)
    print(f"[INFO] Generating {args.rows} transactions (seed) ...")
    df = generate_transactions(users, n_txn=args.rows, start_dt=args.start, freq_in_seconds=args.freq, fraud_rate=args.fraud_rate)
    df.to_csv(args.out, index=False)
    print(f"[OK] Seed CSV written to {args.out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
