from stable_baselines3 import PPO
from envs.trade_env import TradeEnv
from utils.fetch_data import ensure_stock_data, ensure_crypto_data
from utils.config import stock_tickers, crypto_ids
from utils.indicators import compute_technical_indicators
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import time
from plot import plot_training_results  # Visualization function


class TrainingProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        if self.n_calls % 2048 == 0:
            percent = 100 * self.num_timesteps / self.total_timesteps
            tqdm.write(f"[PROGRESS] {self.model_name}: {percent:.1f}% complete")
        return True

    def set_model_name(self, name):
        self.model_name = name


def train_all():
    all_tickers = stock_tickers + list(crypto_ids.keys())
    trained_tickers = []

    for ticker in tqdm(all_tickers, desc="Training Models", ncols=100):
        is_stock = ticker in stock_tickers
        data_path = f"data/stocks/{ticker}.csv" if is_stock else f"data/crypto/{ticker}.csv"
        model_path = Path(f"models/ppo_{ticker}.zip")

        # Prompt user if model already exists
        if model_path.exists():
            retrain = input(f"[EXISTING MODEL] {ticker} already trained. Retrain? (y/n): ").strip().lower()
            if retrain != "y":
                tqdm.write(f"[SKIPPED] Skipping training for {ticker}")
                continue

        # Ensure data is available
        if is_stock:
            ensure_stock_data([ticker])
        else:
            ensure_crypto_data({ticker: crypto_ids[ticker]})

        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            tqdm.write(f"[ERROR] File not found for {ticker}. Skipping.")
            continue

        df = compute_technical_indicators(df)
        env = TradeEnv(df)
        env = Monitor(env)  # Log environment statistics

        tqdm.write(f"\n[TRAINING STARTED] {ticker}")
        model = PPO("MlpPolicy", env, verbose=1)

        callback = TrainingProgressCallback(total_timesteps=50_000)
        callback.set_model_name(ticker)

        start_time = time.time()
        model.learn(total_timesteps=50_000, callback=callback)
        model.save(f"models/ppo_{ticker}")
        elapsed = time.time() - start_time

        tqdm.write(f"[TRAINING COMPLETE] {ticker} in {elapsed:.1f}s")
        tqdm.write("-" * 80)

        trained_tickers.append((ticker, df))

    tqdm.write("\n[VISUALIZATION] Generating charts for all trained tickers...\n")
    for ticker, df in trained_tickers:
        plot_training_results(df, ticker)