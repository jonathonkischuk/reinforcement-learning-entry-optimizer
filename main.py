from train import train_all
from envs.trade_env import TradeEnv
from stable_baselines3 import PPO
from utils.indicators import compute_technical_indicators
from utils.config import stock_tickers, crypto_ids
from plot import plot_training_results, show_all_charts
import pandas as pd


def run_all():
    for ticker in stock_tickers + list(crypto_ids.keys()):
        path = f"data/stocks/{ticker}.csv" if ticker in stock_tickers else f"data/crypto/{ticker}.csv"

        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"[SKIP] Data not found for {ticker}")
            continue

        df = compute_technical_indicators(df)
        env = TradeEnv(df)

        try:
            model = PPO.load(f"models/ppo_{ticker}")
        except FileNotFoundError:
            print(f"[SKIP] Model not found for {ticker}")
            continue

        obs, _ = env.reset()
        done = False
        equity = []

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            equity.append(env.equity)

        eq_df = pd.DataFrame({'Close': equity})
        plot_training_results(eq_df, ticker)

    show_all_charts()


if __name__ == "__main__":
    train_all()
    run_all()
