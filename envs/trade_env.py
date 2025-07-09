from gymnasium import Env, spaces
import numpy as np

class TradeEnv(Env):
    def __init__(self, df, window_size=10, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, 5), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.equity = self.initial_balance
        return self._get_obs(), {}

    def _get_obs(self):
        obs_cols = ['Close', 'Volume', 'rsi', 'macd', 'macd_signal']
        return self.df[obs_cols].iloc[self.current_step - self.window_size:self.current_step].values.astype(np.float32)

    def step(self, action):
        price = self.df.loc[self.current_step, "Close"]
        reward = 0
        done = False

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 1:
            reward = price - self.entry_price
            self.balance += reward
            self.position = 0

        self.equity = self.balance if self.position == 0 else self.balance + (price - self.entry_price)
        self.current_step += 1

        if self.current_step >= len(self.df):
            done = True

        return self._get_obs(), reward, done, False, {}
