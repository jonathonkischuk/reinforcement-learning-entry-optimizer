import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib
import numpy as np


matplotlib.use('TkAgg')

plot_data = []
current_index = 0
fig, ax = None, None

def plot_training_results(df, ticker):
    global plot_data
    equity_curve = df['Close'].cumsum()
    plot_data.append((ticker, equity_curve))


def show_all_charts():
    if not plot_data:
        print("[PLOT] No charts to display.")
        return

    _show_chart(0)


def _show_chart(index):
    global fig, ax, current_index
    current_index = index % len(plot_data)
    ticker, data = plot_data[current_index]

    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.canvas.manager.set_window_title(f"Equity Curve: {ticker}")

    ax.plot(data)
    ax.set_title(f"{ticker} Equity Curve")
    ax.set_ylabel("Equity (approx)")
    ax.set_xlabel("Time Steps")
    ax.grid(True)

    axprev = plt.axes([0.7, 0.01, 0.1, 0.05])
    axnext = plt.axes([0.81, 0.01, 0.1, 0.05])
    bnext = Button(axnext, 'Next →')
    bprev = Button(axprev, '← Prev')
    bnext.on_clicked(_next_chart)
    bprev.on_clicked(_prev_chart)

    fig.canvas.mpl_connect('key_press_event', _on_key)
    plt.show()


def _next_chart(event=None):
    _show_chart(current_index + 1)


def _prev_chart(event=None):
    _show_chart(current_index - 1)


def _on_key(event):
    if event.key == 'right':
        _next_chart()
    elif event.key == 'left':
        _prev_chart()
