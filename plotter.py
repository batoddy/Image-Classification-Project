import pandas as pd

import matplotlib.pyplot as plt


file_path = "/data/train_data.csv"
df = pd.read_csv(file_path)


# Plotting the loss for each weight (w0, w1, ..., w4)

for i in range(5):

    plt.figure()

    plt.plot(df["epoch"], df[f"gd_loss_{i}"], label=f"GD Loss (w{i})")

    plt.plot(df["epoch"], df[f"sgd_loss_{i}"], label=f"SGD Loss (w{i})")

    plt.plot(df["epoch"], df[f"adam_loss_{i}"], label=f"Adam Loss (w{i})")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(f"Loss Progression for w{i}")

    plt.legend()

    plt.grid(True)

    plt.show()


# Plotting the time for each weight (w0, w1, ..., w4)

for i in range(5):

    plt.figure()

    plt.plot(df["epoch"], df[f"gd_time_{i}"], label=f"GD Time (w{i})")

    plt.plot(df["epoch"], df[f"sgd_time_{i}"], label=f"SGD Time (w{i})")

    plt.plot(df["epoch"], df[f"adam_time_{i}"], label=f"Adam Time (w{i})")

    plt.xlabel("Epoch")

    plt.ylabel("Time (Ticks)")

    plt.title(f"Computation Time Progression for w{i}")

    plt.legend()

    plt.grid(True)

    plt.show()
