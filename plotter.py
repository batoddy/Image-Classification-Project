import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def plot_epoch_loss(file_path):
    try:
        data = pd.read_csv(file_path)
        epoch = data["epoch"]
        gd_loss = data["gd_loss"]
        sgd_loss = data["sgd_loss"]
        adam_loss = data["adam_loss"]
        gd_time = data["gd_time"]
        sgd_time = data["sgd_time"]
        adam_time = data["adam_time"]

        # Loss Plot
        plt.figure(figsize=(10, 6))
        plt.plot(epoch, gd_loss, color="red", label="GD Loss")
        plt.plot(epoch, sgd_loss, color="blue", label="SGD Loss")
        plt.plot(epoch, adam_loss, color="green", label="Adam Loss")
        plt.xlim(0, 20)
        plt.title("Epoch vs Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Time Plot
        plt.figure(figsize=(10, 6))
        plt.plot(epoch, gd_time, color="red", label="GD Time")
        plt.plot(epoch, sgd_time, color="blue", label="SGD Time")
        plt.plot(epoch, adam_time, color="green", label="Adam Time")
        plt.title("Epoch vs Time")
        plt.xlabel("Epoch")
        plt.ylabel("Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


# # Example usage
# loss_filepath = r"C:/Users/Ahmet/Desktop/train_data.csv"
# plot_epoch_loss(loss_filepath)

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def w_visual_single(file_path):
    """
    Visualizes a single file using t-SNE, with normalization, noise addition, and perplexity adjustment.
    """
    try:
        # Veriyi oku
        data = pd.read_csv(file_path, header=None)

        print(data.shape)
        # Eksik değerleri doldur
        data.fillna(data.mean(), inplace=True)

        # Veriyi normalize et
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)

        # t-SNE Uygula
        tsne = TSNE(n_components=2, perplexity=3, random_state=42)
        data_2d = tsne.fit_transform(data_normalized)

        # Görselleştirme
        plt.figure(figsize=(8, 6))
        plt.scatter(data_2d[:, 0], data_2d[:, 1], color="blue", alpha=0.7)
        plt.title("t-SNE Visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


base_w_file_path = (
    "C:/Users/batuh/OneDrive/Masaüstü/Software/Image Classification Project/data"
)
w_visual_single(os.path.join(base_w_file_path, "train_data.csv"))
