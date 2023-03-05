# This is a Module for Color(RGB) Image Compression by SVD Decomposition
# Author: Liu LinFeng   MATrIc.No: G2201402L
# Email: lliu034@e.ntu.edu.sg

import numpy as np
import matplotlib.pyplot as plt


def color_img_compression_svd(original_img, rank):
    # Initialization of Image_Compressed
    img_compressed = np.zeros(original_img.shape)

    # Processing SVD Compression in 3 Channels(RGB)
    for channel in range(3):
        u, sigma, v = np.linalg.svd(img[:, :, channel])  # Doing SVD by numpy
        sigma = sorted(sigma, reverse=True)  # Sort the sigma from high value to low value

        # Truncate the original sigma to low rank
        low_rank_sigma_matrix = np.zeros(shape=(rank, rank), dtype=float)
        for rank_index in range(rank):
            low_rank_sigma_matrix[rank_index, rank_index] = sigma[rank_index]

        # Get the Compressed Image by Recomposition
        img_compressed[:, :, channel] = u[:, 0:rank].dot(low_rank_sigma_matrix).dot(v[0:rank, :])

    # Normalization of RGB Channel
    for channel in range(3):
        max_val = np.max(img_compressed[:, :, channel])
        min_val = np.min(img_compressed[:, :, channel])
        img_compressed[:, :, channel] = (img_compressed[:, :, channel] - min_val) / (max_val - min_val)
    img_compressed = np.round(img_compressed * 255).astype("uint8")

    return img_compressed


if __name__ == '__main__':
    plt.figure(figsize=(12, 12))
    img = plt.imread("original_image/Totoro.jpg")
    f_origin = plt.subplot(331)
    f_origin.imshow(img)
    f_origin.set_title("Origin")

    for i in range(8):
        compressed_rank = 100*(i+1)
        img_com = color_img_compression_svd(img, compressed_rank)
        f_compressed = plt.subplot(3, 3, i+2)
        f_compressed.set_title("p=" + str(compressed_rank))
        f_compressed.imshow(img_com)

    plt.show()



