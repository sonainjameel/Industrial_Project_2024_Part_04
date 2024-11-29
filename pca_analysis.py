import argparse
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from envi2 import *
import cv2

def read_and_process_images(dir_path):
    def numeric_sort_key(filename):
        return -int(filename.split('_')[0])  # Negate for descending order

    reflectance_images = []
    for i in sorted(os.listdir(dir_path), key=numeric_sort_key):
        if '.hdr' in i:
            print(i)
            sample_cube, wavelengths, header = read_envi(header_file=Path(dir_path, i), normalize=False)
            mean = np.mean(sample_cube, axis=2, keepdims=True)
            std = np.std(sample_cube, axis=2, keepdims=True)
            standardized_cube = (sample_cube - mean) / std
            reflectance_images.append(standardized_cube)

    max_height = max(image.shape[0] for image in reflectance_images)
    max_width = max(image.shape[1] for image in reflectance_images)

    resized_images = [cv2.resize(img, (max_width, max_height), interpolation=cv2.INTER_LINEAR)
                      for img in reflectance_images]

    stitched_rows = []
    for start_idx in range(0, len(resized_images), 8):  # Adjust batch size as needed
        row_images = resized_images[start_idx:start_idx + 8][::-1]
        stitched_rows.append(np.concatenate(row_images, axis=1))

    stitched_image_grid = np.concatenate(stitched_rows, axis=0)
    return stitched_image_grid, wavelengths

def perform_pca(stitched_image_grid, components):
    height, width, num_wavelengths = stitched_image_grid.shape
    reshaped_grid = stitched_image_grid.reshape(-1, num_wavelengths)

    pca = PCA(n_components=components)
    principal_components = pca.fit_transform(reshaped_grid)
    pca_images = principal_components.reshape(height, width, components)

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, components + 1), pca.explained_variance_ratio_[:components] * 100 , color='b', alpha=0.7, label='Explained Variance')
    plt.xlabel('Principal Component', fontsize=24)
    plt.ylabel('Explained Variance (%)', fontsize=24)
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.show()

    return pca, pca_images

def plot_pca_results(pca, pca_images, wavelengths, labels):
    for i in range(min(3, pca_images.shape[2])):
        plt.figure(figsize=(10, 8))
        im=plt.imshow(pca_images[:, :, i], cmap='jet')
        plt.xlabel(labels[0], fontsize=24)
        plt.ylabel(labels[1], fontsize=24)
        cbar = plt.colorbar(im)  # Add color bar
        cbar.ax.tick_params(labelsize=24)  # Change font size of the color scale
        # Disable axis ticks (scale)
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, pca.components_[0, :], label="PC1")
    plt.xlabel('Wavelength (nm)', fontsize=24)
    plt.ylabel('Loading Value', fontsize=24)
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run specific PCA tasks based on config.yaml.")
    parser.add_argument("--task", type=str, help="Name of the task to run (optional).")
    args = parser.parse_args()

    with open("config2.yaml", "r") as file:
        config = yaml.safe_load(file)

    tasks = config['tasks']
    if args.task:
        tasks = [task for task in tasks if task['name'] == args.task]
        if not tasks:
            print(f"Task '{args.task}' not found in config2.yaml.")
            return

    for task in tasks:
        name = task['name']
        dir_path = task['dir_path']
        components = task['components']

        print(f"Processing task: {name}")
        stitched_image_grid, wavelengths = read_and_process_images(dir_path)
        pca, pca_images = perform_pca(stitched_image_grid, components)

        if name == "structure_oil":
            plot_pca_results(pca, pca_images, wavelengths, labels=["Structure", "Oil Level"])
        elif name == "structure_charring":
            plot_pca_results(pca, pca_images, wavelengths, labels=["Structure", "Charring Level"])
        elif name == "oil_charring":
            plot_pca_results(pca, pca_images, wavelengths, labels=["Oil Level", "Charring"])
        elif name == "cracking":
            plot_pca_results(pca, pca_images, wavelengths, labels=["Cracking", "Intensity"])

if __name__ == "__main__":
    main()
