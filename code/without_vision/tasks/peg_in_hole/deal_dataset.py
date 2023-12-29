import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import os
import argparse

def extract_training_dataset(path, dataset_path, seed=0, store=True, draw_statistic=False, store_dissimilar=False):

        pre_data = pd.read_excel(path, engine='openpyxl')
        data_with_name = np.array(pre_data)
        data = data_with_name[:, 1:15]
        names = data_with_name[:, 0]
        print(f"pose_data_shape: {data.shape} and {names.shape} images")

        if seed == 0:
                data_mean = np.mean(data, axis=0)
        else:
                data_mean = data[seed, :]
        
        distance_matx = data - data_mean
        distance_matx_abs = abs(data - data_mean)
        distance = np.sum(distance_matx_abs, axis=1)
        distance_abs = np.abs(distance)
        distance_mean = np.mean(distance_abs)
        distance_abs = distance_abs.reshape(-1)
        min_threshold = np.sort(distance_abs)[1]
        max_threshold = np.sort(distance_abs)[-300]

        # print(names[24600])
        # print(distance_abs[24600])
        
        # print(names[6406])
        # print(distance_abs[6406])

        print("min_threshold: ", min_threshold)
        print("max_threshold: ", max_threshold)
        similar_index = np.where(distance_abs < min_threshold * 1.3)
        dissimilar_index = np.where(distance_abs > max_threshold * 0.6)

        print("number of similar images: ", len(similar_index[0]))
        print("number of dissimilar images: ", len(dissimilar_index[0]))

        similar_name = names[similar_index]
        dissimilar_name = names[dissimilar_index]

        dataset_path = "dataset"
        sub_similar_data = f"{seed}_similar"
        sub_dissimilar_data = f"{seed}_dissimilar"

        if store == True:
                folder_path = dataset_path
                similar_folder = f'{seed}_similar'
                dissimilar_folder = f'{seed}_dissimilar'
                similar_path = os.path.join(folder_path, similar_folder)
                dissimilar_path = os.path.join(folder_path, dissimilar_folder)

                # Check if the main folder exists
                if os.path.exists(folder_path):
                        print(f"The folder '{folder_path}' exists.")

                        if os.path.exists(similar_path):
                                print(f"The subfolder '{similar_path}' exists within the main folder.")
                        elif len(similar_name) > 0:
                                os.mkdir(similar_path)
                                print(f"The subfolder '{similar_path}' has created.")

                        if store_dissimilar:
                                if os.path.exists(dissimilar_path):
                                        print(f"The subfolder '{dissimilar_path}' exists within the main folder.")
                                else:
                                        os.mkdir(dissimilar_path)
                                        print(f"The subfolder '{dissimilar_path}' has created.")

                else:
                        print(f"The folder '{folder_path}' does not exists.")
                        os.makedirs(similar_path)
                        print(f"The folder '{similar_path}' created.")
                        os.makedirs(dissimilar_path)
                        print(f"The folder '{dissimilar_path}' created.")
                
                # Check if a subfolder within the main folder exists
                if len(similar_name) > 0:
                        for i in range(len(similar_name)):
                                source_file = os.path.join("graphics_images", str(similar_name[i]))
                                destination_folder = os.path.join(dataset_path, sub_similar_data)
                                shutil.copy(source_file, destination_folder)
                                print(f"Copying {str(similar_name[i])} to {destination_folder}")

                        print(f"Finished {i+1} images to {destination_folder}")

                elif len(similar_name) <= 0:
                        print(f"Do not exist similar images")

                if store_dissimilar:
                        for i in range(len(dissimilar_name)):
                                source_file = os.path.join("graphics_images", str(dissimilar_name[i]))
                                destination_folder = os.path.join(dataset_path, sub_dissimilar_data)
                                shutil.copy(source_file, destination_folder)
                                print(f"Copying {str(dissimilar_name[i])} to {destination_folder}")

                        print(f"Finished {i+1} images to {destination_folder}")
                
                print(f"                                             ")
                print(f"{len(dissimilar_name)} dissimilar images has been stored and {len(similar_name)} similar images has been stored")

        if draw_statistic == True:
                plt.plot(distance_abs, marker='o', linestyle='-')
                plt.axhline(distance_mean, color='red', linestyle='--', label='mean_value')
                plt.axhline(min_value, color='yellow', linestyle='--', label='min_threshold')
                plt.axhline(max_value, color='black', linestyle='--', label='max_threshold')

                # Save the plot as an image (e.g., PNG format)
                fig_name = "statistic.png"
                fig_path = os.path.join(dataset_path, fig_name)

                plt.savefig(fig_path)
                # Show the plot (optional)
                plt.show()


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Your script's description")

        # Define hyperparameters as command-line arguments
        parser.add_argument('--path', type=str, default='dataset/data.xlsx', help='path of excel file')
        parser.add_argument('--dataset_path', type=str, default='dataset', help='path of dataset')
        parser.add_argument('--seed', type=int, default=-1, help='baseline of position and orientation')
        parser.add_argument('--store', type=bool, default=True, help='store the dataset')
        parser.add_argument('--store_dissimilar', type=bool, default=False, help='store the dissimilar dataset')
        parser.add_argument('--draw_statistic', type=bool, default=False, help='draw statistic of the data')
        parser.add_argument('--end', type=int, default=20000, help='range of seeds')
        parser.add_argument('--start', type=int, default=20000, help='start of seeds range')
        parser.add_argument('--stride', type=int, default=10, help='stride of seeds')

        args = parser.parse_args()

        # Access hyperparameters in your script
        path = args.path
        dataset_path = args.dataset_path
        seed = args.seed
        store = args.store
        draw_statistic = args.draw_statistic
        store_dissimilar = args.store_dissimilar
        end = args.end
        start = args.start
        stride = args.stride

        # Use hyperparameters in your code
        print(seed)
        if seed >= 0:
                extract_training_dataset(path=path, dataset_path=dataset_path, seed=seed, store=store, draw_statistic=draw_statistic, store_dissimilar=store_dissimilar) #                

        if seed < 0:
                for i in range(start, end, stride):
                        print(i)
                        extract_training_dataset(path=path, 
                                                 dataset_path=dataset_path, 
                                                 seed=i, 
                                                 store=store, 
                                                 draw_statistic=draw_statistic, 
                                                 store_dissimilar=store_dissimilar) #
        


# pre_data = pd.read_excel(path, engine='openpyxl')

# print(pre_data.shape)
# # names = np.array(pre_data[:, 0])
# # print(names.shape)
# # print(names[0])
# data_with_name = np.array(pre_data)
# print(data_with_name.shape)
# data = data_with_name[:, 1:15]
# print(data.shape)
# names = data_with_name[:, 0]
# # data = data[:512, :]

# data_mean = np.mean(data, axis=0)
# print("data_mean.shape", data_mean.shape)

# distance_matx = data - data_mean
# distance = np.sum(distance_matx, axis=1)
# distance_abs = np.abs(distance)
# distance_mean = np.mean(distance_abs)
# distance_abs = distance_abs.reshape(-1)
# min_value = np.sort(distance_abs)[0]
# max_value = np.sort(distance_abs)[-300]
# # mean = np.mean(distance_abs)
# # distance_abs_nonmean = distance_abs - mean
# # mean_mean = np.mean(distance_abs_nonmean)
# # print("shape: ", distance_abs.shape)
# # print("mean_mean: ", mean_mean)
# print("min_distance: ", min_value)
# print("second_max_distance: ", max_value)
# similar_index = np.where(distance_abs < min_value * 1.01)
# unsimilar_index = np.where(distance_abs > max_value * 0.6)
# print(len(similar_index[0]))
# print(len(unsimilar_index[0]))

# print(similar_index)


# print(similar_name)


# # similar_index = self.select_neighbour(data)

# # print("similar_index.shape", similar_index.shape)   

# plt.plot(distance_abs, marker='o', linestyle='-')
# plt.axhline(distance_mean, color='red', linestyle='--', label='mean_value')
# plt.axhline(min_value, color='yellow', linestyle='--', label='min_threshold')
# plt.axhline(max_value, color='black', linestyle='--', label='max_threshold')

# # Save the plot as an image (e.g., PNG format)
# plt.savefig("line_plot.png")

# # Show the plot (optional)
# plt.show()
