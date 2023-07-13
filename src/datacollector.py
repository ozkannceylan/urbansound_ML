import os
import time
import pandas as pd
import numpy as np

class DataCollector:
    def __init__(self, folder_paths, metadata_path):
        self.folder_paths = folder_paths
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(self.metadata_path)
        
    def collect_data_paths(self, start_fold=0, end_fold=9):
        s = time.time()
        data_paths = []

        for i in range(start_fold, end_fold):
            curr_folder_path = self.folder_paths[i]

            if curr_folder_path == self.metadata_path:
                continue

            curr_files = os.listdir(curr_folder_path)
            file_paths = [os.path.join(curr_folder_path, file) for file in curr_files]

            for j in range(len(file_paths)):
                curr_file = curr_files[j]
                class_id = self.metadata[self.metadata['slice_file_name'] == curr_file]
                arr = np.array(class_id['classID'])

                data_paths.append([file_paths[j], arr[0]])

        e = time.time()
        print((e - s) / 60, " mins")
        print(len(data_paths))
        
        return data_paths
