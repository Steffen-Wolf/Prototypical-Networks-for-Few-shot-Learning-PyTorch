from torch.utils.data import Dataset
import numpy as np
import zarr
import random
from itertools import islice

class FastDataset(Dataset):
    """
    Attributes
    ----------
    ds_file: str
        File to zarr array that contains dataset
    num_queries: int
        Number of query points sampled at every iteration
    num_support: int
        Number of support points sampled at every iteration
    num_class_per_iteration: int
        Number of different instances sampled at every iteration
    lim_images: int
        Total number of images in the dataset (None = full dataset)
    lim_instances_per_image: int
        Total number of instances per image (None = all instances)
    lim_clicks_per_instance: int
        Total number of clicks per instance (None = all instances)
    """
    def __init__(self,
                 ds_file,
                 num_queries=5,
                 num_support=2,
                 num_class_per_iteration=5,
                 lim_images=None,
                 lim_instances_per_image=None,
                 lim_clicks_per_instance=None):
        super().__init__()
        self.ds_file = ds_file
        self.cache = zarr.open(ds_file, "r")
        self._data = None
        self.num_queries = num_queries
        self.num_support = num_support
        self.num_class_per_iteration = num_class_per_iteration
        self.lim_images = lim_images
        self.lim_instances_per_image = lim_instances_per_image
        self.lim_clicks_per_instance = lim_clicks_per_instance
        self._length = 2048
        assert(lim_clicks_per_instance >= num_queries + num_support)

    def __len__(self):
        return self._length

    @property
    def data(self):
        if self._data is None:
            self._data = []

            ds_array = zarr.open(self.ds_file, "r")
            for img_key in islice(ds_array, self.lim_images):
                image_instances = []
                for instance_key in islice(ds_array[img_key], self.lim_instances_per_image):
                    image_data = {}
                    image_data["foreground"] = ds_array[img_key][instance_key]["foreground"][:self.lim_clicks_per_instance]
                    image_data["background"] = ds_array[img_key][instance_key]["background"][:self.lim_clicks_per_instance]
                    image_instances.append(image_data)
            self._data.append(image_instances)
        return self._data

    @property
    def labeled_pixels(self):
        num_labeled_pixels = 0
        for img_instanges in self.data:
            for inst in img_instanges:
                num_labeled_pixels += len(inst["foreground"])
        return num_labeled_pixels

    def sample(self, source, size, seed):
        max_id = len(source)
        if max_id <= size:
            return source
        # use index as seed to get a random, but reproducable order
        sample_idx = np.random.choice(max_id, size, replace=False)
        if isinstance(source, list):
            return [source[i] for i in sample_idx]
        else:
            return source[sample_idx]

    def __getitem__(self, seed):
        image_instances = self.sample(random.choice(self.data), self.num_class_per_iteration, seed)

        emb_data = []
        target_data = []
        target_idx = 0
        num_clicks_per_instance = self.num_queries + self.num_support
        for inst in image_instances:
            emb_sample = self.sample(inst["foreground"], num_clicks_per_instance, seed+1)
            emb_data.append(emb_sample)

            target_data.append([target_idx]*len(emb_sample))
            target_idx += 1

        return np.concatenate(emb_data), np.concatenate(target_data)
