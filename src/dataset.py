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
                 num_images=None,
                 num_instances_per_image=None,
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
        self._length = 1024 

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
          
    def sample(self, source, size):
        max_id = len(source)
        if max_id > size:
            # sample with without repitition
            sample_idx = np.random.permutation(max_id)[:size]
        else:
            # not enough instances in this image
            # TODO: raise warning?
            # sample with repetition
            sample_idx = np.random.randint(max_id, size=size)

        if isinstance(source, list):
            return  [source[i] for i in sample_idx]
        else:
            return source[sample_idx]

    def __getitem__(self, _):
        image_instances = random.choice(self.data)
        image_instances = self.sample(image_instances, self.num_class_per_iteration)

        emb_data = []
        target_data = []
        target_idx = 0
        num_clicks_per_instance = self.num_queries + self.num_support
        for inst in image_instances:
            emb_sample = self.sample(inst["foreground"], num_clicks_per_instance)
            emb_data.append(emb_sample)

            target_data.append([target_idx]*len(emb_sample))
            target_idx += 1

        return np.concatenate(emb_data), np.concatenate(target_data)
