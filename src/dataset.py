from torch.utils.data import Dataset
import numpy as np
import zarr
import random
from itertools import islice
from functools import partial
from skimage.io import imsave
from scipy.ndimage import zoom

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
                 batchsize=4,
                 num_class_per_iteration=None,
                 lim_images=None,
                 lim_instances_per_image=None,
                 lim_clicks_per_instance=None,
                 emb_key="embedding",
                 crop_to=(256, 256),
                 length=2048):
        super().__init__()
        self.ds_file = ds_file
        self.cache = zarr.open(ds_file, "r")
        self._data = None
        self._aug_choices = None
        self.num_queries = num_queries
        self.num_support = num_support
        self.num_class_per_iteration = num_class_per_iteration
        self.lim_images = lim_images
        self.lim_instances_per_image = lim_instances_per_image
        self.lim_clicks_per_instance = lim_clicks_per_instance
        self._length = length
        self._aug_scale = 0.5
        self._aug_pos = None
        self.emb_key = emb_key
        self.batchsize = batchsize
        self.crop_to = crop_to

        # trigger data generation to avoid timeout
        # during first iterations
        self.data

        assert (lim_clicks_per_instance is None) or (lim_clicks_per_instance >= num_queries + num_support)

    def __len__(self):
        return self._length

    def check_emb_keys(self, ds):
        if isinstance(self.emb_key, (list, tuple)):
            for k in self.emb_key:
                if k not in ds:
                    print(f"could not find {k} in {self.ds_file} at {k}")
                    return False
        else:
            if self.emb_key not in ds:
                return False
        return True

    @property
    def data(self):
        if self._data is None:
            self._data = []

            ds_array = zarr.open(self.ds_file, "r")
            for img_key in islice(ds_array, self.lim_images):
                
                coordinates = {}
                coordinates["coords"] = []
                if "raw" not in ds_array[img_key]:
                    print("could not find raw in ",
                          self.ds_file, " at ", img_key)
                    continue

                if not self.check_emb_keys(ds_array[img_key]):
                    continue

                coordinates["raw"] = ds_array[img_key]["raw"]
                if isinstance(self.emb_key, (list, tuple)):
                    coordinates["embedding"] = [ds_array[img_key][k] for k in self.emb_key]
                else:
                    coordinates["embedding"] = ds_array[img_key][self.emb_key]

                for instance_key in islice(ds_array[img_key]["foreground"], self.lim_instances_per_image):
                    image_data = {}
                    # shift = None
                    # if raw.shape[-2:] != self.crop_to:
                    #     # crop image around the center
                    #     shift = ((raw.shape[-2] - self.crop_to[-2])//2, (raw.shape[-1] - self.crop_to[-1])//2)
                    #     print("fg object ", fg[:10])
                    #     raw = self.crop(raw, shift)

                    fg = ds_array[img_key]["foreground"][instance_key][:]
                    bg = ds_array[img_key]["background"][instance_key][:]
                        # fg, _ = self.crop_coords(fg, shift, self.crop_to)
                        # bg, _ = self.crop_coords(bg, shift, self.crop_to)
                        # print("after crop fg,", fg.shape)
                        # print("after crop bg,", bg.shape)
                    image_data["foreground"] = fg[:self.lim_clicks_per_instance]
                    image_data["background"] = bg[:self.lim_clicks_per_instance]

                    coordinates["coords"].append(image_data)
                assert len(
                    coordinates["coords"]), f"no instances found in {self.ds_file}/{img_key}"
                self._data.append(coordinates)
        return self._data

    @property
    def labeled_images_instances_pixels(self):
        num_labeled_pixels = 0
        num_labeled_images = 0
        num_labeled_instances = 0
        for img_instanges in self.data:
            num_labeled_images += 1
            for inst in img_instanges["coords"]:
                num_labeled_instances += 1
                num_labeled_pixels += len(inst["foreground"])

        return num_labeled_images, num_labeled_instances, num_labeled_pixels

    def sample(self, source, size):
        max_id = len(source)
        if max_id <= size:
            return source

        sample_idx = np.random.choice(max_id, size, replace=False)
        return [source[i] for i in sample_idx]
        
    def flip(self, img, axis=(-2)):
        return np.ascontiguousarray(np.flip(img, axis=axis))
    
    def flip_coord(self, coord, axis=(-2, ), img_shape=(256, 256)):
        for a in axis:
            # switch coordinates
            b = {-2: -1, -1: -2}[a]
            coord[:, b] = img_shape[b] - coord[:, b] - 1
        return coord

    def rot(self, img, k=1):
        return np.ascontiguousarray(np.rot90(img, k=k, axes=(-2,-1)))

    def rot_coord(self, coord, k=1, img_shape=(256, 256)):
        coord_out = np.copy(coord)
        if k == 1:
            coord_out[:, -2] = coord[:, -1]
            coord_out[:, -1] = img_shape[-2] - coord[:, -2] - 1
        elif k == 2:
            coord_out[:, -2] = img_shape[-2] - coord[:, -2] - 1
            coord_out[:, -1] = img_shape[-1] - coord[:, -1] - 1
        elif k == 3:
            coord_out[:, -2] = img_shape[-1] - coord[:, -1] - 1
            coord_out[:, -1] = coord[:, -2]          
        elif k > 3:
            raise NotImplementedError()
        return coord_out

    def transpose(self, img):
        return np.ascontiguousarray(np.transpose(img, (-2, -1)))
        
    def transpose_coord(self, coord, img_shape=(256, 256)):
        coord_out = np.copy(coord)
        coord_out[:, -2] = coord[:, -2]
        coord_out[:, -1] = coord[:, -1]
        return coord_out

    def update_aug_scale(self):
        self._aug_scale = random.uniform(0.5, 1.)

    def scale(self, img):
        zf = ([1, ] * (img.ndim - 2)) + ([self._aug_scale,] * 2)
        scaled_img = zoom(img, zf, mode='reflect')
        out_img = 0 * img
        w = scaled_img.shape[-2]
        h = scaled_img.shape[-1]
        out_img[..., :w, :h] = scaled_img
        return out_img

    def scale_cood(self, coord, img_shape=(256, 256)):
        coord_out = np.copy(coord)
        coord_out[:, -2] = self._aug_scale * coord[:, -2]
        coord_out[:, -1] = self._aug_scale * coord[:, -1]
        return coord_out

    @property
    def aug_choices(self):
        if self._aug_choices is None:
            self._aug_choices = [(None, None)]
                # (self.transpose, self.transpose_coord)]
            for k in [1, 2, 3]:
                self._aug_choices.append((partial(self.rot, k=k),
                                        partial(self.rot_coord, k=k)))
            for axis in [(-2, ), (-1, ), (-2, -1)]:
                self._aug_choices.append((partial(self.flip, axis=axis),
                                          partial(self.flip_coord, axis=axis)))
        return self._aug_choices

    def get_random_augment(self):
        self.update_aug_scale()
        return random.choice(self.aug_choices)

    def get_crop(self, img, coord):
        if any((a > b for a, b in zip(img.shape, self.crop_to))):
            ox = random.randint(0, img.shape[0]-self.crop_to[0]-1)
            oy = random.randint(0, img.shape[1]-self.crop_to[1]-1)
            shift = (ox, oy)
            if len(self.in_bound_mask(coord, shift)) == 0:
                # get new random shift
                return self.get_crop(img, coord)
            else:
                return shift
        return None

    def crop(self, img, shift):
        print("croping ", img.shape, self.crop_to)
        ox, oy = shift
        return img[..., ox:ox+self.crop_to[0], oy:oy+self.crop_to[1]]

    def in_bound_mask(self, coord, shift):
        m0 = coord[:, 0] >= shift[0]
        m1 = coord[:, 1] >= shift[1]
        m2 = coord[:, 0] < shift[0] + self.crop_to[0]
        m3 = coord[:, 1] < shift[1] + self.crop_to[1]
        return np.logical_and.reduce((m0, m1, m2, m3))

    def crop_coords(self, coord, shift, imgshape):
        ibm = self.in_bound_mask(coord, shift)
        coord = coord[ibm].copy()
        coord[:, 0] -= shift[0]
        coord[:, 1] -= shift[1]
        return coord, ibm
        
    
    def __getitem__(self, _):

        if self.num_class_per_iteration is None:
            image_instances = random.choice(self.data)
        else:
            image_instances = self.sample(random.choice(self.data), self.num_class_per_iteration)

        instance_coordinates = []
        bg_coordinates = []
        target_data = []
        target_idx = 0
        num_clicks_per_instance = self.num_queries + self.num_support
        raw = image_instances["raw"][:].astype(np.float32)

        if isinstance(self.emb_key, (list, tuple)):
            e = [esqueeze(q[:]).astype(np.float32)
                 for q in image_instances["embedding"]]
            embedding = np.concatenate(e, axis=0)
        else:
            embedding = image_instances["embedding"][:].astype(np.float32)

        for inst in image_instances["coords"]:
            coord_sample = self.sample(inst["foreground"],
                                       num_clicks_per_instance)
            bg_coord_sample = self.sample(inst["background"],
                                          num_clicks_per_instance*3)
            instance_coordinates.append(coord_sample)
            bg_coordinates.append(bg_coord_sample)

            target_data.append([target_idx]*len(coord_sample))
            target_idx += 1

        # raw, inp, instance_coordinates, y, background_coordinates
        instance_coordinates = np.concatenate(instance_coordinates).astype(np.int64)
        bg_coordinates = np.concatenate(bg_coordinates).astype(np.int64)
        target_data = np.concatenate(target_data)
        
        # shift = self.get_crop(raw, instance_coordinates)

        # if shift is not None:
        #     ic, mask = self.crop_coords(
        #         instance_coordinates, shift, self.crop_to)
        #     bc, _ = self.crop_coords(bg_coordinates, shift, self.crop_to)
        #     min_len = self.num_queries + self.num_support

        #     if len(ic) <= min_len or len(bc) <= min_len:
        #         return self.__getitem__(0)

        #     # while len(ic) <= min_len or len(bc) <= min_len:
        #         # shift = self.get_crop(raw, instance_coordinates)
        #         # ic, mask = self.crop_coords(
        #         #     instance_coordinates, shift, self.crop_to)
        #         # bc, _ = self.crop_coords(bg_coordinates, shift, self.crop_to)

        #     instance_coordinates = ic
        #     bg_coordinates = bc

        #     target_data = target_data[mask]
        #     raw = self.crop(raw, shift)
        #     embedding = self.crop(embedding, shift)
        # print(raw.shape, embedding.shape)
        # assert(len(instance_coordinates) > self.num_queries + self.num_support)
        # assert(len(bg_coordinates) > self.num_queries + self.num_support)
        return raw[None, None], embedding[None], instance_coordinates, target_data, bg_coordinates

def esqueeze(x):
    if x.ndim == 3:
        return x
    if x.ndim < 3:
        return esqueeze(x[None])
    if x.ndim > 3:
        return esqueeze(x[0])
