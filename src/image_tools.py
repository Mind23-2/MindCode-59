import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose
import mindspore as ms
from mindspore.common.tensor import Tensor
import numpy as np
import mindspore.dataset as ds


def convert_image_to_tensor(image):
    """convert an image to pytorch tensor

        Parameters:
        ----------
        image: numpy array , h * w * c

        Returns:
        -------
        image_tensor: pytorch.FloatTensor, c * h * w
        """
    data_set = ds.ImageFolderDataset(image)
    trans = [py_vision.HWC2CHW()]

    return data_set.map(operations=trans, input_columns="image")



def convert_chwTensor_to_hwcNumpy(tensor):
    """convert a group images pytorch tensor(count * c * h * w) to numpy array images(count * h * w * c)
            Parameters:
            ----------
            tensor: numpy array , count * c * h * w

            Returns:
            -------
            numpy array images: count * h * w * c
            """

    return np.transpose(tensor.numpy(), (0,2,3,1))
