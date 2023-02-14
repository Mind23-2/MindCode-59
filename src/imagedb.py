import os
import numpy as np
import mindspore.dataset as de
from mindspore.mindrecord import FileWriter
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.engine as de

class ImageDB(object):
    def __init__(self, image_annotation_file, prefix_path='', mode='train'):
        self.prefix_path = prefix_path
        self.image_annotation_file = image_annotation_file
        self.classes = ['__background__', 'face']
        self.num_classes = 2
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        self.mode = mode


    def load_image_set_index(self):
        """Get image index

        Parameters:
        ----------
        Returns:
        -------
        image_set_index: str
            relative path of image
        """
        assert os.path.exists(self.image_annotation_file), 'Path does not exist: {}'.format(self.image_annotation_file)
        with open(self.image_annotation_file, 'r') as f:
            image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
        return image_set_index


    def load_imdb(self):
        """Get and save ground truth image database

        Parameters:
        ----------
        Returns:
        -------
        gt_imdb: dict
            image database with annotations
        """
        #cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        #if os.path.exists(cache_file):
        #    with open(cache_file, 'rb') as f:
        #        imdb = cPickle.load(f)
        #    print '{} gt imdb loaded from {}'.format(self.name, cache_file)
        #    return imdb

        gt_imdb = self.load_annotations()

        #with open(cache_file, 'wb') as f:
        #    cPickle.dump(gt_imdb, f, cPickle.HIGHEST_PROTOCOL)
        return gt_imdb


    def real_image_path(self, index):
        """Given image index, return full path

        Parameters:
        ----------
        index: str
            relative path of image
        Returns:
        -------
        image_file: str
            full path of image
        """

        index = index.replace("\\", "/")

        if not os.path.exists(index):
            image_file = os.path.join(self.prefix_path, index)
        else:
            image_file=index
        if not image_file.endswith('.jpg'):
            image_file = image_file + '.jpg'
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file


    def load_annotations(self,annotion_type=1):
        """Load annotations

        Parameters:
        ----------
        annotion_type: int
                      0:dsadsa
                      1:dsadsa
        Returns:
        -------
        imdb: dict
            image database with annotations
        """

        assert os.path.exists(self.image_annotation_file), 'annotations not found at {}'.format(self.image_annotation_file)
        with open(self.image_annotation_file, 'r') as f:
            annotations = f.readlines()

        imdb = []
        for i in range(self.num_images):
            annotation = annotations[i].strip().split(' ')
            index = annotation[0]
            im_path = self.real_image_path(index)
            imdb_ = dict()
            imdb_['image'] = im_path

            if self.mode == 'test':
               # gt_boxes = map(float, annotation[1:])
               # boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
               # imdb_['gt_boxes'] = boxes
                pass
            else:
                label = annotation[1]
                imdb_['label'] = int(label)
                imdb_['flipped'] = 0
                imdb_['bbox_target'] = np.zeros((4,))
                imdb_['landmark_target'] = np.zeros((10,))
                if len(annotation[2:])==4:
                    bbox_target = annotation[2:6]
                    imdb_['bbox_target'] = np.array(bbox_target).astype(float)
                if len(annotation[2:])==14:
                    bbox_target = annotation[2:6]
                    imdb_['bbox_target'] = np.array(bbox_target).astype(float)
                    landmark = annotation[6:]
                    imdb_['landmark_target'] = np.array(landmark).astype(float)
            imdb.append(imdb_)

        return imdb


    def append_flipped_images(self, imdb):
        """append flipped images to imdb

        Parameters:
        ----------
        imdb: imdb
            image database
        Returns:
        -------
        imdb: dict
            image database with flipped image annotations added
        """
        print('append flipped images to imdb', len(imdb))
        for i in range(len(imdb)):
            imdb_ = imdb[i]
            m_bbox = imdb_['bbox_target'].copy()
            m_bbox[0], m_bbox[2] = -m_bbox[2], -m_bbox[0]

            landmark_ = imdb_['landmark_target'].copy()
            landmark_ = landmark_.reshape((5, 2))
            landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]
            landmark_[[3, 4]] = landmark_[[4, 3]]

            item = {'image': imdb_['image'],
                     'label': imdb_['label'],
                     'bbox_target': m_bbox,
                     'landmark_target': landmark_.reshape((10)),
                     'flipped': 1}

            imdb.append(item)
        self.image_set_index *= 2
        print('after append flipped images to imdb', len(imdb))
        return imdb


def data_to_mindrecord_byte_image(imdb, prefix="", file_num=1, mindrecord_dir=""):
    """Create MindRecord file."""
    mindrecord_path = os.path.join(mindrecord_dir, prefix)

    writer = FileWriter(mindrecord_path, file_num)

    mtcnn_json = {
        "image": {"type": "bytes"},
        "label": {"type": "int32"},
        "bbox_target": {"type": "float32", "shape": [4]},
        "landmark_target": {"type": "float32", "shape": [10]},
    }
    writer.add_schema(mtcnn_json, "mtcnn_json")

    for i in range(len(imdb)):
        imdb_ = imdb[i]
        with open(imdb_['image'], 'rb') as f:
            image = f.read()
        label = imdb_['label']
        bbox_target = imdb_['bbox_target']
        landmark_target = imdb_['landmark_target']

        row = {"image": image,
               "label": label,
               "bbox_target": bbox_target,
               "landmark_target": landmark_target
               }
        if i % 1000 == 0:
            print("writing {} into mindrecord".format(i + 1))
        writer.write_raw_data([row])
    writer.commit()
    print("create mindrecord done")


def create_mtcnn_dataset(mindrecord_file, batch_size=128, device_num=1, rank_id=0,
                         is_training=True, num_parallel_workers=8):
    """Create MTCNN dataset with MindDataset."""
    ds = de.MindDataset(mindrecord_file,
                        columns_list=["image", "label", "bbox_target", "landmark_target"],
                        num_shards=device_num, shard_id=rank_id,
                        num_parallel_workers=4)

    train_list = [C.Decode(), C.HWC2CHW()]
    ds = ds.map(operations=train_list, input_columns=["image"])
    ds = ds.batch(batch_size, drop_remainder=True)

    return ds

def create_eval_dataset(dataset_path, target = "Ascend", device_id = 1):
    ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=False)
    train_list = [C.Decode(), C.HWC2CHW()]
    ds = ds.map(operations=train_list, input_columns=["image"])
    ds = ds.batch(batch_size, drop_remainder=True)

    return ds
    

if __name__ == '__main__':
    mindrecord_file = "/root/xidian_wks/mtcnn/data_set/img12.mindrecord"
    dataset = create_mtcnn_dataset(mindrecord_file, batch_size=128, device_num=1,
                                   rank_id=0)
    count = 0
    for item in dataset.create_dict_iterator(output_numpy=True):
        print("sample: {}".format(item))
        count += 1
    print("Got {} samples".format(count))




    

