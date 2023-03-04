import nibabel as nib
import os
from skimage.transform import resize
import pickle
import numpy as np

def load_labels(path: str)-> map:
  return {
    'OAS2_0002_MR1': 1,
    'OAS2_0002_MR2': 0,
    'OAS2_0002_MR3': 1,
    'OAS2_0007_MR1': 0,
    'OAS2_0007_MR3': 1,
  }

def load_image(path: str)-> map:
  res = {}
  for base in os.listdir(path):
    suffix = r'/RAW/mpr-1.nifti.hdr'
    absolute_path = os.path.join(path, base + suffix)
    img = nib.load(absolute_path).get_fdata() # 原大小为(256, 256, 128, 1)
    img = resize(img, (128,128,128), order=0) # 这一步已经将memmap 对象转化为标准的内存对象，需要保证内存大小，大小为(128, 128, 128, 1)
    res[base] = np.resize(img, (1, 128, 128, 128))
  return res

class Dataset():
  def __init__(self):
    print("Dataset initializing...")
    self.dataset = []
  def create_dataset(self, imgs_path: str, labels_path: str, path: str):
    print("Creating dataset...")
    print("images path: ", imgs_path)
    print("labels path: ", labels_path)
    labels = load_labels(labels_path)
    imgs = load_image(imgs_path)
    print("num of images: ", len(imgs))
    print("num of labels: ", len(labels))
    for key in imgs.keys():
      print("imgs: ", key)
      self.dataset.append([imgs[key], labels[key]])
    print("Dataset created!")
    print("Dataset saving in: ", path)
    file = open(save_path, 'wb')
    pickle.dump(self.dataset, file)
    file.close()
    print("Finished")
  def load_dataset(self, dataset_path: str) -> list:
    print("Loading dataset...")
    print("dataset path: ", dataset_path)
    file = open(dataset_path, 'rb')
    self.dataset = pickle.load(file)
    file.close()
    print("Dataset loaded!")
    return self.dataset
  def get_split_dataset(self):
    imgs = [img_label[0] for img_label in self.dataset]
    labels = [img_label[1] for img_label in self.dataset]
    return np.array(imgs), np.resize(np.array(labels), (len(self.dataset), 1))

if __name__ == '__main__':
  imgs_path = "/home/ubuntu/labfield/DATA"
  labels_path = "./data/labels.txt"
  save_path = "./data/dataset"

  dataset_save = Dataset()
  dataset_save.create_dataset(imgs_path, save_path, save_path)

  dataset_load = Dataset()
  dataset = dataset_load.load_dataset(save_path)
  print("size of dataset: ", len(dataset))

  for i in range(len(dataset)):
    print("label of image %d : %d" % (i, dataset[i][1]))
    print("type of image %d: " % i, type(dataset[i][0]))
  
  imgs, labels = dataset_save.get_split_dataset()
  print("amount of images", len(imgs))
  print("labels: ", labels)
