
import torch
import numpy as np
from PIL import Image
import open_clip
import h5py
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lightning as L
import torch
from torch.utils.data import DataLoader

from disent.dataset import DisentDataset
from disent.dataset.data import Shapes3dData
from disent.dataset.sampling import SingleSampler
from disent.dataset.transform import ToImgTensorF32
from disent.frameworks.vae import BetaVae
from disent.metrics import metric_dci
from disent.metrics import metric_mig
from disent.metrics import metric_unsupervised
from disent.metrics import metric_factor_vae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64
from disent.model.ae import EncoderConv64
from disent.schedule import CyclicSchedule


model_data_list=[('RN50', 'openai'),
 ('RN50', 'yfcc15m'),
 ('RN50', 'cc12m'),
 ('RN50-quickgelu', 'openai'),
 ('RN50-quickgelu', 'yfcc15m'),
 ('RN50-quickgelu', 'cc12m'),
 ('RN101', 'openai'),
 ('RN101', 'yfcc15m'),
 ('RN101-quickgelu', 'openai'),
 ('RN101-quickgelu', 'yfcc15m'),
 ('RN50x4', 'openai'),
 ('RN50x16', 'openai'),
 ('RN50x64', 'openai'),
 ('ViT-B-32', 'openai'),
 ('ViT-B-32', 'laion400m_e31'),
 ('ViT-B-32', 'laion400m_e32'),
 ('ViT-B-32', 'laion2b_e16'),
 ('ViT-B-32', 'laion2b_s34b_b79k'),
 ('ViT-B-32', 'datacomp_xl_s13b_b90k'),
 ('ViT-B-32', 'datacomp_m_s128m_b4k'),
 ('ViT-B-32', 'commonpool_m_clip_s128m_b4k'),
 ('ViT-B-32', 'commonpool_m_laion_s128m_b4k'),
 ('ViT-B-32', 'commonpool_m_image_s128m_b4k'),
 ('ViT-B-32', 'commonpool_m_text_s128m_b4k'),
 ('ViT-B-32', 'commonpool_m_basic_s128m_b4k'),
 ('ViT-B-32', 'commonpool_m_s128m_b4k'),
 ('ViT-B-32', 'datacomp_s_s13m_b4k'),
 ('ViT-B-32', 'commonpool_s_clip_s13m_b4k'),
 ('ViT-B-32', 'commonpool_s_laion_s13m_b4k'),
 ('ViT-B-32', 'commonpool_s_image_s13m_b4k'),
 ('ViT-B-32', 'commonpool_s_text_s13m_b4k'),
 ('ViT-B-32', 'commonpool_s_basic_s13m_b4k'),
 ('ViT-B-32', 'commonpool_s_s13m_b4k'),
 ('ViT-B-32-256', 'datacomp_s34b_b86k'),
 ('ViT-B-32-quickgelu', 'openai'),
 ('ViT-B-32-quickgelu', 'laion400m_e31'),
 ('ViT-B-32-quickgelu', 'laion400m_e32'),
 ('ViT-B-16', 'openai'),
 ('ViT-B-16', 'laion400m_e31'),
 ('ViT-B-16', 'laion400m_e32'),
 ('ViT-B-16', 'laion2b_s34b_b88k'),
 ('ViT-B-16', 'datacomp_xl_s13b_b90k'),
 ('ViT-B-16', 'datacomp_l_s1b_b8k'),
 ('ViT-B-16', 'commonpool_l_clip_s1b_b8k'),
 ('ViT-B-16', 'commonpool_l_laion_s1b_b8k'),
 ('ViT-B-16', 'commonpool_l_image_s1b_b8k'),
 ('ViT-B-16', 'commonpool_l_text_s1b_b8k'),
 ('ViT-B-16', 'commonpool_l_basic_s1b_b8k'),
 ('ViT-B-16', 'commonpool_l_s1b_b8k'),
 ('ViT-B-16-plus-240', 'laion400m_e31'),
 ('ViT-B-16-plus-240', 'laion400m_e32'),
 ('ViT-L-14', 'openai'),
 ('ViT-L-14', 'laion400m_e31'),
 ('ViT-L-14', 'laion400m_e32'),
 ('ViT-L-14', 'laion2b_s32b_b82k'),
 ('ViT-L-14', 'datacomp_xl_s13b_b90k'),
 ('ViT-L-14', 'commonpool_xl_clip_s13b_b90k'),
 ('ViT-L-14', 'commonpool_xl_laion_s13b_b90k'),
 ('ViT-L-14', 'commonpool_xl_s13b_b90k'),
 ('ViT-L-14-336', 'openai'),
 ('ViT-H-14', 'laion2b_s32b_b79k'),
 ('ViT-g-14', 'laion2b_s12b_b42k')]
model_data_list=model_data_list[:30]


dataset_name="SAT.images"
factors_size=(245, 115)
h5uri=f"./{dataset_name}.h5"
uri_hash="ce9e0b28bf99dd6c7f4154526622d827"
file_hash="f0ae0877c8c29469934bab34daef9665"


import logging

from disent.dataset.data._groundtruth import Hdf5GroundTruthData
from disent.dataset.util.datafile import DataFileHashedDlH5

class CustomHDF5(Hdf5GroundTruthData):
    """
    3D Shapes Dataset:
    - https://github.com/deepmind/3d-shapes

    Files:
        - direct:   https://storage.googleapis.com/3d-shapes/3dshapes.h5
          redirect: https://storage.cloud.google.com/3d-shapes/3dshapes.h5
          info:     https://console.cloud.google.com/storage/browser/_details/3d-shapes/3dshapes.h5
    """

    # TODO: name should be `shapes3d` so that it is a valid python identifier
    name = dataset_name

    factor_names = ("factor_name1", "factor_name2")
    factor_sizes = factors_size  # TOTAL: 480000
    img_shape = (256,256,3)

    datafile = DataFileHashedDlH5(
        uri=h5uri,
        uri_hash={"fast": uri_hash,
                  #"full": "099a2078d58cec4daad0702c55d06868"
                  },
        # processed dataset file
        file_hash={"fast": file_hash,
                   #"full": "b5187ee0d8b519bb33281c5ca549658c"
                   },
        # h5 re-save settings
        hdf5_dataset_name="images",
        hdf5_chunk_size=(1, 256,256,3),
        hdf5_obs_shape=img_shape,
    )


data=CustomHDF5(prepare=True)

def rep_func(x):
    feats=model.encode_image(x.to(device))
    feats/=feats.norm(dim=-1, keepdim=True)
    print(feats.shape)
    return feats

#data = h5py.File('3dshapes.h5', 'r')['images']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metric=metric_dci
data=CustomHDF5(prepare=True)
result_dict={"Model":[],"Dataset":[]}
for model_name,dataset_name in model_data_list:
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=dataset_name)
    model.eval()
    with torch.no_grad():
        preprocess.transforms.insert(0,transforms.ToPILImage())
        dataset = DisentDataset(dataset=data, sampler=SingleSampler(), transform=preprocess)
        model=model.to(device)
        get_repr = lambda x: rep_func(x)
        metrics = {
        **metric_dci(dataset, get_repr, num_train=1000, num_test=500, show_progress=True),
        #**metric_mig(dataset, get_repr, num_train=69),
        #**metric_factor_vae(dataset,get_repr,num_train=49,num_eval=20,batch_size=16,show_progress=True)
        #**metric_unsupervised(dataset,get_repr,num_train=1900)
        #**metric_sap(dataset,get_repr)
        #**metric_flatness(dataset,get_repr,num_train=1000)
        }
        print(model_name,dataset_name)
        print(metrics)
        print("============")
        result_dict["Model"].append(model_name)
        result_dict["Dataset"].append(dataset_name)
        for key in metrics.keys():
            if key not in result_dict.keys():
                result_dict[key]=[]
            result_dict[key].append(metrics[key])
print('metrics:', metrics)
import pandas as pd
df=pd.DataFrame.from_dict(result_dict)
df.to_csv("results.csv",index=False)