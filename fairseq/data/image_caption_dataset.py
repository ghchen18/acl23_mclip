import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from fairseq.data import FairseqDataset, data_utils
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import h5pickle as h5py 

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import warnings
warnings.filterwarnings("ignore")


def rgb_func(x):
    return x.convert("RGB")

class ImageDataset(FairseqDataset):
    def __init__(self, image_dir, image_ids, cfg = None):
        self.image_ids = image_ids
        self.image_dir = image_dir 
        self.vit_type = getattr(cfg.dataset, 'ViT_type', 'clip')
        self.fixed_num_tokens = 5
        self.transform = self.get_transform()

    def get_transform(self, resolution = 224):
        if resolution == 224:
            n_px = 224 
        elif resolution == 288:
            n_px = 320 
        elif resolution == 384:
            n_px = 416
        
        if 'clip' in self.vit_type:
            return Compose([
                Resize(248, interpolation=BICUBIC),
                CenterCrop(224),
                Lambda(rgb_func),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        elif 'swinT' in self.vit_type:
            return Compose([
                Resize(size=248, interpolation=BICUBIC),
                CenterCrop(224),
                Lambda(rgb_func),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            print("Error for transform image...") 

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_path = f"{self.image_dir}/{self.image_ids[idx]}"
        with Image.open(image_path) as img:
            return self.transform(img), self.image_ids[idx]
        
    def num_tokens(self, index):
        return self.fixed_num_tokens

    def size(self, index):
        return self.fixed_num_tokens
    
    @property
    def sizes(self):
        size_array = np.ones(len(self.image_ids), dtype=np.int) * self.fixed_num_tokens
        return size_array


class ImageCaptionDataset(FairseqDataset):
    def __init__(self, image_ids, split, cap_ds, cap_dict, shuffle=True, cfg=None, src_lang_id = None, tgt_lang_id = None, tgt_lang='en'):
        self.read_raw_images =  getattr(cfg.task, 'read_raw_images', False)
        self.cfg = cfg
        self.image_ids = image_ids 

        if split == 'train':
            features_dir = self.cfg.task.mscoco_dir 
        else:
            features_dir = self.cfg.task.mscoco_dir  ## You need to check and revise this path
        self.img_ds = ImageDataset(features_dir, image_ids, cfg = cfg)
        
        self.cap_ds = cap_ds
        self.cap_dict = cap_dict
        self.shuffle = shuffle
        
        self.src_lang_id = torch.LongTensor([src_lang_id]) if src_lang_id is not None else None 
        self.tgt_lang_id = torch.LongTensor([tgt_lang_id]) if tgt_lang_id is not None else None 

        cap_sizes = self.cap_ds.sizes if len(self.cap_ds.sizes.shape) < 2 else self.cap_ds.sizes[:, -1]    
        image_sizes = np.expand_dims(np.ones_like(cap_sizes), -1) * 5
        self.sizes = np.concatenate((image_sizes, np.expand_dims(cap_sizes, -1)), axis = 1)

    def __getitem__(self, index):
        try:
            source_features, _ = self.img_ds[index]
            if isinstance(self.cap_ds[index], dict):
                source_tokens = self.cap_ds[index]['source']
                target = self.cap_ds[index]['target']
            else:
                target = self.cap_ds[index]
                source_tokens = target[:2]
        except BaseException as err:
            index = index - 1 if index > 0 else index + 1
            return  self.__getitem__(index)

        if 'coco' in self.cfg.task.mscoco_dir:
            image_id = self.image_ids[index]
            image_id = image_id.split('_')[-1]
            image_id = image_id.split('.')[0]
            image_id = torch.Tensor([int(image_id)]).type_as(target)
        elif 'multi30k' in self.cfg.task.mscoco_dir:
            image_id = self.image_ids[index]
            image_id = image_id.split('/')[-1]
            image_id = image_id.split('.')[0]
            image_id = torch.Tensor([int(image_id)]).type_as(target)
        else:
            image_id = torch.Tensor([0]).type_as(target)

        return {
            'id': index,
            'source': source_tokens,
            'features': source_features,
            'target': target,
            'src_lang_id': self.src_lang_id,
            'tgt_lang_id': self.tgt_lang_id,
            'image_id': image_id, 
        }

    def __len__(self):
        return len(self.img_ds)

    def num_tokens(self, index):
        return self.size(index)[1]

    def size(self, index):
        if len(self.cap_ds.sizes[index]) > 1:
            cap_size = self.cap_ds.sizes[index][-1]
        else:
            cap_size = self.cap_ds.sizes[index] 
        # number of image feature vectors, number of tokens in caption
        return self.img_ds.sizes[index], cap_size 
    
    def prefetch(self, indices):
        self.cap_ds.prefetch(indices)

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))

        # Inspired by LanguagePairDataset.ordered_indices
        indices = indices[np.argsort(self.cap_ds.sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.img_ds.sizes[indices], kind='mergesort')]

    def collater(self, samples, pad_to_length=None):
        indices = []
        source_feature_samples = []
        source_token_samples = []
        source_lengths = []
        image_ids_samples = []
        target_samples = []
        target_ntokens = 0

        for sample in samples:
            index = sample['id']
            indices.append(index)
            if sample['source'] is not None:
                source_token_samples.append(sample['source'])

            source_feature_samples.append(sample['features'])           
            source_lengths.append(self.img_ds.sizes[index])
            image_ids_samples.append(sample['image_id'])  
            
            target_samples.append(sample['target'])
            target_ntokens += self.cap_ds.sizes[index] if len(self.cap_ds.sizes[index].shape) == 0 else self.cap_ds.sizes[index][-1]
            
        source_lengths = torch.LongTensor(source_lengths)
        num_sentences = len(samples)
        if num_sentences == 0:
            return None

        indices = torch.tensor(indices, dtype=torch.long)
        source_feature_batch = default_collate(source_feature_samples)
        source_location_batch = None  #default_collate(source_location_samples)
        image_ids_batch = default_collate(image_ids_samples)

        target_batch = data_utils.collate_tokens(target_samples,
                                                pad_idx=self.cap_dict.pad(),
                                                eos_idx=self.cap_dict.eos(),
                                                left_pad=False,
                                                move_eos_to_beginning=False)
        
        rotate_batch = data_utils.collate_tokens(target_samples,
                                                pad_idx=self.cap_dict.pad(),
                                                eos_idx=self.cap_dict.eos(),
                                                left_pad=False,
                                                move_eos_to_beginning=True)
        if len(source_token_samples) > 0:
            source_batch = data_utils.collate_tokens(source_token_samples,
                                                pad_idx=self.cap_dict.pad(),
                                                eos_idx=self.cap_dict.eos(),
                                                left_pad=True,
                                                move_eos_to_beginning=False)
        else:
            source_batch = None 


        src_lang_id = self.src_lang_id.repeat(num_sentences, 1)
        tgt_lang_id = self.tgt_lang_id.repeat(num_sentences, 1)

        return {
            'id': indices,
            'net_input': {
                'src_tokens': source_batch,
                'src_lengths': source_lengths,
                'features': source_feature_batch,
                'prev_output_tokens': rotate_batch,
                'src_lang_id': src_lang_id,
                'tgt_lang_id': tgt_lang_id,
            },
            'target': target_batch,
            'ntokens': target_ntokens,
            'nsentences': num_sentences,
            'image_ids': image_ids_batch,
        }
