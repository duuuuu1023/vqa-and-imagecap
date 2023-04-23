# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle
import base64

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000
from PIL import Image
import numpy as np
import random
import os

from transformers import BertTokenizer

from .utils import nested_tensor_from_tensor_list, read_json

MAX_DIM = 224


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = tv.transforms.Compose([
    RandomRotation(),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class CocoCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = [(self._process(val['image_id']), val['caption'])
                      for val in ann['annotations']]
        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            self.annot = self.annot[: limit]

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True)
        self.max_length = max_length + 1

    def _process(self, image_id):
        val = str(image_id).zfill(12)
        return val + '.jpg'

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
            1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask
# The path to data and image features.
VQA_DATA_ROOT = '/root/autodl-tmp/Transformers-VQA/data/'#autodl-tmp/Transformers-VQA/data
MSCOCO_IMGFEAT_ROOT = '/root/autodl-tmp/Transformers-VQA/data/img/'
SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/trainval_label2ans.json"))
        
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset, model = 'lxmert'):
        super().__init__()
        self.raw_dataset = dataset
        self.model = model
        if args.tiny:
            topk = TINY_IMG_NUM
            self.raw_dataset.data = self.raw_dataset.data[:topk]
        elif args.fast:
            topk = FAST_IMG_NUM
            self.raw_dataset.data = self.raw_dataset.data[:topk]
        else:
            topk = None
        self.offset = {}
        for split in self.raw_dataset.splits:
            f = open(os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_offset.txt' % (SPLIT2NAME[split])))
            offset = f.readlines()
            for l in offset:
                self.offset[l.split('\t')[0]] = int(l.split('\t')[1].strip())
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.offset.keys():
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item: int):
        datum = self.data[item]
        img_id = datum['img_id']
        #print(img_id)
        ques_id = datum['question_id']
        ques = datum['sent']

        img_offset = self.offset[img_id]
        img_split = img_id[5:7]
        if(img_split == 'tr'):
            f = open(os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME['train'])))
        elif(img_split == 'va'):
            f = open(os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME['valid'])))
        else:
            f = open(os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME['test'])))
        f.seek(img_offset)
        img_info = f.readline()
        f.close()
        
        assert img_info.startswith('COCO') and img_info.endswith('\n'), 'Offset is inappropriate'
        img_info = img_info.split('\t')

        decode_img = self._decodeIMG(img_info)
        img_h = decode_img[0]
        img_w = decode_img[1]
        feats = decode_img[-1].copy()
        boxes = decode_img[-2].copy()
        del decode_img

        # Normalize the boxes (to 0 ~ 1)
        if self.model == 'uniter':
            boxes = self._uniterBoxes(boxes)
        else:
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h

            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)


        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques



    def _decodeIMG(self, img_info):
        img_h = int(img_info[1])
        img_w = int(img_info[2])
        boxes = img_info[-2]
        boxes = np.frombuffer(base64.b64decode(boxes), dtype=np.float32)
        boxes = boxes.reshape(36,4)
        boxes.setflags(write=False)
        feats = img_info[-1]
        feats = np.frombuffer(base64.b64decode(feats), dtype=np.float32)
        feats = feats.reshape(36,-1)
        feats.setflags(write=False)
        return [img_h, img_w, boxes, feats]
    
    def _uniterBoxes(self, boxes):
        new_boxes = np.zeros((boxes.shape[0],7),dtype='float32')
        new_boxes = np.zeros((boxes.shape[0],7),dtype='float32')
        new_boxes[:,1] = boxes[:,0]
        new_boxes[:,0] = boxes[:,1]
        new_boxes[:,3] = boxes[:,2]
        new_boxes[:,2] = boxes[:,3]
        new_boxes[:,4] = new_boxes[:,3]-new_boxes[:,1]
        new_boxes[:,5] = new_boxes[:,2]-new_boxes[:,0]
        new_boxes[:,6]=new_boxes[:,4]*new_boxes[:,5]
        return new_boxes        

    



class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


