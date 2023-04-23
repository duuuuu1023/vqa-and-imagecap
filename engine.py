# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

import math
import sys
import tqdm
import torch.nn as nn
from models import utils


def train_one_epoch(model, criterion, data_loader,
                    optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()

    epoch_loss = 0.0
    total = len(data_loader)
    criterion1 = nn.CrossEntropyLoss()
    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks,question,question_mask,postion_id,target in data_loader:
            inputs = {}
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            target = target.to(device)
            question =question.to(device)
            question_mask = question_mask.to(device)
            postion_id = position.to(device)
            
            inputs['input_ids']=question
            inputs['pixel_values'] = images
            inputs['attention_mask'] = question_mask
            inputs['position_ids']= postion_id
            outputs,outputs_y = model(inputs,samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            loss1=torch.mean(self.model.encoder.logits_per_image())
            loss2= torch.mean(self.model.encoder.logits_per_text())
            loss3 = criterion1(outputs_y,target)
            loss = loss+loss1.detach()+loss2.detach()+loss3
            loss_value = loss.item()
            epoch_loss += loss_value
            
            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            pbar.update(1)

    return epoch_loss / total

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    validation_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks,question,question_mask,postion_id,target in data_loader:
            inputs = {}
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            target = target.to(device)
            question =question.to(device)
            question_mask = question_mask.to(device)
            postion_id = position.to(device)
            
            inputs['input_ids']=question
            inputs['pixel_values'] = images
            inputs['attention_mask'] = question_mask
            inputs['position_ids']= postion_id
            outputs,outputs_y = model(inputs,samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            loss1=torch.mean(self.model.encoder.logits_per_image())
            loss2= torch.mean(self.model.encoder.logits_per_text())
            loss3 = criterion1(outputs_y,target)
            loss = loss+loss1.detach()+loss2.detach()+loss3
            loss_value = loss.item()
            epoch_loss += loss_value
            validation_loss += loss.item()

            pbar.update(1)
        
    return validation_loss / total