from modeling import BertCrossattLayer,BertSelfattLayer,CrossEncoder
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from utils import NestedTensor, nested_tensor_from_tensor_list,get_rank
from backbone import build_backbone
from transformer import TransformersDEcoder
from transformers import DistilBertTokenizer, DistilBertModel,ViTModel,ViTFeatureExtractor
from torchvision import transforms
import numpy as np
import json
import os
import tqdm
from PIL import Image
import math
from sklearn.metrics import accuracy_score,f1_score
from log import init_loger
from datetime import timedelta
import time
import re
import nltk
from nltk.corpus import stopwords
#stop_words = stopwords.words('english')
from nltk.stem import PorterStemmer 
#st=PorterStemmer()
from textblob import TextBlob 
from textblob import Word
def data_clean(x):
    x = x.lower()                                # 所有字母转为小写
    #x = ' '.join([word for word in x.split(' ') if word not in stop_words])  # 删除停用词
    x = x.encode('ascii', 'ignore').decode()     # 删除 unicode 字符（乱码,例如：pel韈ula）
    x = re.sub("@\S+", " ", x)                   # 删除提及(例如：@zhangsan)
    x = re.sub("https*\S+", " ", x)              # 删除URL链接
    x = re.sub("#\S+", " ", x)                   # 删除标签（例如：#Amazing）
    x = re.sub("\'\w+", '', x)                   # 删除记号和下一个字符（例如：he's）
    x = re.sub(r'[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！\[\\\]^_`{|}~]+', ' ', x)   # 删除特殊字符 
    x = re.sub(r'\w*\d+\w*', '', x)              # 删除数字
    x = re.sub('\s{2,}', " ", x)                 # 删除2个及以上的空格
    x = x.strip()                                # 删除两端无用空格
    #x = str(TextBlob(x).correct())               # 拼写校对  ->  速度慢
    #x = " ".join([st.stem(word) for word in x.split()])            # 词干提取（例如：dysfunctional  -> dysfunct）
    #x = " ".join([Word(word).lemmatize() for word in x.split()])   # 词性还原（例如：dysfunct -> dysfunctional）
    #  x = re.sub(u'[\u4e00-\u9fa5]', ' ', x)  # 删除英文中的中文字符
    x = x.split()     # 分词

    return x
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
class Caption(nn.Module):
    def __init__(self,corss_attention ,transformer,vit_transformer,decoder,hidden_dim=768,num_cls=3129,vocab_size=12609):
        super().__init__()
        self.cross_attention = corss_attention
        self.transformer = transformer
        self.decoder = decoder
        self.vit_transformer = vit_transformer
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)
        self.classfier = Clssfier(hidden_dim,num_cls)
        self.num_cls = num_cls
    

    def forward(self, images_feature,inputs_text,tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        out_vit = self.vit_transformer(**images_feature).last_hidden_state
        out_txt = self.transformer(**inputs_text).last_hidden_state
        out_cross_t,out_cross_v = self.cross_attention(out_txt,out_vit)
        output_a = self.classfier(out_cross_t[:,0,:])
        #print(tgt.dtype)
        #tgt = self.transformer.embeddings(tgt.int())
        output_decoder = self.decoder(tgt,out_cross_t[:,1:,:],tgt_mask,src_mask, tgt_padding_mask, src_padding_mask)
        output_c = self.mlp(output_decoder)
        return output_a,output_c
        #

class Clssfier(nn.Module):
    def __init__(self,input_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(input_dim,num_class)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out = self.sigmoid(self.linear(x))
        return out
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        #print('*****')
        #print(x.shape)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        #print(x.shape)
        return x


def build_model(config):
    transformer = DistilBertModel.from_pretrained("distilbert-base-uncased")
    vit_transformer = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    decoder = TransformersDEcoder(config)#待调试
    criterion = torch.nn.CrossEntropyLoss()
    cross_encoder = CrossEncoder(config)
    model = Caption(cross_encoder,transformer,vit_transformer,decoder)
    return model, criterion

class CocoCaption(Dataset):
    def __init__(self,config,mode='training'):
        self.root = config.root
        self.annoqa = json.load(open(config.ann))
        if mode == 'validation':
            self.annot = self.annoqa[: config.limit]
        if mode == 'training':
            self.annot = self.annoqa[:]
        self.mode = mode
        self.max_length = config.max_length + 1
        self.ans2label = json.load(open("/root/autodl-tmp/catr-master/annotations/trainval_ans2label.json"))
        self.label2ans = json.load(open("/root/autodl-tmp/catr-master/annotations/trainval_label2ans.json"))
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower=True)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.read_word2id(config.word_path)
        assert len(self.ans2label) == len(self.label2ans)
    def _process(self, image_id):
        val = str(image_id)
        return val + '.jpg'
    def read_word2id(self,filepath):
        self.word_id={}
        with open(filepath,'r') as f:
            txts = f.readlines()
            
            for txt in txts:
                #print(txt.split('\n'))
                data = txt.split('\n')[0].split(' ')
                print(data)
                self.word_id[data[0]]=int(data[1])
    def token_process(self,caption,max_length):
        caption = data_clean(caption)
        self.deoder_length = self.max_length+1
        #caption = caption[:-2].split(' ')
        cap_len = len(caption)
        txts = []
        #print(self.word_id.keys)
        word_key = list(self.word_id.keys())
        for word in caption:
            if word in word_key:
                id = int(self.word_id[word])
                txts.append(id)
            else:
                txts.append(0)
        if cap_len>self.deoder_length-2:
            txt = txts[:self.deoder_length-2]
            txt.append(3)
        else :
            txts.append(3)
            pad = [1]*(self.deoder_length-2-cap_len)
            txt = txts+pad
        txt.insert(0,2)
        pad=[1]*self.deoder_length
        for i in range(self.deoder_length):
            if txt[i]==1:
                pad[i]=0
        caption_encoded = {'input_ids':txt,
                         'attention_mask':pad}
        return caption_encoded
        
    @property
    def num_answers(self):
        return len(self.ans2label)
    def __len__(self):
        return len(self.annot)
    def __getitem__(self, idx):
        annqa = self.annot[idx]
        image_id = self._process(annqa['img_id'])
        label = annqa['label']
        sent = annqa['sent']
        caption = annqa['caption']
        image = Image.open(os.path.join(self.root, image_id))
        if image.mode != 'RGB':
            image=image.convert('RGB')
        try:
            image_feature = self.feature_extractor(image,'pt')['pixel_values'][0]
        except ValueError:
            print(image_id)
        caption_encoded = self.token_process(
            caption, max_length=self.max_length)
        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
            1 - np.array(caption_encoded['attention_mask'])).astype(bool)
        sent_encoded = self.tokenizer.encode_plus(
            sent, max_length=self.max_length, padding='max_length', return_attention_mask=True, return_token_type_ids=False, truncation=True)
        sentence = np.array(sent_encoded['input_ids'])
        sentence_mask = (
            1 - np.array(sent_encoded['attention_mask'])).astype(bool)
        target = torch.zeros(self.num_answers)#onehot 000000000
        for ans, score in label.items():
                target[self.ans2label[ans]] = score
        sentence = torch.from_numpy(sentence)
        sentence_mask = torch.from_numpy(sentence_mask)
        captions = torch.from_numpy(caption)
        cap_mask = torch.from_numpy(cap_mask)
        
        return image_feature, sentence,sentence_mask,captions, cap_mask,target
def generate_square_subsequent_mask(sz,config):
    mask = (torch.triu(torch.ones((sz, sz), device=config.device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt,config):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len,config)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=config.device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
def train_one_epoch(model, criterion, data_loader,
                    optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()

    epoch_loss = 0.0
    total = len(data_loader)
    criterion1 = nn.BCELoss()
    acc_c=[]
    acc_a =[]
    f1_c = []
    f1_a = []
    answer_loss = 0.0
    with tqdm.tqdm(total=total) as pbar:
        for images, sentence,sentence_mask,caps, cap_masks,target in data_loader:
            #print(images.shape)
            inputs = {}
            inputs_img={}
            images = images.to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            target = target.to(device)
            question =sentence.to(device)
            question_mask = sentence_mask.to(device)
            #print(question_mask.shape)
            inputs['input_ids']=question
            inputs_img['pixel_values'] = images
            inputs['attention_mask'] = question_mask
            #tgt_mask = subsequent_mask(caps.shape[-1])
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask  =create_mask(inputs['input_ids'][:,1:],caps[:,1:],config)
            #print(src_mask.shape)
            #print(tgt_mask.shape)
            #print(src_padding_mask.shape)
            #print(tgt_padding_mask.shape)
            #print(tgt_mask)
            outputs,outputs_y = model(inputs_img,inputs,caps[:, :-1],  None,tgt_mask, src_padding_mask, tgt_padding_mask)
            #print(sentence.shape)
            #print(outputs.shape)
            #print(caps.shape)
            #print(outputs_y.permute(0, 2, 1).shape)
            #print(caps[:, 1:])
            loss = criterion(outputs_y.permute(0, 2, 1), caps[:, 1:])
            #loss3 = criterion1(outputs,target)
            loss = loss#+config.weight_an*loss3
            loss_value = loss.item()
            epoch_loss += loss_value
            y_c = torch.argmax(outputs_y,dim=-1)
            #print(y_c.shape)
            #print(caps[:, :-1].shape)
            #print(target.shape)
            #print(outputs.shape)
            acc_cc = accuracy_score(cap_masks[:, :-1].cpu().reshape(-1,1),y_c.detach().cpu().reshape(-1,1))#,f1_score
            #acc_aa = accuracy_score(target.cpu(),outputs.detach().cpu())
            acc_c.append(acc_cc)
            #acc_a.append(acc_aa)
            #acc_f1_a = f1_score(target.cpu(),outputs.detach().cpu())
            acc_f1_c = f1_score(cap_masks[:, :-1].cpu().reshape(-1,1),y_c.detach().cpu().reshape(-1,1),average='micro')
            f1_c.append(acc_f1_c)
            #f1_a.append(acc_f1_c)
            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            #answer_loss+= loss3.detach().cpu()
            pbar.update(1)

    return epoch_loss/total,np.mean(acc_c),np.mean(f1_c)#,answer_loss/total#np.mean(acc_aa),np.mean(f1_c),np.mean(f1_a)

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()
    criterion1 = nn.BCELoss()
    validation_loss = 0.0
    total = len(data_loader)
    acc_c=[]
    acc_a =[]
    f1_c = []
    f1_a = []
    answer_loss = 0.0
    with tqdm.tqdm(total=total) as pbar:
        for images, sentence,sentence_mask,caps, cap_masks,target in data_loader:
            inputs = {}
            inputs_img={}
            images = images.to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            target = target.to(device)
            question =sentence.to(device)
            question_mask = sentence_mask.to(device)     
            inputs['input_ids']=question
            inputs_img['pixel_values'] = images
            inputs['attention_mask'] = question_mask
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask  =create_mask(inputs['input_ids'][:,1:],caps[:,1:],config)
            outputs,outputs_y = model(inputs_img,inputs, caps[:, :-1],None, tgt_mask, src_padding_mask, tgt_padding_mask)
            loss = criterion(outputs_y.permute(0, 2, 1), caps[:, 1:])
            #loss1=torch.mean(self.model.encoder.logits_per_image())
            #loss2= torch.mean(self.model.encoder.logits_per_text())
            #loss3 = criterion1(outputs,target)
            loss = loss#+config.weight_an*loss3
            loss_value = loss.item()
            validation_loss += loss.item()
            y_c = torch.argmax(outputs_y,dim=-1)
            acc_cc = accuracy_score(caps[:, :-1].cpu().reshape(-1,1),y_c.detach().cpu().reshape(-1,1))#,f1_score
            #acc_aa = accuracy_score(target.cpu(),outputs.detach().cpu())
            acc_c.append(acc_cc)
            #acc_a.append(acc_aa)
            #acc_f1_a = f1_score(target.cpu(),outputs.detach().cpu())
            acc_f1_c = f1_score(caps[:, :-1].cpu().reshape(-1,1),y_c.detach().cpu().reshape(-1,1),average='micro')
            f1_c.append(acc_f1_c)
            #f1_a.append(acc_f1_c)
            pbar.update(1)
            #answer_loss+=loss3.detach().cpu()
    return validation_loss/total,np.mean(acc_c),np.mean(f1_c)#,answer_loss/total#np.mean(acc_aa),np.mean(f1_c),np.mean(f1_a)

def train(config,logger):
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')
    start_time = time.time()
    seed = config.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    best_acc = 0.0
    best_loss = 100.0
    model, criterion = build_model(config)
    
    model.to(device)
    for para in list(model.transformer.parameters()):
        para.requires_grad=False
    for para in list(model.vit_transformer.parameters()):
        para.requires_grad=False
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)
    dataset_train = CocoCaption(config)
    dataset_val = CocoCaption(config,mode='validation')
    print(f"Train: {len(dataset_train)}")
    print(f"Valid: {len(dataset_val)}")
    
    logger.info("Loading data...")
    data_loader_train = DataLoader(
        dataset_train, config.batch_size,shuffle = True,drop_last=False, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 drop_last=False, num_workers=config.num_workers)
    time_dif = get_time_dif(start_time)
    logger.info(f"Time usage:{time_dif}")
    if os.path.exists(config.checkpoint):
        print("Loading Checkpoint...")
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1
    print("Start Training..")
    for epoch in range(config.start_epoch, config.epochs):
        print(f"Epoch: {epoch}")
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.epochs))
        epoch_loss,acc_c,f1_c = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
        lr_scheduler.step()
        print(f"Training Loss: {epoch_loss},caption accuracy: {acc_c},caption f1_score: {f1_c}")
        validation_loss,vacc_c,vf1_c = evaluate(model, criterion, data_loader_val, device)
        print(f"Validation Loss: {validation_loss},caption accuracy: {vacc_c},caption f1_score: {vf1_c}")
        msg = 'Train Loss: {0:>5.6%}, Train caption Acc: {1:>6.4%},  Train caption F1: {2:>6.4%},  Val Loss: {3:>5.6%}, Val caption Acc: {4:>6.4%},  Val caption F1: {5:>6.4%}'
        logger.info(msg.format(epoch_loss, acc_c,f1_c,validation_loss,vacc_c,vf1_c))
        if vacc_c>best_acc:# and vanswer_loss<best_loss:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch+1,
            }, config.checkpoint+str(epoch+1)+'.ckpt')
            best_acc = vacc_c
            #best_loss = vanswer_loss
        print()
class Config:
    model_name = 'cv4'
    d_model = 768
    hidden_size=768
    num_attention_heads=8
    hidden_dropout_prob = 0.2
    intermediate_size =768
    attention_probs_dropout_prob = 0.2
    hidden_act='relu'
    cross_att_layer_num = 3
    d_model = 768
    num_cls = 3129
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = '/root/autodl-tmp/data/train2014'
    ann ='/root/autodl-tmp/catr-master/2014100000_train.json'
    ann_val ='/root/autodl-tmp/catr-master/2014100000_val.json'
    max_length=30
    limit=5000
    lr_drop = 10
    checkpoint = 'saved_models/model'
    seed = 123
    lr = 0.0001
    weight_decay = 0.0002
    batch_size=256
    num_workers=14
    start_epoch = 0 
    epochs=500
    clip_max_norm=20
    weight_an = 50
    hidden_dim=768
    layer_norm_eps = 0.00001
    pad_token_id = 1
    vocab_size=12609
    max_position_embeddings = 128
    word_path = 'word2id.txt'
    dropout = 0.2
if __name__ == '__main__':
    config = Config
    logger = init_loger(config)
    train(config,logger)