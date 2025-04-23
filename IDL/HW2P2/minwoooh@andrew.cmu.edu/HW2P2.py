import torch
from torchsummary import summary
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import os
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn import metrics as mt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torchvision.models as models
import torch.serialization
import wandb
import csv

torch.cuda.set_device(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

config = {
    'batch_size': 64,
    'lr': 0.001,
    'epochs': 20,
    'data_dir': "/home/work/jupyter/minwoo/CMU/Intro_deep_learning/HW2P2/content/data/11-785-f24-hw2p2-verification/cls_data",
    'data_ver_dir': "/home/work/jupyter/minwoo/CMU/Intro_deep_learning/HW2P2/content/data/11-785-f24-hw2p2-verification/ver_data",
    'checkpoint_dir': "/home/work/jupyter/minwoo/CMU/Intro_deep_learning/HW2P2/content/data/11-785-f24-hw2p2-verification/checkpoint"
}

# Dataset 클래스
class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_file, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.pairs = []
        if csv_file.endswith('.csv'):
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    else:
                        self.pairs.append(row)
        else:
            with open(csv_file, 'r') as f:
                for line in f.readlines():
                    self.pairs.append(line.strip().split(' '))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path1, img_path2, match = self.pairs[idx]
        img1 = Image.open(os.path.join(self.data_dir, img_path1))
        img2 = Image.open(os.path.join(self.data_dir, img_path2))
        return self.transform(img1), self.transform(img2), int(match)


class TestImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_file, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.pairs = []
        if csv_file.endswith('.csv'):
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    else:
                        self.pairs.append(row)
        else:
            with open(csv_file, 'r') as f:
                for line in f.readlines():
                    self.pairs.append(line.strip().split(' '))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path1, img_path2 = self.pairs[idx]
        img1 = Image.open(os.path.join(self.data_dir, img_path1))
        img2 = Image.open(os.path.join(self.data_dir, img_path2))
        return self.transform(img1), self.transform(img2)

# ArcFace 정의
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.9):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class Network(nn.Module):
    def __init__(self, num_classes=8631):
        super(Network, self).__init__()
        self.backbone = torchvision.models.resnet101(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # 마지막 FC 레이어 제거
        self.arc_face = ArcMarginProduct(2048, num_classes, s=64.0, m=0.9)  # 512를 2048로 수정

    def forward(self, x, labels=None):  # labels 인자의 기본값을 None으로 설정
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        if labels is not None:
            out = self.arc_face(feats, labels)
            return {"feats": feats, "out": out}
        else:
            return {"feats": feats}


model = Network().to(DEVICE)
summary(model, (3, 112, 112))

# Optimizer, Loss, Scheduler 설정
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

# 데이터 로딩
data_dir = config['data_dir']
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'dev')

# train transforms
train_transforms = transforms.Compose([
    transforms.Resize(112),  # Image resizing
    transforms.RandAugment(),  # RandAugment 적용
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Verification dataset transforms
val_transforms = transforms.Compose([
    transforms.CenterCrop(112),  # Center crop instead of resize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 데이터셋 및 데이터로더 설정
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

data_dir = config['data_ver_dir']

# 검증용 데이터셋과 데이터로더
pair_dataset = ImagePairDataset(data_dir, csv_file='/home/work/jupyter/minwoo/CMU/Intro_deep_learning/HW2P2/content/data/11-785-f24-hw2p2-verification/val_pairs.txt', transform=val_transforms)
pair_dataloader = torch.utils.data.DataLoader(pair_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, num_workers=4)

# 테스트용 데이터셋과 데이터로더
test_pair_dataset = TestImagePairDataset(data_dir, csv_file='/home/work/jupyter/minwoo/CMU/Intro_deep_learning/HW2P2/content/data/11-785-f24-hw2p2-verification/test_pairs.txt', transform=val_transforms)
test_pair_dataloader = torch.utils.data.DataLoader(test_pair_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, num_workers=4)

scaler = torch.cuda.amp.GradScaler()

# AverageMeter와 Accuracy 계산 함수
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

# Verification 메트릭 계산 함수
def get_ver_metrics(labels, scores, FPRs):
    fpr, tpr, _ = mt.roc_curve(labels, scores, pos_label=1)
    roc_curve = interp1d(fpr, tpr)
    EER = 100. * brentq(lambda x : 1. - x - roc_curve(x), 0., 1.)
    AUC = 100. * mt.auc(fpr, tpr)
    tnr = 1. - fpr
    pos_num = labels.count(1)
    neg_num = labels.count(0)
    ACC = 100. * max(tpr * pos_num + tnr * neg_num) / len(labels)
    TPRs = [('TPR@FPR={}'.format(FPR), 100. * roc_curve(float(FPR))) for FPR in FPRs] if isinstance(FPRs, list) else []
    return {'ACC': ACC, 'EER': EER, 'AUC': AUC, 'TPRs': TPRs}

import numpy as np

def rand_bbox(size, lam):
    """ 
    Generates random bounding box for CutMix. 
    Args:
        size (tuple): Shape of the input image (batch_size, channels, height, width)
        lam (float): Lambda value (proportion of the area to cut)
    
    Returns:
        tuple: Coordinates for the bounding box (x1, y1, x2, y2)
    """
    W = size[2]  # Width of the image
    H = size[3]  # Height of the image
    cut_rat = np.sqrt(1. - lam)  # Calculate the cut ratio
    cut_w = np.int(W * cut_rat)  # Cut width
    cut_h = np.int(H * cut_rat)  # Cut height

    # Uniformly sample the center of the bounding box
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Calculate the bounding box coordinates
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# Mixup 적용 함수
def mixup_data(x, y, alpha=0.8):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Cutmix 적용 함수
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(DEVICE)
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    return x, target_a, target_b, lam

# Verification 메트릭 계산 함수
def get_ver_metrics(labels, scores, FPRs):
    fpr, tpr, _ = mt.roc_curve(labels, scores, pos_label=1)
    roc_curve = interp1d(fpr, tpr)
    EER = 100. * brentq(lambda x : 1. - x - roc_curve(x), 0., 1.)
    AUC = 100. * mt.auc(fpr, tpr)
    tnr = 1. - fpr
    pos_num = labels.count(1)
    neg_num = labels.count(0)
    ACC = 100. * max(tpr * pos_num + tnr * neg_num) / len(labels)

    if isinstance(FPRs, list):
        TPRs = [('TPR@FPR={}'.format(FPR), 100. * roc_curve(float(FPR))) for FPR in FPRs]
    else:
        TPRs = []

    return {
        'ACC': ACC,
        'EER': EER,
        'AUC': AUC,
        'TPRs': TPRs,
    }

# 앙상블용 모델 저장 함수
def save_model_with_performance(model, optimizer, scheduler, metrics, epoch, path, performance):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metric': metrics,
        'epoch': epoch
    }, f"{path}/model_epoch_{epoch}.pth")
    performance.append((epoch, metrics['valid_ret_acc']))


# 앙상블을 위한 상위 3개 에포크 로드
def load_best_models_for_ensemble(performance, model, path):
    top_3_epochs = sorted(performance, key=lambda x: x[1], reverse=True)[:3]
    print(f"Selected best epochs for ensemble: {top_3_epochs}")
    
    models = []
    for epoch, _ in top_3_epochs:
        checkpoint = torch.load(f"{path}/model_epoch_{epoch}.pth", weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
    return models

# 앙상블 수행
def ensemble(models, dataloader, device):
    scores = []
    for model in models:
        model.eval()
        model.to(device)
        batch_scores = []
        for images1, images2 in dataloader:
            images = torch.cat([images1, images2], dim=0).to(device)
            with torch.no_grad():
                outputs = model(images, None)
                feats = F.normalize(outputs['feats'], dim=1)
                feats1, feats2 = feats.chunk(2)
                similarity = F.cosine_similarity(feats1, feats2)
                
                # similarity는 여러 값으로 이뤄진 시퀀스일 수 있으므로, 이를 단일 배열로 만들어야 함
                batch_scores.extend(similarity.cpu().numpy().tolist())
        
        # 각 모델의 결과를 일관된 크기의 리스트로 만든 후, 최종적으로 np.array로 변환
        scores.append(np.array(batch_scores))
    
    ensemble_scores = np.mean(np.array(scores), axis=0)
    return ensemble_scores


# 성능 기록 리스트 초기화
epoch_performance = []

# 학습 함수
def train_epoch(model, dataloader, optimizer, lr_scheduler, scaler, device, config):
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # Zero gradients

        # 이미지와 라벨을 GPU로 보냅니다
        images, labels = images.to(device), labels.to(device)

        # forward
        with torch.cuda.amp.autocast():  # Mixed Precision 적용
            outputs = model(images, labels)
            loss = criterion(outputs['out'], labels)

        # 역전파 및 최적화
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 메트릭 업데이트
        acc = accuracy(outputs['out'], labels)[0].item()  # 정확도 계산
        loss_m.update(loss.item())
        acc_m.update(acc)

        # Progress bar 업데이트
        batch_bar.set_postfix(acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg), loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg))
        batch_bar.update()

    batch_bar.close()

    # Scheduler 업데이트
    if lr_scheduler is not None:
        lr_scheduler.step()

    return acc_m.avg, loss_m.avg

# 검증 함수
@torch.no_grad()
def valid_epoch_cls(model, dataloader, device, config):
    model.eval()
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, desc='Val Cls.')

    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # forward
        outputs = model(images, labels)
        loss = criterion(outputs['out'], labels)

        # 메트릭 업데이트
        acc = accuracy(outputs['out'], labels)[0].item()
        loss_m.update(loss.item())
        acc_m.update(acc)

        batch_bar.set_postfix(acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg), loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg))
        batch_bar.update()

    batch_bar.close()
    return acc_m.avg, loss_m.avg

def valid_epoch_ver(model, pair_data_loader, device, config):
    model.eval()
    scores = []
    match_labels = []

    batch_bar = tqdm(total=len(pair_data_loader), dynamic_ncols=True, leave=False, desc='Val Veri.')

    for i, (images1, images2, labels) in enumerate(pair_data_loader):
        images = torch.cat([images1, images2], dim=0).to(device)

        # forward
        outputs = model(images, labels=None)
        feats = F.normalize(outputs['feats'], dim=1)
        feats1, feats2 = feats.chunk(2)
        similarity = F.cosine_similarity(feats1, feats2)

        # detach()를 사용하여 그래디언트 추적을 해제
        scores.extend(similarity.detach().cpu().numpy().tolist())
        match_labels.extend(labels.cpu().numpy().tolist())

        batch_bar.update()

    batch_bar.close()

    FPRs = ['1e-4', '5e-4', '1e-3', '5e-3', '5e-2']
    metric_dict = get_ver_metrics(match_labels, scores, FPRs)

    # EER 평가 지표 출력
    print(f"EER: {metric_dict['EER']:.4f}%")
    return metric_dict['ACC'], metric_dict['EER']

# 테스트 함수
@torch.no_grad()
def test_epoch_ver(model, pair_data_loader, device):
    model.eval()
    scores = []

    batch_bar = tqdm(total=len(pair_data_loader), dynamic_ncols=True, leave=False, desc='Test Veri.')

    for i, (images1, images2) in enumerate(pair_data_loader):
        images = torch.cat([images1, images2], dim=0).to(device)

        # forward
        outputs = model(images, labels=None)
        feats = F.normalize(outputs['feats'], dim=1)
        feats1, feats2 = feats.chunk(2)
        similarity = F.cosine_similarity(feats1, feats2)
        scores.extend(similarity.cpu().numpy().tolist())

        batch_bar.update()

    batch_bar.close()
    return scores

# 학습 및 검증 실행
for epoch in range(config['epochs']):
    print(f"\nEpoch {epoch+1}/{config['epochs']}")

    # 학습
    train_cls_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, DEVICE, config)
    print(f"Train Cls. Acc: {train_cls_acc:.4f}%, Train Loss: {train_loss:.4f}")

    # 검증 (Classification)
    valid_cls_acc, valid_loss = valid_epoch_cls(model, val_loader, DEVICE, config)
    print(f"Val Cls. Acc: {valid_cls_acc:.4f}%, Val Loss: {valid_loss:.4f}")

    # 검증 (Verification) - EER 포함
    valid_ver_acc, valid_ver_eer = valid_epoch_ver(model, pair_dataloader, DEVICE, config)
    print(f"Val Ver. Acc: {valid_ver_acc:.4f}%, Val Ver. EER: {valid_ver_eer:.4f}%")

    # 성능 기록 및 모델 저장
    metrics = {
        'train_cls_acc': train_cls_acc, 
        'valid_cls_acc': valid_cls_acc, 
        'valid_ret_acc': valid_ver_acc, 
        'valid_ret_eer': valid_ver_eer  # EER 포함
    }
    save_model_with_performance(model, optimizer, scheduler, metrics, epoch, config['checkpoint_dir'], epoch_performance)

# 앙상블 수행
best_models = load_best_models_for_ensemble(epoch_performance, model, config['checkpoint_dir'])
ensemble_scores = ensemble(best_models, test_pair_dataloader, DEVICE)

# 결과 저장
with open("submission.csv", "w") as f:
    f.write("ID,Label\n")
    for i, score in enumerate(ensemble_scores):
        f.write(f"{i},{score}\n")