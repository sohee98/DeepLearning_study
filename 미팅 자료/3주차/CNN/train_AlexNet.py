import models.AlexNet as AlexNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
# from torchvision.transforms import ToTensor
# from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pdb as pdb

# 각 모델 불러오기
# AlexNet = models.AlexNet()
# VGG = models.VGG()

# pytorch device 정의하기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creating a dinstinct transform class for the train, validation and test dataset
## 데이터 전처리 - tr.Compose()안의 순서대로 전처리 작업 수행
# 인풋데이터 227*227로 사이즈 조정 -> 랜덤으로 수평으로 뒤집는다 -> 텐서 데이터로 변환 -> 정규화
# RandomHorizontalFlip : 이미지를 랜덤으로 수평으로 뒤집는다
tranform_train = transforms.Compose([transforms.Resize((227,227)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
tranform_test = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])




# 논문 5. Details of learning 참고 parameter
batch_size = 128    # 1step당 처리할 데이터 수
momentum = 0.9
lr_decay = 0.0005
lr_init = 0.01
image_dim = 227  # pixels
num_classes = 1000  # 1000개의 class 지정
# device_ids = [0, 1, 2]
num_epochs = 90

if __name__ == '__main__':
    seed = torch.initial_seed()  # seed value 설정

    ## model : AlexNet
    model = AlexNet().to(device) # 모델 생성
    # model = torch.nn.parallel.DataParallel(model, divice_ids=device_ids)  # 여러 GPU 사용하기

    # ## dataset, data loader 설정
    # train_dataset = datasets.MNIST('./data', train=True, download=True, transform=tranform_train)
    # test_dataset = datasets.MNIST('./data', train=False, download=True, transform=tranform_test)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # preparing the train, validation and test dataset
    torch.manual_seed(43)
    # 라이브러리에서 제공하는 데이터셋
    train_ds = CIFAR10("data/", train=True, download=True, transform=tranform_train) #40,000 original images + transforms
    val_size = 10000 #there are 10,000 test images and since there are no transforms performed on the test, we keep the validation as 10,000
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size]) #Extracting the 10,000 validation images from the train set
    test_ds = CIFAR10("data/", train=False, download=True, transform=tranform_test) #10,000 images

    #passing the train, val and test datasets to the dataloader
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)    # dataset을 배치사이즈 형태로 만들어서 실제로 학습할때 이용할 수 있는 형태로 만든다.
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    print("train_data 수 : ", len(train_ds))
    print("validation_data 수 : ", len(val_ds))
    print("test_data 수 : ", len(test_ds))

    total_batch = len(train_dl) 
    print('총 배치의 수 : {}'.format(total_batch))

    ## loss
    load_model = True
    criterion = nn.CrossEntropyLoss()

    ## optimizer : SGD + Momentum
    optimizer = optim.SGD(
        params=model.parameters(),  # 모델의 파라미터들
        lr=lr_init,                 # learning rate : 한걸음의 보폭 (0.01)
        momentum=momentum,          # momentum : weight 업데이트할 때 이전의 방향도 반영함 (0.9)
        weight_decay=lr_decay       # lr 점점 감소 (0.0005)
    )


    ## scheduler : StepLR - step size마다 gamma비율로 lr을 감소시킴
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 30epoch마다 0.1비율로 lr를 감소시킴


    # training
    total_steps = 1
    for epoch in range(num_epochs):
        # for imgs, classes in train_loader:
        #     imgs, classes = imgs.to(device), classes.to(device)

        #     output = model(imgs)
        #     loss = F.cross_entropy(output, classes)  # loss 계산

        #     optimizer.zero_grad()   # gradient 초기화
        #     loss.backward()  # backpropagation (compute gradient)
        #     optimizer.step()  # parameter update (SGD)
        # lr_scheduler.step()
        # total_steps += 1
        loss_ep = 0
    
        for batch_idx, (data, targets) in enumerate(train_dl):
            data = data.to(device=device)
            targets = targets.to(device=device)
            ## Forward Pass
            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores,targets)
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
        print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_dl)}")

        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for batch_idx, (data,targets) in enumerate(val_dl):
                data = data.to(device=device)
                targets = targets.to(device=device)
                ## Forward Pass
                scores = model(data)
                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)
            print(
                f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
            )

    
    # testing
#     model.eval()
#     correct = 0
#     for data, target in test_loader:
#         output = model(data)
#         prediction = output.data.max(1)[1]
#         correct += prediction.eq(target.data).sum()

# print('\nTest set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))

