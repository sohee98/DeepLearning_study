import models.AlexNet as AlexNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import pandas as pd

# 각 모델 불러오기
# AlexNet = models.AlexNet()
# VGG = models.VGG()

# class CustomDataset(torch.utils.data.Dataset): 
#     def __init__(self,df,path,option,augmentation=None):
#         self.df = df
#         self.option = option
#         self.augmentation = augmentation
#         self.path = path

#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx): 
#         file_path  = self.df.iloc[idx,0]
#         image=  Image.open(os.path.join(self.path,file_path)).convert('RGB')
#         image = np.array(image)
        
#         if self.augmentation is not None:
#             image = self.augmentation(image=image)['image']
        
#         if self.option =='train':
#             label = self.df.iloc[idx,1]
#             label = torch.tensor(label, dtype=torch.int64)
#             return image, label
        
#         return image




# pytorch device 정의하기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 논문 5. Details of learning 참고 parameter
batch_size = 128    # 1step당 처리할 데이터 수
momentum = 0.9
lr_decay = 0.0005
lr_init = 0.01
image_dim = 227  # pixels
num_classes = 1000  # 1000개의 class 지정
# device_ids = [0, 1, 2, 3]
num_epochs = 90

if __name__ == '__main__':
    seed = torch.initial_seed()  # seed value 설정

    ## model : AlexNet
    model = AlexNet(num_classes=num_classes).to(device) # 모델 생성
    # model = torch.nn.parallel.DataParallel(model, divice_ids=device_ids)  # 여러 GPU 사용하기

    ## dataset, data loader 설정
    train_dataset = datasets.MNIST('./data', train=True, download=True)
    test_dataset = datasets.MNIST('./data', train=False, download=True)
    # train_dataset = pd.read_csv('./data/MNIST/mnist_train.csv')
    # test_dataset = pd.read_csv('./data/MNIST/mnist_test.csv')
    print("train_data 수 : ", len(train_dataset))
    print("test_data 수 : ", len(test_dataset))


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)

    ## optimizer : SGD + Momentum
    optimizer = optim.SGD(
        params=model.parameters(),  # 모델의 파라미터들
        lr=lr_init,                 # learning rate : 한걸음의 보폭 (0.01)
        momentum=momentum,          # momentum : weight 업데이트할 때 이전의 방향도 반영함 (0.9)
        weight_decay=lr_decay       # lr 점점 감소 (0.0005)
    )

    total_batch = len(train_loader)
    print('총 배치의 수 : {}'.format(total_batch))

    ## scheduler : StepLR - step size마다 gamma비율로 lr을 감소시킴
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 30epoch마다 0.1비율로 lr를 감소시킴

    # training
    total_steps = 1
    for epoch in range(num_epochs):
        for imgs, classes in train_loader:
            imgs, classes = imgs.to(device), classes.to(device)

            output = AlexNet(imgs)
            loss = F.cross_entropy(output, classes)  # loss 계산

            optimizer.zero_grad()   # gradient 초기화
            loss.backward()  # backpropagation (compute gradient)
            optimizer.step()  # parameter update (SGD)
        lr_scheduler.step()
        total_steps += 1
    
    # testing
    AlexNet.eval()
    correct = 0
    for data, target in test_loader:
        output = AlexNet(data)
        prediction = output.data.max(1)[1]
        correct += prediction.eq(target.data).sum()

print('\nTest set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))



# one_image, label = train_dataset[0]
# plt.imshow(one_image.squeeze().numpy(), cmap='gray')