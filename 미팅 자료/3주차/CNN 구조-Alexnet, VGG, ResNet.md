# CNN Network layer 모델

>  CNN 아키텍처
>
>  <img src="https://blog.kakaocdn.net/dn/YwtfW/btqCoRz4mjS/keKEbhbK5sm3XkgGviPcV1/img.png" alt="img" style="zoom:67%;" />

## 1. Alexnet

<img src="https://blog.kakaocdn.net/dn/PV4Qy/btqCFxGGoom/iTthgzuSyEXxTkC6E3af4k/img.png" alt="img" style="zoom:50%;" /><img src="https://blog.kakaocdn.net/dn/lceu6/btqCD9sNnpy/6numBkqx8OJTaBLAkPzaFk/img.png" alt="img" style="zoom: 67%;" />

![img](https://blog.kakaocdn.net/dn/J29tU/btq8avdCrY7/aIWPxEtLDWMEZidf4Hr9O1/img.png)

* 최초로 제안된 거대한 크기의 CNN 아키텍처
* [Input layer - Conv1 - MaxPool1 - Norm1 - Conv2 - MaxPool2 - Norm2 - Conv3 - Conv4 - Conv5 - Maxpool3 - FC1- FC2 - Output layer] 
* Convolution layer : 5개
* Pooling layer : 3개
* Local Response Normalization layer : 2개
* Fully-connected layer : 3개
* 뉴런 약 65만개
* 파라미터 약 6200만개
* 연산량 약 6억 3000만개

![img](https://blog.kakaocdn.net/dn/QI2jJ/btq1xmhigD0/ndO9JG4Yty5doDrAKylmh0/img.png)

* 입력 : 227 * 227 * 3 크기 이미지
* 첫번째 layer : 96개의 11*11 필터가 stride 4로 적용 => 출력 : `55*55*96` `(<=(227-11)/4+1=55)`
  * 파라미터 : `(11*11*3)*96 = 35K`
  * POOL 1 : 3*3필터 stride 2로 적용 => 출력 볼륨 : `(55-3)/2+1 = 27`
  * 

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device   # cuda 지정
```

```python
# 논문 5. Details of learning 참고 parameter
batch_size = 128
momentum = 0.9
lr_decay = 0.0005
lr_init = 0.01
image_dim = 227    # pixels
num_classes = 1000   # 1000개의 class 지정
device_ids = [0, 1, 2, 3]
```

```python
class AlexNet(nn.Module):
  def __init__(self, num_classes=1000):
    super().__init__()
    ##### CNN layers 
    self.net = nn.Sequential(
        ## conv1
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
        	# 입력채널수=3, 출력채널수=96, 필터크기=11*11, stride=4, 패딩=0
        nn.ReLU(inplace=True),  # non-saturating function
        	# ReLU함수 사용
        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  
        	# NORM1 : 논문의 LRN 파라미터 그대로 지정 
        nn.MaxPool2d(kernel_size=3, stride=2),
        	# 풀링layer : 필터크기=3*3, stride=2
        
        ## conv2
        nn.Conv2d(96, 256, kernel_size=5, padding=2), 
        	# 입력채널수=96, 출력채널수=256, 필터크기=5*5, stride=1, 패딩=2
        nn.ReLU(inplace=True),		# ReLU함수 사용
        	# ReLU함수 사용
        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        nn.MaxPool2d(kernel_size=3, stride=2),		
        	# 풀링layer : 필터크기=3*3, stride=2
        
        ## conv3
        nn.Conv2d(256, 384, 3, padding=1),
       		# 입력채널수=256, 출력채널수=384, 필터크기=3*3, stride=1, 패딩=1
        nn.ReLU(inplace=True),
        
        ## conv4
        nn.Conv2d(384, 384, 3, padding=1),
        	# 입력채널수=384, 출력채널수=384, 필터크기=3*3, stride=1, 패딩=1
        nn.ReLU(inplace=True),		# ReLU함수 사용
        
        ## conv5
        nn.Conv2d(384, 256, 3, padding=1),
        	# 입력채널수=384, 출력채널수=256, 필터크기=3*3, stride=1, 패딩=1
        nn.ReLU(inplace=True),		# ReLU함수 사용
        nn.MaxPool2d(kernel_size=3, stride=2),
        	# 풀링layer : 필터크기=3*3, stride=2

    )

    ##### FC layers
    self.classifier = nn.Sequential(
        # fc1
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
        nn.ReLU(inplace=True).
        # fc2
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )
    # bias, weight 초기화 
    def init_bias_weights(self):
      for layer in self.net:
        if isinstance(layer, nn.Conv2d):
          nn.init.normal_(layer.weight, mean=0, std=0.01)   # weight 초기화
          nn.init.constant_(layer.bias, 0)   # bias 초기화
      # conv 2, 4, 5는 bias 1로 초기화 
      nn.init.constant_(self.net[4].bias, 1)
      nn.init.constant_(self.net[10].bias, 1)
      nn.init.constant_(self.net[12].bias, 1)
    # modeling 
    def forward(self, x):
      x = self.net(x)   # conv
      x = x.view(-1, 256*6*6)   # keras의 reshape (텐서 크기 2d 변경)
      return self.classifier(x)   # fc   
```

```python
if __name__== '__main__':
  seed = torch.initial_seed()  # seed value 설정
  model = AlexNet(num_classes=num_classes).to(device)
  model = torch.nn.parallel.DataParallel(model, divice_ids=device_ids)  # 모델 설정
  print(model)

  # dataset, data loader 설정
  dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
      transforms.CenterCrop(IMAGE_DIM),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ]))

  dataloader = data.DataLoader(
      dataset,
      shuffle=True,
      pin_memory=True,
      num_workers=8,
      drop_last=True,
      batch_size=batch_size)

  # optimizer
  optimizer = optim.SGD(
      params = model.parameters(),
      lr = lr_init,
      momentum = momentum,
      weight_decay = lr_decay  # lr 점점 감소
  )

  lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)   # lr 점점 감소

  # training
  total_steps=1
  for epoch in range(num_epochs):
    lr_scheduler.step()

    for imgs, classes in dataloader:
      imgs, classes = imgs.to(device), classes.to(device)

      output = alexnet(imgs)
      loss = F.cross_entropy(output, classes)  # loss 계산

      optimizer.zero_grad()
      loss.backward()  # backpropa
      optimizer.step()  # parameter update
```

* MNIST

  ```python
  import torch
  from torchvision import datasets, transforms
  
  batch_size = 1
  train_loader = torch.utils.data.DataLoader(
      datasets.MNIST(
          root = "datasets/", # 현재 경로에 datasets/MNIST/ 를 생성 후 데이터를 저장한다.
          train = True, # train 용도의 data 셋을 저장한다.
          download = True,
          transform = transforms.Compose([
              transforms.ToTensor(), # tensor 타입으로 데이터 변경
              transforms.Normalize(mean = (0.5,), std = (0.5,)) # data를 normalize 하기 위한 mean과 std 입력
          ])
      ),
      batch_size=batch_size, 
      shuffle=True
  )
  ```

  ```python
  image, label = next(iter(train_loader))
  print(image.shape, label.shape)
  # torch.Size([1, 1, 28, 28]) torch.Size([1])
  ```

  





## 2. VGG

* 구조가 간단하지만 GoogleNet과 에러 0.1% 차이

* 작은 필터를 여러층으로 쌓는 ResNet의 모체

* Small Filter Size & Deep Networks

* 커널사이즈를 3*3으로 고정

* AlexNet과 차이점

  * AlexNet에서 11x11, 5x5와 같은 넓은 크기의 커널로 Convolutation연산을 적용하는 것보다 여러 개의 3x3 Convolution 연산을 수행하는 것이 더 뛰어난 Feature를 추출합니다.
  * AlexNet 대비 더 많은 채널수와 깊은 Layer 구성
  * 3x3 크기의 커널을 연속해서 Convolution 적용한 뒤에 Max Pooling 적용하여 Convolution Feature map Block을 생성
  * Block 내에는 동일한 커널 크기와 Channel 개수를 적용하여 동일한 크기의 feature map들을 생성
  * 이전 Block 내에 있는 Feature Map의 크기는 2배로 줄어들지만 채널은 2배로 늘어남

* 모델링

  * Convolutional Layer를 3*3 필터, Padding=1로 원본크기에 변화를 주지 않음
  * Max-pooling을 사용해 사이즈를 절반으로 줄여나가면서 특징들을 추출
  * 각각의 Conv Layer뒤에 활성화 함수로 ReLU를 사용
  * FC Layer로 4096, 4096, num_classes=1000으로 3개의 층
  * 중간중간 Dropout 0.5 주었다.
  * 최종결과 Softmax

* 최적화

  * Batch Size 256, momentum 0.9, 
  * Learning Rate 0.01- lr에 0.1을 곱했고 총 3번 곱했다

* 층수에 따라 16개 층은 VGG16, 19개 층은 VGG19라고 불림

* <img src="https://blog.kakaocdn.net/dn/7n8gM/btqCtYxJRrl/6nlJFbarX2txmzsKzLveQk/img.png" alt="img" style="zoom:50%;" />

* 6개의 구조(A, A-LRN, B, C, D, E)를 만들어 성능을 비교했다.

* 이들 중, D구조가 VGG16이고 E구조가 VGG19라고 보면 된다.

* 깊어질수록 성능이 좋아진다.

* ![img](https://blog.kakaocdn.net/dn/x7APM/btqCsGRJYKo/0PYNRnSEadZ8Qif7mk2iM1/img.png)

* 그림에서 확인할 수 있듯이, VGG는 합성곱 계층(검정색)과 풀링 계층(빨간색)으로 구성된다.

  다만, 합성곱 계층(검정색)과 완전연결 계층(파란색)을 합쳐서 모두 16층(VGG19의 경우 19층)으로 심화한 것이 특징이다.

  인풋으로는 224 x 224 x 3 이미지(224 x 224 RGB 이미지)를 입력받을 수 있다.

  인풋 값이 16개의 층을 지난 후 softmax 함수로 활성화 된 출력값들은 1000개의 뉴런으로 구성된다.

  이 말은 즉, 1000개의 클래스로 분류하는 목적으로 만들어진 네트워크라는 뜻이다.

* https://minjoos.tistory.com/6

 

![img](https://blog.kakaocdn.net/dn/cd5lZx/btqEE4RS8ML/RISHmDZILiKk9OZkFwbt0k/img.png)





```python
import torch.nn as nn
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(s.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
```

* pytorch에서 class 형태의 모델은 항상 nn.Module을 상속 받아야 하며, super(모델명, self).**init**()을 통해 nn.Module.**init**()을 실행시키는 코드가 필요하다
* forward()는 모델이 학습 데이터를 입력 받아서 forward prop을 진행시키는 함수
* VGG의 여러 모델간(VGG16, VGG19...)의 호환성을 위해, 가변적인 부분인 features은 입력으로 받고, 나머지 고정된 부분을 class 내에 설계한다.
* self.modules() -> 모델 클래스에서 정의된 layer들을 iterable로 차례로 반환
* isinstance() -> 차례로 layer을 입력하여, layer의 형태를 반환(nn.Conv2d, nn.BatchNorm2d ...)
* nn.init.kaiming_normal -> he initialization의 한 종류
* ![img](https://blog.kakaocdn.net/dn/bJFqC9/btqEG8LIhCZ/RM88i3LJQ4kX81HiDNHZz1/img.png)
* torch.nn.init.constant_(tensor, val) -> tensor을 val로 초기화
* torch.nn.init.normal_(tensor, mean=0.0, std=1.0) -> tensor을 mean, std의 normal distrubution으로 초기화

```python
def make_layers(cfg, batch_norm=False): 
    layers = [] 
    in_channels = 3 
    for v in cfg: 
        if v == 'M':# max pooling 
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)] 
    	else: 
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1) 
            if batch_norm: 
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)] 
            else: 
                layers += [conv2d, nn.ReLU(inplace=True)] 
            in_channels = v 
        
    return nn.Sequential(*layers)
```





https://github.com/CryptoSalamander/pytorch_paper_implementation/tree/master/vgg

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VGG(nn.Module):
    def __init__(self, config, num_classes=1000, cifar=False):
        super(VGG, self).__init__()
        self.features = make_layer(config)
        
        # ImageNet
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)  
        )
        if cifar:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, 10)  
            ) 
        
    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out,1)
        out = self.classifier(out)
        return out
    

    
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layer(config):
    layers = []
    in_planes = 3
    for value in config:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_planes, value, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_planes = value
    return nn.Sequential(*layers)

def VGG11(cifar=False):
    return VGG(config = cfg['A'], cifar = cifar)

def VGG13(cifar=False):
    return VGG(config = cfg['B'], cifar = cifar)

def VGG16(cifar=False):
    return VGG(config = cfg['D'], cifar = cifar)

def VGG19(cifar=False):
    return VGG(config = cfg['E'], cifar = cifar)
```







## 3. ResNet

https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
