import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device   # cuda 지정

# 논문 5. Details of learning 참고 parameter
batch_size = 128
momentum = 0.9
lr_decay = 0.0005
lr_init = 0.01
image_dim = 227    # pixels
num_classes = 1000   # 1000개의 class 지정
device_ids = [0, 1, 2, 3]


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()  # 부모의 초기화 메서드 호출
        ##### CNN layers
        self.net = nn.Sequential(
            ## conv1 (227*227 -> 55*55) [(227-11)/4+1=55]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            # 입력채널수=3(RGB), 출력채널수=96, 필터크기=11*11, stride=4, 패딩=0
            nn.ReLU(inplace=True),  # non-saturating function
            # ReLU함수 사용
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            # NORM1 : 논문의 LRN 파라미터 그대로 지정
            # 정규화 k(가산계수)=2, n(size)=5, 알파=0.0001, 베타=0.75
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 풀링layer : 필터크기=3*3, stride=2 (55 -> 27)
            # Maxpooling : 정해진 크기 안에서 가장 큰 값만 뽑아냄

            ## conv2 (27 -> 27)
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            # 입력채널수=96, 출력채널수=256, 필터크기=5*5, stride=1, 패딩=2
            nn.ReLU(inplace=True),      # ReLU함수 사용
            # ReLU함수 사용
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 풀링layer : 필터크기=3*3, stride=2 (27 -> 13)

            ## conv3
            nn.Conv2d(256, 384, 3, padding=1),
            # 입력채널수=256, 출력채널수=384, 필터크기=3*3, stride=1, 패딩=1
            nn.ReLU(inplace=True),      # ReLU함수 사용

            ## conv4
            nn.Conv2d(384, 384, 3, padding=1),
            # 입력채널수=384, 출력채널수=384, 필터크기=3*3, stride=1, 패딩=1
            nn.ReLU(inplace=True),      # ReLU함수 사용

            ## conv5
            nn.Conv2d(384, 256, 3, padding=1),
            # 입력채널수=384, 출력채널수=256, 필터크기=3*3, stride=1, 패딩=1
            nn.ReLU(inplace=True),      # ReLU함수 사용
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 풀링layer : 필터크기=3*3, stride=2 (13 -> 6)

        )

        ##### FC layers
        self.classifier = nn.Sequential(
            # fc1
            nn.Dropout(p=0.5, inplace=True),    # Dropout 사용 : 0.5 확률
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            # 선형 변환 : input=256*6*6=9216, output=4096 (1차원 벡터로 변환)
            nn.ReLU(inplace=True),      # ReLU함수 사용

            # fc2
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(4096, 4096),  # input=4096, output=4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),   # input=4096, output=1000개
        )

        # bias, weight 초기화 (layer마다 초기화)
        def init_bias_weights(self):
            for layer in self.net:
                if isinstance(layer, nn.Conv2d):    # conv layer 이면
                    nn.init.normal_(layer.weight, mean=0, std=0.01)  # weight 초기화-정규분포에서 가져온 값으로
                    nn.init.constant_(layer.bias, 0)  # bias 초기화-0으로
            # conv 2, 4, 5는 bias 1로 초기화
            nn.init.constant_(self.net[4].bias, 1)  # 2번째 conv layer: bias=1
            nn.init.constant_(self.net[10].bias, 1) # 4번째 conv layer: bias=1
            nn.init.constant_(self.net[12].bias, 1) # 5번째 conv layer: bias=1

        # modeling
        def forward(self, x):
            x = self.net(x)  # conv layer 적용
            x = x.view(-1, 256 * 6 * 6)  # 256*6*6 으로 구조 변경
            return self.classifier(x)  # fc layer 적용 return
