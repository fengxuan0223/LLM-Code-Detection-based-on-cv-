import functools
import torch
import torch.nn as nn
from .resnet import resnet50
from .base_model import BaseModel, init_weights
from .simple_mlp import SimpleMLP
from .codebert import CodeBERTClassifier
from .bilstm import BiLSTM

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        # if self.isTrain and not opt.continue_train:
        #     #self.model = resnet50(pretrained=True)
        #     self.model = SimpleMLP()
        #     self.model.fc = nn.Linear(2048, 1)
        #     torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
        #
        # if not self.isTrain or opt.continue_train:
        #     self.model = resnet50(num_classes=1)

        # -------------------------------
        # Select backbone
        # -------------------------------
        # if opt.arch == "codebert":
        #     self.model = CodeBERTClassifier()
        if opt.arch == "codebert":
            self.model = CodeBERTClassifier(
                freeze_encoder=opt.freeze_encoder
            )


        elif opt.arch == "simplemlp":
            self.model = SimpleMLP()

        elif opt.arch == "bilstm":
            self.model = BiLSTM()

        else:
            raise ValueError(f"Unknown architecture: {opt.arch}")

        # if self.isTrain:
        #     self.loss_fn = nn.BCEWithLogitsLoss()
        #     # initialize optimizers
        #     if opt.optim == 'adam':
        #         self.optimizer = torch.optim.Adam(self.model.parameters(),
        #                                           lr=opt.lr, betas=(opt.beta1, 0.999))
        #     elif opt.optim == 'sgd':
        #         self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                          lr=opt.lr, momentum=0.0, weight_decay=0)
        #     else:
        #         raise ValueError("optim should be [adam, sgd]")

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()

            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=opt.lr,
                    betas=(opt.beta1, 0.999)
                )
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=opt.lr,
                    momentum=0.0,
                    weight_decay=0
                )
            else:
                raise ValueError("optim should be [adam, sgd]")

        # if not self.isTrain or opt.continue_train:
        #     self.load_networks(opt.epoch)
        # self.model.to(opt.gpu_ids[0])
        #------------------------------------------------------------------------------------
        #以下是自己加的上边的是注释掉的
        # set device
        if len(opt.gpu_ids) > 0:
            self.device = torch.device(f'cuda:{opt.gpu_ids[0]}')
        else:
            self.device = torch.device('cpu')

        # load model if needed
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        self.model.to(self.device)

        #======================================================================================


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.8
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.8} to {param_group["lr"]}')
        print('*'*25)
        return True

    # def set_input(self, input):
    #     self.input = input[0].to(self.device)
    #     self.label = input[1].to(self.device).float()

    def set_input(self, input):
        if isinstance(input, dict):
            # ✅ 正确存储
            self.input_ids = input["input_ids"].to(self.device)
            self.attention_mask = input["attention_mask"].to(self.device)
            self.label = input["label"].to(self.device).float()

            # ✅ 为了兼容 SimpleMLP
            self.input = self.input_ids
        else:
            # 原始图像数据
            self.input = input[0].to(self.device)
            self.label = input[1].to(self.device).float()

    # def forward(self):
    #     self.output = self.model(self.input)

    def forward(self):
        if self.opt.arch == "codebert":
            self.output = self.model(self.input_ids, self.attention_mask)
        else:
            self.output = self.model(self.input)

    def get_loss(self):
        # return self.loss_fn(self.output.squeeze(1), self.label)
        return self.loss_fn(self.output, self.label)

    def optimize_parameters(self):
        self.forward()
        # self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.loss = self.loss_fn(self.output, self.label)

        self.optimizer.zero_grad()
        self.loss.backward()

        # ====== 梯度检查（只打印一次）======
        if self.total_steps == 1:
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    print(f"[Grad OK] {name}: {p.grad.norm().item():.6f}")
                    break
            else:
                print("[Grad ERROR] Model has NO gradient!")

        # ===================================

        self.optimizer.step()

