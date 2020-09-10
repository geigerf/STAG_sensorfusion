import datetime
import re
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from shared.BaseModel import BaseModel
from shared.resnet_3x3 import resnet18


''' The final model '''

class SmallNet(nn.Module):
    '''
    This model extract features from tactile data.
    '''
    def __init__(self, inplanes=16, dropout=0):
        super(SmallNet, self).__init__()
        self.features = resnet18(inplanes=inplanes, dropout=dropout)
        self.features.fc  = nn.Threshold(-1e20, -1e20) # a pass-through layer for snapshot compatibility


    def forward(self, pressure):
        x = self.features(pressure)
        return x
    
    
class IMUnet(nn.Module):
    '''
    This model extract features from IMU data.
    '''
    def __init__(self, imu_in=6, imu_out=3, hidden_units=12, imu_dropout=0):
        super(IMUnet, self).__init__()
        self.in_layer = nn.Linear(imu_in, hidden_units)
        self.dropout = nn.Dropout(p=imu_dropout, inplace=False)
        self.out_layer = nn.Linear(hidden_units, imu_out)
        
        
    def forward(self, imu):
        x = self.in_layer(imu)
        x = self.dropout(x)
        x = self.out_layer(x)
        return x


class TouchNet(nn.Module):
    '''
    This model represents our classification network for 1..N input frames.
    '''
    def __init__(self, num_classes=27, inplanes=16, dropout=0,
                 imu_in=6, imu_out=3, imu_hidden=12, imu_dropout=0):
        super(TouchNet, self).__init__()
        self.Tactilenet = SmallNet(inplanes=inplanes, dropout=dropout)
        self.IMUnet = IMUnet(imu_in, imu_out, imu_hidden, imu_dropout)
        self.combination = nn.Conv2d(inplanes*2, inplanes*2,
                                      kernel_size=1, padding=0)
        self.classifier = nn.Linear(inplanes*2 + imu_out, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x0 = self.Tactilenet(x[0])
        # combine
        x0 = self.combination(x0)
        x0 = self.avgpool(x0)
        # This line is problematic for MX Cube AI
        # All it does is squeeze the last two dimensions from the tensor
        #x = x.view(x.size(0), -1)
        # Different way to do the same thing
        x0 = torch.nn.Flatten()(x0)
        x1 = self.IMUnet(x[1])
        x = torch.cat([x0, x1], dim=1)
        x = self.classifier(x)
        return x


class ClassificationModel(BaseModel):
    '''
    This class encapsulates the network and handles I/O.
    '''
    @property
    def name(self):
        return 'ClassificationModel'


    def initialize(self, numClasses, baseLr = 1e-3, inplanes=16, dropout=0,
                   cuda=True, imu_in=6, imu_out=3, imu_hidden=12,
                   imu_dropout=0):
        self.cuda = cuda
        BaseModel.initialize(self)

        print('Base LR = %e' % baseLr)
        self.baseLr = baseLr
        self.numClasses = numClasses

        self.model = TouchNet(num_classes=self.numClasses, inplanes=inplanes,
                              dropout=dropout, imu_in=imu_in, imu_out=imu_out,
                              imu_hidden=imu_hidden, imu_dropout=imu_dropout)
        # Count number of trainable parameters
        self.ntParams = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.nParams = sum(p.numel() for p in self.model.parameters())

        self.model = torch.nn.DataParallel(self.model)
        if self.cuda:
            self.model.cuda()
            cudnn.benchmark = True

        self.optimizer = torch.optim.Adam([
            {'params': self.model.module.parameters(),'lr_mult': 1.0},
            ], self.baseLr)

        self.optimizers = [self.optimizer]

        if self.cuda:
            self.criterion = nn.CrossEntropyLoss().cuda()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.epoch = 0
        self.error = 1e20 # last error
        self.bestPrec = 1e20 # best error

        self.dataProcessor = None


    def step(self, inputs, isTrain = True, params = {}):
        if isTrain:
            self.model.train()
            assert not inputs['objectId'] is None
        else:
            self.model.eval()

        if self.cuda:
            pressure_imu = inputs['pressure_imu']
            pressure_imu[0] = torch.autograd.Variable(inputs['pressure_imu'][0].cuda(),
                                               requires_grad = isTrain)
            pressure_imu[1] = torch.autograd.Variable(inputs['pressure_imu'][1].cuda(),
                                               requires_grad = isTrain)
            objectId = torch.autograd.Variable(inputs['objectId'].cuda(),
                                               requires_grad=False) if 'objectId' in inputs else None    
        else:
            pressure_imu = inputs['pressure_imu']
            pressure_imu[0] = torch.autograd.Variable(inputs['pressure_imu'][0],
                                               requires_grad = isTrain)
            pressure_imu[1] = torch.autograd.Variable(inputs['pressure_imu'][1],
                                               requires_grad = isTrain)
            objectId = torch.autograd.Variable(inputs['objectId'],
                                               requires_grad=False) if 'objectId' in inputs else None

        if isTrain:
            output = self.model(pressure_imu)
        else:
            with torch.no_grad():
                output = self.model(pressure_imu)

        _, pred = output.data.topk(1, 1, True, True)
        res = {
            'gt': None if objectId is None else objectId.data,
            'pred': pred,
            }

        if objectId is None:
            return res, {}

        loss = self.criterion(output, objectId.view(-1))

        (prec1, prec3) = self.accuracy(output, objectId,
                                       topk=(1, min(3, self.numClasses)))

        if isTrain:
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        losses = OrderedDict([
                            ('Loss', loss.data.item()),
                            ('Top1', prec1),
                            ('Top3', prec3),
                            ])

        return res, losses


    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.data.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.data.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res[0], res[1]


    def importState(self, save):
        params = save['state_dict']
        if hasattr(self.model, 'module'):
            try:
                self.model.load_state_dict(params, strict=True)
            except:
                self.model.module.load_state_dict(params, strict=True)
        else:
            params = self._clearState(params)
            self.model.load_state_dict(params, strict=True)

        self.epoch = save['epoch'] if 'epoch' in save else 0
        self.bestPrec = save['best_prec1'] if 'best_prec1' in save else 1e20
        self.error = save['error'] if 'error' in save else 1e20
        print('Imported checkpoint for epoch %05d with loss = %.3f...' % (self.epoch, self.bestPrec))


    def _clearState(self, params):
        res = dict()
        for k,v in params.items():
            kNew = re.sub('^module\.', '', k)
            res[kNew] = v

        return res


    def exportState(self):
        dt = datetime.datetime.now()
        state = self.model.state_dict()
        for k in state.keys():
            #state[k] = state[k].share_memory_()
            state[k] = state[k].cpu()
        return {
            'state_dict': state,
            'epoch': self.epoch,
            'error': self.error,
            'best_prec1': self.bestPrec,
            'datetime': dt.strftime("%Y-%m-%d %H:%M:%S")
            }


    def updateLearningRate(self, epoch):
        self.adjust_learning_rate_new(epoch, self.baseLr)


    def adjust_learning_rate_new(self, epoch, base_lr, period = 100): # train for 2x100 epochs
        gamma = 0.1 ** (1.0/period)
        lr_default = base_lr * (gamma ** (epoch))
        print('New lr_default = %f' % lr_default)

        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr_mult'] * lr_default
