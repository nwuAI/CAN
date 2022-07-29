import torch
from Modules.backbone import Backbone
from torch.nn import DataParallel
import os
import warnings
if __name__ == '__main__':
    model = Backbone(6,5)
#    torch.save(model.module.state_dict(),'/opt/data/private/chenyunbang/segmentation/save_model/MALC_coarse/checkpoint_pretrain.pth.tar')
    epoch = 0
    optimizer = torch.optim.SGD([{'params': model.encode3D1.parameters()},
                                 {'params': model.encode3D2.parameters()},
                                 {'params': model.encode3D3.parameters()},
                                 {'params': model.bottleneck3D.parameters()},
                                 {'params': model.encode3Ds1.parameters()},
                                 {'params': model.encode3Ds2.parameters()},
                                 {'params': model.encode3Ds3.parameters()},
                                 {'params': model.encode3Ds4.parameters()},
                                 {'params': model.bottlenecks3D.parameters()},
                                 {'params': model.encode2D1.parameters()},
                                 {'params': model.encode2D2.parameters()},
                                 {'params': model.encode2D3.parameters()},
                                 {'params': model.encode2D4.parameters()},
                                 {'params': model.bottleneck2D.parameters()},
                                 {'params': model.up_contact4.parameters()},
                                 {'params': model.conv4.parameters()},
                                 {'params': model.sa1.parameters()},
                                 {'params': model.CA4.parameters()},
                                 {'params': model.attentionblock3.parameters()},
                                 {'params': model.up_contact3.parameters()},
                                 {'params': model.conv3.parameters()},
                                 {'params': model.CA3.parameters()},
                                 {'params': model.attentionblock2.parameters()},
                                 {'params': model.up_contact2.parameters()},
                                 {'params': model.conv2.parameters()},
                                 {'params': model.CA2.parameters()},
                                 {'params': model.attentionblock1.parameters()},
                                 {'params': model.up_contact1.parameters()},
                                 {'params': model.conv1.parameters()},
                                 {'params': model.CA1.parameters()},
                                 {'params': model.dsv4.parameters()},
                                 {'params': model.dsv3.parameters()},
                                 {'params': model.dsv2.parameters()},
                                 {'params': model.dsv1.parameters()},
                                 {'params': model.scale_att.parameters()},
                                 {'params': model.conv6.parameters(), 'lr': 0.001},
                                 ],
                                lr=1e-7, momentum=0.9, weight_decay=1e-4)
    model = DataParallel(model).cuda()
    torch.save({
        'epoch':0,
        'state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_pred': 0.0,
    }, os.path.join('/opt/data/private/qianmi/CAN/save_model/pwml/', 'checkpoint_pretrain.pth.tar'))