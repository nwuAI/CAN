from Modules.backbone import Backbone
import torch
import argparse
import os
import numpy as np
from Modules.mri import LoadMRIData
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import DataParallel
import nibabel as nib

RESUME_PATH = '/opt/data/private/CAN/code/segmentation/save_model/pwml/checkpoint_final.pth.tar'
SAVE_DIR = '/opt/data/private//CAN/save_model/pwml/'
DATA_DIR = '/opt/data/private/CAN/code/Datasets/pwml/'
DATA_LIST = '/opt/data/private//CAN/data_list/'


class Solver():
    def __init__(self, args, num_class):
        self.args = args
        self.num_class = num_class

        test_data = LoadMRIData(args.data_dir, args.data_list, 'test', num_class, num_slices=args.num_slices,
                                Encode3D=args.encode3D)
        self.test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

        model = Backbone(num_class, args.num_slices)

        optimizer = torch.optim.SGD([{'params': model.encode3D1.parameters()},
                                     {'params': model.encode3D2.parameters()},
                                     {'params': model.encode3D3.parameters()},
                                     {'params': model.encode3D4.parameters()},
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
                                     {'params': model.conv6.parameters(), 'lr': args.lr},
                                     ],
                                    lr=1e-7, momentum=0.9, weight_decay=args.weight_decay)

        self.model, self.optimizer = model, optimizer

        # Using cuda
        if args.cuda:
            self.model = DataParallel(self.model).cuda()

        # Resuming checkpoint
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)

        if args.cuda:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}'".format(args.resume))

    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        volume_dice_score_list = []
        volume_iou_score_list = []
        batch_size = self.args.test_batch_size
        with torch.no_grad():
            for ind, sample_batched in enumerate(tbar):

                volume = sample_batched['image_3D'].type(torch.FloatTensor)
                skull = sample_batched['skull_3D'].type(torch.FloatTensor)
                labelmap = sample_batched['label_3D'].type(torch.LongTensor)
                volume = torch.squeeze(volume)
                skull= torch.squeeze(skull)

                labelmap = torch.squeeze(labelmap)
                sample_name = sample_batched['name']
                if self.args.cuda:
                    volume, skull,labelmap = volume.cuda(), skull.cuda(),labelmap.cuda()
                z_ax, x_ax, y_ax = np.shape(volume)
                
                volume_prediction = []
                for i in range(0, 256, batch_size):
                    
                    image_2D=volume[i,:,:][np.newaxis, :, :]
                    image_2D=image_2D.unsqueeze(0)

                    if i <= int(self.args.num_slices * 2):
                        image_stack01 = volume[0:int(i), :, :][None]
                        image_stack02 = volume[int(i+1):int(self.args.num_slices * 2 + 1), :, :][None]
                        skull_stack01 = skull[0:int(i), :, :][None]
                        skull_stack02 = skull[int(i+1):int(self.args.num_slices * 2 + 1), :, :][None]
                    elif i==int(self.args.num_slices * 2 + 1):
                        image_stack01 = volume[0:int(i-1), :, :][None]
                        image_stack02 = volume[int(i + 1):int(self.args.num_slices * 2 + 1), :, :][None]
                        skull_stack01 = skull[0:int(i-1), :, :][None]
                        skull_stack02 = skull[int(i + 1):int(self.args.num_slices * 2 + 1), :, :][None]
                    elif i >= 245 and i< 255:
                        image_stack01 = volume[z_ax - int(self.args.num_slices * 2 + 1):int(i), :, :][None]
                        image_stack02 = volume[int(i+1):, :, :][None]
                        skull_stack01 = skull[z_ax - int(self.args.num_slices * 2 + 1):int(i), :, :][None]
                        skull_stack02 = skull[int(i+1):, :, :][None]
                    elif i == 255:
                        image_stack01 = volume[z_ax - int(self.args.num_slices * 2 + 1):-1, :, :][None]
                        image_stack02 = volume[i+1:, :, :][None]
                        skull_stack01 = skull[z_ax - int(self.args.num_slices * 2 + 1):-1, :, :][None]
                        skull_stack02 = skull[i+1:, :, :][None]
                    else:
                        image_stack01 = volume[i - self.args.num_slices:i, :, :][None]
                        image_stack02 = volume[i+1 :i+self.args.num_slices+1, :, :][None]
                        skull_stack01 = skull[i - self.args.num_slices:i, :, :][None]
                        skull_stack02 = skull[i+1 :i+self.args.num_slices+1, :, :][None]

                    image_3D = torch.cat((image_stack01, image_stack02), dim=1)
                    skull_3D=torch.cat((skull_stack01,skull_stack02),dim=1)
                    
                    outputs = self.model(image_3D,skull_3D,image_2D)
                    pred = outputs
                    _, batch_output = torch.max(pred, dim=1)
                    volume_prediction.append(batch_output)
                volume_prediction = torch.cat(volume_prediction)
                volume_dice_score = score_perclass(volume_prediction, labelmap, self.num_class)
                volume_dice_score = volume_dice_score.cpu().numpy()
                volume_dice_score_list.append(volume_dice_score)
                tbar.set_description('Validate Dice Score: %.3f' % (np.mean(volume_dice_score)))

                visual = True
                if visual:
                    savedir_pred = os.path.join(self.args.save_dir, 'not-good1')
                    if not os.path.exists(savedir_pred):
                        os.makedirs(savedir_pred)
                    volume_prediction = volume_prediction.cpu().numpy().astype(np.uint8)
                    volume_prediction = np.transpose(volume_prediction, (1, 2, 0))
                    nib_pred = nib.Nifti1Image(volume_prediction,affine=np.eye(4))
                    nib.save(nib_pred, os.path.join(savedir_pred, sample_name[0] + '.nii.gz'))
            del volume_prediction

            dice_score_arr = np.asarray(volume_dice_score_list)

            ####################################save best model for dice###################
            if self.args.num_class is 139:
                label_list = np.array([4, 11, 23, 30, 31, 32, 35, 36, 37, 38, 39, 40,
                                       41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55,
                                       56, 57, 58, 59, 60, 61, 62, 63, 64, 69, 71, 72, 73,
                                       75, 76, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 112,
                                       113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                                       128, 129, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
                                       143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
                                       156, 157, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
                                       171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
                                       184, 185, 186, 187, 190, 191, 192, 193, 194, 195, 196, 197, 198,
                                       199, 200, 201, 202, 203, 204, 205, 206, 207])
                total_idx = np.arange(0, len(label_list))
                ignore = np.array([42, 43, 64, 69])

                valid_idx = [i + 1 for i in total_idx if label_list[i] not in ignore]
                valid_idx = [0] + valid_idx

                dice_socre_vali = dice_score_arr[:, valid_idx]
            else:
                dice_socre_vali = dice_score_arr
 
            avg_dice_score = np.mean(dice_socre_vali)
            std_dice_score = np.std(dice_socre_vali)
            print('Validation:')
            print("Mean of dice score : " + str(avg_dice_score))
            print("Std of dice score : " + str(std_dice_score))

def score_perclass(vol_output, ground_truth, num_classes):
    dice_perclass = torch.zeros(num_classes)
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        Pred = (vol_output == i).float()
        inter = torch.sum(torch.mul(GT, Pred))
        union = torch.sum(GT) + torch.sum(Pred) + 0.0001
        dice_perclass[i] = (2 * torch.div(inter, union))
    return dice_perclass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (dcriterionefault: 8)')
    parser.add_argument('--resume', type=str, default=RESUME_PATH,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default=SAVE_DIR, type=str, metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='directory to load data')
    parser.add_argument('--data-list', type=str, default=DATA_LIST,
                        help='directory to read data list')
    parser.add_argument('--encode3D', action='store_true', default=True,
                        help='directory to read data list')
    parser.add_argument('--se-loss', action='store_false', default=False,
                        help='apply se classification loss')
    parser.add_argument('--use-weights', action='store_false', default=False,
                        help='apply class weights for 2DCE loss')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b-test', '--test-batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-num-slices', '--num-slices', default=5, type=int,
                        metavar='N', help='slice thickness for spatial encoding')
    parser.add_argument('-num-class', '--num-class', default=28, type=int,
                        metavar='N', help='number of classes for segmentation')
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='check whether to use cuda')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(args)
    solver = Solver(args, args.num_class)
    print('Load model...')
    solver.validation(0)

if __name__ == '__main__':
    main()