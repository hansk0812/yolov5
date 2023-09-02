# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        
        try:
            h = model.hyp  # hyperparameters
        except AttributeError:
            h = {"cls_pw": 1, "obj_pw": 1, "fl_gamma": 0, "anchor_t": 4.0}  
            # class pred weight, obj pred weight, focal loss gamma, anchor target max, box loss gain, obj loss gain, cls loss gain 

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        try:
            m = de_parallel(model).model[-1]  # Detect() module
        except TypeError:
            m = de_parallel(model).model.model[-1]

        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # self-annotated ; #anchors={3,5} ; P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors 3
        self.nc = m.nc  # number of classes 80+1
        self.nl = m.nl  # number of layers 3
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets - bounding boxes 

        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss 
        lobj = torch.zeros(1, device=self.device)  # object loss

        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # grid boxes based on anchor offsets
        # indices - [image_id, anchor_id, grid_id (y,x)] * 3

        """
        print ([x.shape for x in p], targets.shape)
        # [torch.Size([2, 3, 80, 80, 85]), torch.Size([2, 3, 40, 40, 85]), torch.Size([2, 3, 20, 20, 85])] torch.Size([22, 6])
        # targets [{0,1}, cls, xmax, ymax, xmin, ymin]

        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        print ([x.shape for x in tcls])
        # [torch.Size([78]), torch.Size([123]), torch.Size([93])]
        
        print ([x.shape for x in tbox])
        # [torch.Size([78, 4]), torch.Size([123, 4]), torch.Size([93, 4])]
        
        print ([[y.shape for y in x] for x in indices])
        # [[torch.Size([78]), torch.Size([78]), torch.Size([78]), torch.Size([78])], 
        # [torch.Size([123]), torch.Size([123]), torch.Size([123]), torch.Size([123])], 
        # [torch.Size([93]), torch.Size([93]), torch.Size([93]), torch.Size([93])]]
        
        print ([x.shape for x in anchors])
        # [torch.Size([78, 2]), torch.Size([123, 2]), torch.Size([93, 2])]
        """
        
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
        # [torch.Size([2, 3, 80, 80, 85]), torch.Size([2, 3, 40, 40, 85]), torch.Size([2, 3, 20, 20, 85])]

            b, a, gj, gi = indices[i]  # image - [0, batch_size-1], anchor - [0,2], 
            # gridy - [0-80], [0-40], [0-20], gridx - [0-80], [0-40], [0-20]
            
            """
            print ('image', b, 'anchor', a, 'gridy', gj, 'gridx', gi)
            exit()
            """

            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj grids without class label pi[4]

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
                # nboxes*2, nboxes*2, nboxes*1  constant, nboxes*80

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box nboxes * 4
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target) grids nboxes*4,[nboxes * 4]
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou: # sort by iou for all boxes in image
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                # objectness IOU lower bound and weighting factor
                    iou = (1.0 - self.gr) + self.gr * iou
                
                # iou.shape: [129], [231], [123], gj, gi - same shapes as iou; ranges: 80, 40, 20  
                tobj[b, a, gj, gi] = iou  # iou per grid ratio
                
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # BCE(linear interpolation weights)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # cn = alpha; targets
                    t[range(n), tcls[i]] = self.cp # cp = 1 - alpha
                    lcls += self.BCEcls(pcls, t)  # BCE 

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj) # BCE(predicted class confidence, bbox iou) 
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
            # autobalance across anchors
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
        # autobalance across batches
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        """
        print ([x.shape for x in p], targets.shape)
        # [torch.Size([2, 3, 80, 80, 85]), torch.Size([2, 3, 40, 40, 85]), torch.Size([2, 3, 20, 20, 85])] torch.Size([22, 6])
        """

        na, nt = self.na, targets.shape[0]  # number of anchors - 3, targets - 19x6
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain 
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt) 3x19 [0]*19,[1]*19,[2]*19
        
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices 3x19x7

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets
        
        for i in range(self.nl): # 3
            anchors, shape = self.anchors[i], p[i].shape # [2], [2x3x80x80x85], [2x3x40x40x85], [2x3x20x20x85]  
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain - [1,1,80,80,80,80,1] - [batch_img_id,cls,bbox,anchor_idx]
            
            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            
            # anchors = [number of grids / anchor width, number of grids / anchor height] 
            # t  = [number of grids / box width, number of grids / box height] 
            
            if nt:
                # Matches
                
                # (3xnx2) / (3x1x2) = (3xnx2) 
                r = t[..., 4:6] / anchors[:, None]  # wh ratio t[...,[4,5]] == width, height
                
                # anchor_t = (640/80) -> 8 // 2 = 4
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare (3 x nboxes x 2) - max(3 x nboxes) < (anchor_t=4)
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))

                t = t[j]  # filter
                
                # 7 to 5 outputs
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image_id, anchor_id, grid_id
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box - bbox to grid # [nboxes * 4] * 3
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        
        return tcls, tbox, indices, anch # grid boxes
