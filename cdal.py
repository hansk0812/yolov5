from utils.loss import ComputeLoss
from utils.metrics import bbox_iou

import torch

class ComputeCDAL(ComputeLoss):

    def __call__(self, p, targets):  # predictions, targets - bounding boxes 

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
                """lbox += (1.0 - iou).mean()  # iou loss"""

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
                    """lcls += self.BCEcls(pcls, t)  # BCE""" 
            
            """obji = self.BCEobj(pi[..., 4], tobj) # BCE(predicted class confidence, bbox iou) 
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
            # autobalance across anchors
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]


        if self.autobalance:
        # autobalance across batches
            self.balance = [x / self.balance[self.ssi] for x in self.balance]"""
        
        # cls loss, box loss, obj loss
        return None, torch.zeros(3, device=self.device)

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        """
        print ([x.shape for x in p], targets.shape)
        # [torch.Size([2, 3, 80, 80, 85]), torch.Size([2, 3, 40, 40, 85]), torch.Size([2, 3, 20, 20, 85])] torch.Size([22, 6])
        # targets - [image_id, cls, bbox]
        """

        na, nt = self.na, targets.shape[0]  # number of anchors - 3, targets - 19x6
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain 
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt) 3x19 [0]*19,[1]*19,[2]*19
        
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices 3x19x7
        # [image_id, class, bbox, anchor_id]

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
            anchors, shape = self.anchors[i], p[i].shape # [2], [80x80x85], [40x40x85], [20x20x85]  
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain - [1,1,640,640,640,640,1] - [batch_img_id,cls,bbox,anchor_idx]

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7) [image_id, class, bbox, anchor_id]
            if nt: # gt bbox count > 0
                # Matches
                #print (t[...,4:6], anchors[:,None], t[...,4:6] / anchors[:,None])
                #exit()
                r = t[..., 4:6] / anchors[:, None]  # wh ratio t[...,[4,5]] == width, height
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

