import os
import argparse

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset_file", help="Path to file with subset of coco dataset image names")
    args = ap.parse_args()

    with open(args.subset_file, 'r') as f:
        imgs = [x.strip() for x in f.readlines()]
    
    ultralytics_dataset = "/home/hans/code/yolo_ultralytics_dataset/datasets/coco/"

    with open(os.path.join(ultralytics_dataset, "train2017.txt"), 'r') as f:
        with open(os.path.join(ultralytics_dataset, args.subset_file.split('/')[-1]), 'w') as g:
            coco_imgs = [x.strip() for x in f.readlines()]
            for coco_img in coco_imgs:
                index_list = [idx for idx, x in enumerate(imgs) if x in coco_img] 
                if len(index_list) > 0:
                    imgs[index_list[0]] = coco_img
            
            for img in imgs:
                g.write(img + "\n")

