import numpy as np
import torch
import torch.distributed as dist
import os.path as osp
import matplotlib.pyplot as plt
import cv2
import json
from tqdm import tqdm
import os
import sys
from operator import itemgetter

sys.path.append('./GroundingDINOGroundingDINO/groundingdino')
from groundingdino.util.inference import Model as GroundingDINOModel
from groundingdino.util import box_ops

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def tag_masks(masks):
    H,W = masks[0]['segmentation'].shape
    ret_mask = np.zeros((H,W),dtype=np.int64)
    for idx,m in enumerate(masks):
        ret_mask[m['segmentation']]=idx+1
    return ret_mask

def rle2mask(rle_dict):
    height, width = rle_dict["size"]
    mask = np.zeros(height * width, dtype=np.uint8)
    rle_array = np.array(rle_dict["counts"])
    starts = rle_array[0::2]
    lengths = rle_array[1::2]
    for start, length in zip(starts, lengths):
        mask[start-1:start-1 + length] = 1
    mask = mask.reshape((height, width), order='F')
    return mask

def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if len(runs) % 2 != 0:
        runs = np.append(runs, len(pixels))
    runs[1::2] -= runs[::2]
    seg=[]
    for x in runs:
        seg.append(int(x))
    size=[]
    for x in img.shape:
         size.append(int(x))
    result=dict()
    result['counts']=seg
    result['size']=size
    return result

def save_masks(save_path,masks):
    out_list = []
    for idx,m in enumerate(masks):
        rle_result=mask2rle(m['segmentation'])
        out_list.append(rle_result)
    with open(save_path,'w') as f:
        json.dump(out_list,f)

def load_video_frames(data_path, split):
    video_frames = []
    with open(data_path) as f:
        data = json.load(f)
        videos = data["videos"]
        for video, video_data in videos.items():
            expressions = video_data["expressions"]
            frames = video_data["frames"]

            obj_id_to_first_exp = {}
            
            sorted_exp_keys = sorted(expressions.keys(), key=lambda x: int(x))
            
            for exp_key in sorted_exp_keys:
                exp_info = expressions[exp_key]
                obj_id = exp_info["obj_id"]
                exp_text = exp_info["exp"]
                
                if obj_id not in obj_id_to_first_exp:
                    obj_id_to_first_exp[obj_id] = exp_text

            for frame in frames:
                for obj_id, target_exp in obj_id_to_first_exp.items():
                    frame_data = {
                        "video": video,
                        "frame": frame,
                        "expression": target_exp,
                        "obj_id": obj_id,       
                        "split": split
                    }
                    video_frames.append(frame_data)
    return video_frames


valid_frames = load_video_frames("./datasets/ref-davis/meta_expressions/valid/meta_expressions.json", "valid")

grounding_dino_config = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
grounding_dino_weights = "./GroundingDINOGroundingDINO/groundingdino_swint_ogc.pth"
grounding_dino_model = GroundingDINOModel(
    model_config_path=grounding_dino_config,
    model_checkpoint_path=grounding_dino_weights,
    device="cuda"
)

sam_checkpoint = "./segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"

output_dir = './outputs/sam_db'
if not osp.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

dist.init_process_group(backend="nccl", init_method="env://")
world_size = dist.get_world_size()
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.cuda()
predictor = SamPredictor(sam)


for idx, data in tqdm(enumerate(valid_frames), total=len(valid_frames)):
    if idx % world_size != local_rank:
        continue
        
    video = data['video']
    frame_id = data['frame']
    split = data['split']
    expression = data['expression'] 
    obj_id = data['obj_id']         
    
    image_path = osp.join(f"datasets/ref-davis/{split}/JPEGImages/{video}", f"{frame_id}.jpg")

    if not osp.exists(image_path):
        continue

    save_path = osp.join(output_dir, f"{video}_{frame_id}_obj{obj_id}.pt")
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        continue
    
    detections = grounding_dino_model.predict_with_caption(
        image=image,
        caption=expression, 
        box_threshold=0.25,
        text_threshold=0.25
    )
    
    predictor.set_image(image)
    masks_list = []

    detection_result, matched_classes = detections
    
    if detection_result.xyxy.shape[0] > 0:
        boxes = detection_result.xyxy
        
        boxes_torch = torch.from_numpy(boxes).to(predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_torch, image.shape[:2])
        
        masks, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        masks = masks.squeeze(1).cpu().numpy()
        
        for i in range(masks.shape[0]):
            mask = masks[i]
            
            if np.sum(mask) == 0:
                continue

            where = np.where(mask)
            y_min, x_min = np.min(where[0]), np.min(where[1])
            y_max, x_max = np.max(where[0]), np.max(where[1])
            bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
            
            mask_dict = {
                "segmentation": mask,
                "area": np.sum(mask),
                "bbox": bbox,
                "predicted_iou": scores[i].item(),
                "stability_score": scores[i].item(),
                "point_coords": [[(bbox[0] + bbox[2]/2), (bbox[1] + bbox[3]/2)]],
                "crop_box": [0, 0, mask.shape[1], mask.shape[0]],
                "obj_id": obj_id,    
                "expression": expression 
            }
            masks_list.append(mask_dict)
            
    torch.save(masks_list, save_path)