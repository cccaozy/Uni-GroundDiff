import os
import cv2
import os.path as osp
import torch
import numpy as np
import json
from tqdm import tqdm
from sam2.sam2_video_predictor import SAM2VideoPredictor
import gc
import sys
import warnings
import pdb


warnings.filterwarnings("ignore")

sys.path.append('./GroundingDINO') 
from groundingdino.util.inference import Model as GroundingDINOModel

def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()

def ensure_dir(path):
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)

def load_meta_expressions(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    video_expressions = {}
    for video_name, video_data in data["videos"].items():
        exprs = video_data["expressions"]
        obj_map = {}
        for exp_id, exp_info in exprs.items():
            obj_id = int(exp_info["obj_id"])
            text = exp_info["exp"]
            if obj_id not in obj_map:
                obj_map[obj_id] = text
        video_expressions[video_name] = obj_map
    return video_expressions

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 0.0
    return intersection / union

def load_mask_from_pt(pt_path):
    try:
        mask_data = torch.load(pt_path, map_location='cpu') 
        if isinstance(mask_data, list): mask_data = mask_data[0]
        elif isinstance(mask_data, dict): mask_data = mask_data.get('segmentation', mask_data)
        
        mask = np.array(mask_data)
        mask = (mask > 0).astype(np.bool_) 
        
        parts = pt_path.replace("\\", "/").split("/")
        obj_folder_name = parts[-2]
        if obj_folder_name.isdigit():
            obj_id = int(obj_folder_name)
        else:
            import re
            nums = re.findall(r'\d+', obj_folder_name)
            obj_id = int(nums[0]) if nums else 1 
        return mask, obj_id
    except Exception as e:
        print(f"Error loading {pt_path}: {e}")
        return None, None

def get_best_frame_deva_consensus(masks, sigma=5.0):
    n = len(masks)
    if n == 0: return -1, 0.0
    if n == 1: return 0, 1.0
        
    scores = np.zeros(n)
    for i in range(n):
        support_score = 0.0
        for j in range(n):
            if i == j: continue
            iou = compute_iou(masks[i], masks[j])
            if iou > 0.5: 
                distance = abs(i - j)
                weight = np.exp(-(distance ** 2) / (2 * (sigma ** 2)))
                support_score += iou * weight
        scores[i] = support_score

    best_idx = np.argmax(scores)
    return best_idx, scores[best_idx]  

def check_mask_quality_vs_box(mask, box_xyxy, intersection_threshold=0.5, area_ratio_threshold=0.3):

    h, w = mask.shape
    x1, y1, x2, y2 = box_xyxy

    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    
    box_area = (x2 - x1) * (y2 - y1)
    if box_area <= 0: return False 
    
    mask_area = mask.sum()
    if mask_area == 0: return False

    intersection = mask[y1:y2, x1:x2].sum()
    
    consistency_ratio = intersection / mask_area
    
    occupancy_ratio = intersection / box_area

    return (
        consistency_ratio >= intersection_threshold
        and occupancy_ratio >= area_ratio_threshold
    )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sam2_checkpoint = "./sam2/checkpoints/sam2_hiera_large.pt"
    sam2_model_id = "facebook/sam2-hiera-large"
    
    gdino_config = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gdino_weights = "./GroundingDINO/groundingdino_swint_ogc.pth"
    
    video_root = "./datasets/ref-davis/valid/JPEGImages"
    meta_json = "./datasets/ref-davis/meta_expressions/valid/meta_expressions.json"
    pred_mask_root = "./outputs/pred"
    out_dir = "./outputs/sam2"
    ensure_dir(out_dir)

    if device == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    print("Loading SAM 2...")
    predictor = SAM2VideoPredictor.from_pretrained(
        model_id=sam2_model_id,
        checkpoint=sam2_checkpoint,
        device=device
    )
    
    print("Loading GroundingDINO...")
    gdino_model = GroundingDINOModel(
        model_config_path=gdino_config,
        model_checkpoint_path=gdino_weights,
        device=device
    )
    
    print("Loading Meta Expressions...")
    video_expressions = load_meta_expressions(meta_json)

    video_list = sorted(os.listdir(pred_mask_root))

    for video in video_list:
        video_dir = osp.join(video_root, video)
        video_pred_dir = osp.join(pred_mask_root, video)

        if not osp.exists(video_dir) or not osp.isdir(video_pred_dir):
            continue

        print(f"=== Processing video: {video} ===")
        
        obj_captions = video_expressions.get(video, {})
        
        state = predictor.init_state(video_path=video_dir, offload_video_to_cpu=True)
        prompt_data = [] 

        obj_folders = sorted(os.listdir(video_pred_dir))
        
        for obj_folder in obj_folders:
            obj_dir = osp.join(video_pred_dir, obj_folder)
            if not osp.isdir(obj_dir): continue

            pt_files = sorted([f for f in os.listdir(obj_dir) if f.endswith(".pt")])
            if not pt_files: continue

            prompt_interval = 20
            
            for i in range(0, len(pt_files), prompt_interval):
                window_files = pt_files[i : i + prompt_interval]
                if not window_files: continue

                window_masks = []
                valid_indices = [] 
                current_obj_id = None
                
                for offset, f_name in enumerate(window_files):
                    p_path = osp.join(obj_dir, f_name)
                    m, o_id = load_mask_from_pt(p_path)
                    if m is not None:
                        window_masks.append(m)
                        valid_indices.append(i + offset)
                        current_obj_id = o_id

                if not window_masks: continue
                
                caption = obj_captions.get(current_obj_id, None)

                candidates = [] 
                candidates.append((valid_indices[0], window_masks[0]))
                
                if len(window_masks) > 2:
                    best_local_idx, best_score = get_best_frame_deva_consensus(window_masks, sigma=3.0)
                    if best_local_idx != -1 and best_score > 0.5:
                        if valid_indices[best_local_idx] != valid_indices[0]:
                            candidates.append((valid_indices[best_local_idx], window_masks[best_local_idx]))

                for frame_idx_abs, candidate_mask in candidates:
                    frame_name = pt_files[frame_idx_abs]
                    frame_num = int(osp.splitext(frame_name)[0])
                    
                    if not caption:
                        prompt_data.append({
                            "frame_idx": frame_num,
                            "obj_id": current_obj_id,
                            "type": "mask",
                            "data": candidate_mask
                        })
                        continue

                    img_path = osp.join(video_dir, f"{frame_num:05d}.jpg")
                    if not osp.exists(img_path): continue
                    
                    image_cv = cv2.imread(img_path)
                    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                    
                    with torch.cuda.amp.autocast(enabled=False):
                        detections, phrases = gdino_model.predict_with_caption(
                            image=image_cv,
                            caption=caption,
                            box_threshold=0.3,
                            text_threshold=0.25
                        )
                    
                    if len(detections.xyxy) > 0:
                        best_box_idx = np.argmax(detections.confidence)
                        best_box = detections.xyxy[best_box_idx] 
                        
                        use_mask = check_mask_quality_vs_box(
                            candidate_mask, 
                            best_box, 
                            intersection_threshold=0.5,
                            area_ratio_threshold=0.5   
                        )
                        
                        if use_mask:
                            prompt_data.append({
                                "frame_idx": frame_num,
                                "obj_id": current_obj_id,
                                "type": "mask",
                                "data": candidate_mask
                            })
                        else:
                            prompt_data.append({
                                "frame_idx": frame_num,
                                "obj_id": current_obj_id,
                                "type": "box",
                                "data": best_box
                            })
                    else:
                        prompt_data.append({
                            "frame_idx": frame_num,
                            "obj_id": current_obj_id,
                            "type": "mask",
                            "data": candidate_mask
                        })

        if not prompt_data:
            predictor.reset_state(state)
            continue

        prompt_data.sort(key=lambda x: x["frame_idx"])
        
        unique_prompts = []
        seen = set()
        for p in prompt_data:
            key = (p["frame_idx"], p["obj_id"])
            if key not in seen:
                unique_prompts.append(p)
                seen.add(key)
        
        print(f"  Total prompts: {len(unique_prompts)}")

        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            for p in unique_prompts:
                f_idx = p["frame_idx"]
                o_id = p["obj_id"]
                
                if p["type"] == "mask":
                    mask_np = p["data"]
                    mask_tensor = torch.from_numpy(mask_np).float().to(device)
                    
                    if mask_tensor.ndim == 3:
                        mask_tensor = mask_tensor.squeeze(0) 
                    
                    predictor.add_new_mask(
                        state,
                        frame_idx=f_idx,
                        obj_id=o_id,
                        mask=mask_tensor
                    )
                elif p["type"] == "box":
                    box_np = p["data"]
                    box_tensor = torch.from_numpy(box_np).float()
                    
                    if box_tensor.ndim == 1:
                        box_tensor = box_tensor.unsqueeze(0) 
                    
                    if box_tensor.shape[1] != 4:
                        print(f"Warning: box shape is {box_tensor.shape}, expected [*, 4]")
                        continue
                    
                    predictor.add_new_points_or_box(
                        state,
                        frame_idx=f_idx,
                        obj_id=o_id,
                        box=box_tensor
                    )

        save_dir = osp.join(out_dir, video)
        ensure_dir(save_dir)

        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            for f_idx, obj_ids, masks in tqdm(predictor.propagate_in_video(state), desc=f"Propagating {video}"):

                masks_np = masks.detach().cpu().numpy()
                if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                    masks_np = masks_np[:, 0, :, :]

                masks_tensor_cpu = masks.detach().cpu() > 0.0 
                
                for i, obj_id in enumerate(obj_ids):
                    mask_prob = masks_np[i]
                    mask_bin = (mask_prob > 0.0).astype(np.uint8) 
                    mask_png = mask_bin * 255
                    
                    png_path = osp.join(save_dir, f"{f_idx:05d}_obj{obj_id}.png")
                    cv2.imwrite(png_path, mask_png)

                    single_mask_tensor = masks_tensor_cpu[i]
                    pt_path = osp.join(save_dir, f"{f_idx:05d}_obj{obj_id}.pt")
                    torch.save(single_mask_tensor, pt_path)

        predictor.reset_state(state)
        clean_gpu()

    print("All videos done.")

if __name__ == "__main__":
    main()
