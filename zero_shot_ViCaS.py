import os
import os.path as osp
import json
import torch
import numpy as np
import cv2
import re
from collections import defaultdict 
from tqdm import tqdm
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from scheduler_dev import DDIMSchedulerDev
import argparse
import pycocotools.mask as mask_util
import warnings
import pdb

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument('--max_phrase_num', default=30, type=int)
    
    parser.add_argument('--self_enhanced', action='store_true')
    parser.add_argument('--sam_enhanced', action='store_true')
    parser.add_argument("--self_res", type=int, default=64)
    parser.add_argument("--cross_res", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--tao", type=float, default=0.5)
    
    parser.add_argument("--val_json", type=str, 
                        default="./datasets/ViCaS/splits/v1.0/val.json")
    parser.add_argument("--anno_dir", type=str, 
                        default="./datasets/ViCaS/annotations/v1.0")
    parser.add_argument("--frames_root", type=str, 
                        default="./datasets/ViCaS/frames")
    #
    parser.add_argument("--pred_path", type=str, default="/data/username/attn/")
    parser.add_argument("--output_dir", type=str, default="./outputs_ViCaS/pred")
    parser.add_argument("--sam_root", type=str, default="/data/username/sam_Grounded")
    parser.add_argument("--max_videos", type=int, default=100)
    parser.add_argument(
        "--video_ids", type=str, default="",
        help="Comma-separated video IDs to process, e.g. 000003 or 000000,000003"
    )

    return parser.parse_args()

def aggregate_cross_attention(ldm_stable, phrase_ids, cross_attn_map, video_id, frame_id, p_idx):
    cache_dir = './outputs_ViCaS/scores'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    phrase_key = '_'.join(map(str, phrase_ids))
    cache_path = os.path.join(cache_dir, f'{video_id}_{frame_id}_{p_idx}_{phrase_key}.pt')
    device = cross_attn_map.device
    
    if not os.path.exists(cache_path):
        token_strs = [ldm_stable.tokenizer.decode([tid]) for tid in phrase_ids]

        text_input = ldm_stable.tokenizer(
            token_strs,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[1]
        
        if embeddings.shape[0] == 0:
            return cross_attn_map.mean(dim=0)

        scores = embeddings @ embeddings.T
        scores = scores - scores.min()
        scores = scores / (scores.max() + 1e-8)
        
        weighted_s = scores[-1].softmax(dim=-1)
        torch.save(weighted_s.cpu(), cache_path)
    else:
        weighted_s = torch.load(cache_path, map_location='cpu')
    
    weighted_s = weighted_s.to(device)

    if len(weighted_s) != cross_attn_map.shape[0]:
        return cross_attn_map.mean(dim=0)

    weighted_cross_attn = torch.zeros_like(cross_attn_map[0], device=device)
    for i in range(cross_attn_map.shape[0]):
        weighted_cross_attn += weighted_s[i] * cross_attn_map[i]
            
    return weighted_cross_attn

def build_id_to_phrases_map(caption_raw_en):

    # 匹配所有 [phrase]<mask_ids> 结构
    # Keep each phrase local to its own brackets. Invalid markers such as
    # <mask_?> are ignored without consuming the next valid object marker.
    pattern = re.compile(
        r"\[([^\[\]]*?)\]((?:<mask_\d+(?:\s*,\s*\d+)*>)+)"
    )
    matches = pattern.findall(caption_raw_en)
    
    id_map = defaultdict(list)
    
    for phrase, mask_tags in matches:
        clean_phrase = phrase.strip()
        ids_str = ",".join(
            value
            for group in re.findall(r"<mask_([0-9]+(?:\s*,\s*[0-9]+)*)>", mask_tags)
            for value in group.split(",")
        )
        # 解析 ID 列表
        try:
            id_list = [int(x) for x in ids_str.split(',') if x.strip().isdigit()]
        except ValueError:
            continue
            
        # 将该短语注册到列表中的每一个 ID 下
        for uid in id_list:
            id_map[uid].append(clean_phrase)
            
    return id_map

def build_phrase_entries(caption_raw_en):
    """Parse ViCaS phrases while preserving the full mask-id group."""
    pattern = re.compile(
        r"\[([^\[\]]*?)\]((?:<mask_\d+(?:\s*,\s*\d+)*>)+)"
    )
    entries = []
    for phrase, mask_tags in pattern.findall(caption_raw_en):
        ids = []
        for group in re.findall(
            r"<mask_([0-9]+(?:\s*,\s*[0-9]+)*)>", mask_tags
        ):
            ids.extend(int(x.strip()) for x in group.split(',') if x.strip())
        phrase = phrase.strip()
        if phrase and ids:
            entries.append((phrase, frozenset(ids)))
    return entries

def phrases_for_target_group(caption_raw_en, target_ids):
    target_ids = frozenset(int(x) for x in target_ids)
    return list({
        phrase
        for phrase, phrase_ids in build_phrase_entries(caption_raw_en)
        if phrase_ids.issubset(target_ids)
    })

def find_subsequence(sequence, pattern):
    n = len(sequence)
    m = len(pattern)
    if m == 0: return None
    for i in range(n - m + 1):
        if sequence[i:i+m] == pattern:
            return list(range(i, i+m))
    return None

def db_eval_iou(annotation, segmentation, void_pixels=None):
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
        segmentation[void_pixels] = 0
        annotation[void_pixels] = 0

    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    intersection = np.sum(annotation & segmentation)
    union = np.sum(annotation | segmentation)

    if union == 0:
        return 1.0
    else:
        return intersection / union

def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
        segmentation[void_pixels] = 0
        annotation[void_pixels] = 0

    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)
    
    if np.sum(annotation) == 0 and np.sum(segmentation) == 0:
        return 1.0

    def get_boundary(mask, bound_th):
        h, w = mask.shape
        mask_uint8 = mask.astype(np.uint8)
        bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.sqrt(h * w))
        kernel_size = int(2 * bound_pix + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask_uint8, kernel)
        eroded = cv2.erode(mask_uint8, kernel)
        boundary = dilated - eroded
        return boundary > 0

    gt_boundary = get_boundary(annotation, bound_th)
    fg_boundary = get_boundary(segmentation, bound_th)

    intersection = np.sum(gt_boundary & fg_boundary)
    sum_fg = np.sum(fg_boundary)
    sum_gt = np.sum(gt_boundary)
    
    if sum_fg == 0: precision = 0
    else: precision = intersection / sum_fg
    
    if sum_gt == 0: recall = 0
    else: recall = intersection / sum_gt
    
    if precision + recall == 0:
        f_score = 0.0
    else:
        f_score = 2 * precision * recall / (precision + recall)
        
    return f_score

def decode_gt_mask(seg_data, height, width):
    if not seg_data:
        return np.zeros((height, width), dtype=np.uint8)
    if isinstance(seg_data, dict) and 'counts' in seg_data:
        mask = mask_util.decode(seg_data)
        return mask
    elif isinstance(seg_data, list):
        rles = mask_util.frPyObjects(seg_data, height, width)
        rle = mask_util.merge(rles)
        mask = mask_util.decode(rle)
        return mask
    else:
        return np.zeros((height, width), dtype=np.uint8)

def self_enhanced_fun(self_attn, cross_attn_ori, res, beta=0.5):
    if self_attn.dim() == 2:
        dim = int(np.sqrt(self_attn.shape[0]))
        self_attn = self_attn.reshape(dim, dim, dim, dim)

    if self_attn.shape[0] != res:
        self_attn_flat = self_attn.reshape(1, 1, self_attn.shape[0]**2, self_attn.shape[1]**2)
        self_attn = F.interpolate(self_attn_flat, size=(res**2, res**2), mode='bilinear')
        self_attn = self_attn.reshape(res, res, res, res)

    valid_points_y, valid_points_x = torch.where(cross_attn_ori > beta)
    if len(valid_points_y) == 0:
        return cross_attn_ori

    avg_self_attn = torch.zeros_like(cross_attn_ori)
    for y, x in zip(valid_points_y, valid_points_x):
        tmp = self_attn[int(y), int(x)]
        avg_self_attn += tmp
    
    if avg_self_attn.max() > 0:
        avg_self_attn = avg_self_attn - avg_self_attn.min()
        avg_self_attn = avg_self_attn / avg_self_attn.max()
    
    return avg_self_attn

def load_and_merge_attention(tokenizer, full_text, pred_base_path, video_id, frame_id, res=16):
    full_input_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    max_len = 75 
    chunks = [full_input_ids[i:i + max_len] for i in range(0, len(full_input_ids), max_len)]
    all_cross_maps = []
    all_self_maps = [] 
    
    base_dir_cross = osp.join(pred_base_path, video_id, frame_id, f"cross_{res}")
    base_dir_self = osp.join(pred_base_path, video_id, frame_id, f"self_64")
    
    first_fname = f"{frame_id}.pt"
    first_path = osp.join(base_dir_cross, first_fname)
    if not osp.exists(first_path):
        return None, None, full_input_ids

    for i, chunk in enumerate(chunks):
        fname = f"{frame_id}.pt" if i == 0 else f"{frame_id}_{i}.pt"
        pt_path_cross = osp.join(base_dir_cross, fname)
        if not osp.exists(pt_path_cross):
            all_cross_maps.append(torch.zeros((len(chunk), res, res)))
        else:
            try:
                attn = torch.load(pt_path_cross, map_location='cpu') 
                attn = attn.permute(2, 0, 1)
                if attn.shape[0] == 77: valid_attn = attn[1 : 1 + len(chunk)] 
                else: valid_attn = attn[:len(chunk)]
                all_cross_maps.append(valid_attn)
            except:
                all_cross_maps.append(torch.zeros((len(chunk), res, res)))

        pt_path_self = osp.join(base_dir_self, fname)
        if osp.exists(pt_path_self):
            try:
                s_attn = torch.load(pt_path_self, map_location='cpu')
                if s_attn.dim() == 2:
                    dim = int(np.sqrt(s_attn.shape[0]))
                    s_attn = s_attn.reshape(dim, dim, dim, dim)
                all_self_maps.append(s_attn)
            except: pass

    if not all_cross_maps: return None, None, full_input_ids
    merged_cross = torch.cat(all_cross_maps, dim=0)
    merged_self = torch.stack(all_self_maps, dim=0).mean(dim=0) if all_self_maps else None
    return merged_cross, merged_self, full_input_ids

def sam_refine_mask(sam_proposal_masks, mask, beta=0.3, tao=0.5):
    device = mask.device
    sam_proposal_masks = sam_proposal_masks.to(device)
    
    if mask.dim() == 2:
        mask = mask.unsqueeze(0) # [1, H, W]

    pseudo_part = (mask > beta).float()
    
    if sam_proposal_masks.dim() == 4:
        sam_proposal_masks = sam_proposal_masks.squeeze(1) # [N, H, W]
        
    H, W = mask.shape[-2:]
    if sam_proposal_masks.shape[-2:] != (H, W):
        sam_proposal_masks = F.interpolate(
            sam_proposal_masks.unsqueeze(1), # 需要 [N, C, H, W] 格式
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

    sam_proposal_masks = (sam_proposal_masks > 0.5).float()
    intersection = (sam_proposal_masks * pseudo_part).sum(dim=(1, 2))
    area_sam = sam_proposal_masks.sum(dim=(1, 2))
    area_diff = pseudo_part.sum()

    s1 = intersection / (area_diff + 1e-9)
    s2 = intersection / (area_sam + 1e-9)
    
    is_matched = (s1 > tao) | (s2 > tao) 
    
    # pdb.set_trace()
    if is_matched.any():
        matched_masks = sam_proposal_masks[is_matched] # [M, H, W]
        refined_mask = matched_masks.max(dim=0)[0].unsqueeze(0)
    else:
        refined_mask = pseudo_part 
        
    return refined_mask


def load_video_frames(
    val_json_path, anno_dir, frames_root, max_videos=None,
    frame_interval=1, video_ids=None
):
    frame_data = []
    if not os.path.exists(val_json_path):
        print(f"Error: JSON file not found at {val_json_path}")
        return []

    with open(val_json_path, "r") as f:
        val_ids = json.load(f)

    count = 0
    for vid in tqdm(val_ids, desc="Loading Metadata"):
        if max_videos and count >= max_videos: break
        
        vid_str = f"{int(vid):06d}" 
        if video_ids is not None and vid_str not in video_ids:
            continue
        anno_path = osp.join(anno_dir, f"{vid_str}.json")
        if not osp.exists(anno_path): continue

        with open(anno_path, "r") as f:
            anno = json.load(f)

        video_segmentations = anno.get("segmentations", [])
        segmentations_by_filename = {
            seg.get("filename"): seg for seg in video_segmentations
            if seg.get("filename") and seg.get("is_gt")
        }

        object_referrals = anno.get("object_referrals", [])

        frame_dir = osp.join(frames_root, vid_str)
        if not osp.exists(frame_dir): continue
        
        all_frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
        if not all_frames: continue
        
        count += 1
        
        for fname in all_frames:
            fid = osp.splitext(fname)[0] 

            # Match annotations by filename, rather than relying on list order.
            current_gt = segmentations_by_filename.get(fname)
                
            if int(fid) % frame_interval != 0: continue
            
            frame_data.append({
                "video_id": vid_str,
                "frame_id": fid,
                "caption_raw_en": anno.get("caption_raw_en", ""),
                "caption_parsed_en_gpt": anno.get("caption_parsed_en_gpt", ""),
                "current_gt": current_gt, 
                "image_size": anno.get("image_size", [720, 1280]),
                "object_referrals": object_referrals 
            })
            
    return frame_data

def evaluate_model(cfg, frame_data):
    print(f"Loading models on {cfg.device}...")
    
    # 初始化模型
    scheduler = DDIMSchedulerDev(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
        clip_sample=False, steps_offset=1
    )
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", scheduler=scheduler
    ).to(cfg.device)
    tokenizer = ldm_stable.tokenizer
    os.makedirs(cfg.output_dir, exist_ok=True)

    global_metrics = {
        "J": 0.0, "F": 0.0, "JF": 0.0,
    }
    
    # 遍历每一帧
    for frame_info in tqdm(frame_data, desc="Evaluating"):
        video_id = frame_info["video_id"]
        frame_id = frame_info["frame_id"]
        caption_raw = frame_info["caption_raw_en"]
        
        target_groups = frame_info.get("object_referrals", [])
        if not target_groups:
            continue

        # Preserve multi-track phrase ownership when selecting prompts.

        caption_for_attn = frame_info.get("caption_parsed_en_gpt", caption_raw)
        if not caption_for_attn: caption_for_attn = caption_raw

        cross_attn, self_attn, full_token_ids = load_and_merge_attention(
            tokenizer, caption_for_attn, cfg.pred_path, video_id, frame_id, res=cfg.cross_res
        )
        if cross_attn is None: continue
        
        # 加载 GT
        current_gt_data = frame_info.get("current_gt")
        if current_gt_data is None: continue
        gt_track_ids = current_gt_data.get("track_ids", []) 
        gt_rles = current_gt_data.get("mask_rles", [])
        if not gt_track_ids: continue

        H_img, W_img = frame_info["image_size"]

        item_id = 0 
        for group_info in target_groups:
            target_ids = group_info["track_ids"]
            
            sam_path = osp.join(
                cfg.sam_root, video_id, frame_id, f'{item_id}.pt'
            )
            if cfg.sam_enhanced and os.path.exists(sam_path):
                sam_everything_masks = torch.load(sam_path, map_location=cfg.device)
            else:
                sam_everything_masks = []

            item_id = item_id + 1

            group_folder_name = "_".join(map(str, target_ids))
            
            relevant_phrases = set(
                phrases_for_target_group(caption_raw, target_ids)
            )

            if not relevant_phrases:
                continue 

            combined_attn_map = None

            valid_gt_ids = [tid for tid in target_ids if tid in gt_track_ids]
            if not valid_gt_ids: continue 

            for phrase in relevant_phrases:
                phrase_ids = tokenizer(phrase, add_special_tokens=False).input_ids
                indices = find_subsequence(full_token_ids, phrase_ids)
                if indices is None: continue
                
                valid_maps = [cross_attn[i] for i in indices if i < cross_attn.shape[0]]
                if not valid_maps: continue
                stacked_maps = torch.stack(valid_maps)

                attn_map_phrase = aggregate_cross_attention(
                    ldm_stable=ldm_stable,
                    phrase_ids=phrase_ids,   
                    cross_attn_map=stacked_maps,
                    video_id=video_id,
                    frame_id=frame_id,
                    p_idx=target_ids[0] 
                )

                if attn_map_phrase.max() > 0:
                    attn_map_phrase = (attn_map_phrase - attn_map_phrase.min()) / (attn_map_phrase.max() - attn_map_phrase.min())

                if combined_attn_map is None:
                    combined_attn_map = attn_map_phrase
                else:
                    combined_attn_map = torch.max(combined_attn_map, attn_map_phrase)

            if combined_attn_map is None: continue
            
            if cfg.self_enhanced and self_attn is not None:
                combined_attn_map = self_enhanced_fun(self_attn, combined_attn_map, cfg.self_res, beta=cfg.beta)

            attn_map_up = F.interpolate(
                combined_attn_map.unsqueeze(0).unsqueeze(0), 
                size=(H_img, W_img), 
                mode="bilinear", 
                align_corners=False
            )[0, 0]
            
            final_mask_tensor = (attn_map_up > cfg.alpha).float()

            if len(sam_everything_masks) > 0:
                final_mask = sam_refine_mask(
                    sam_everything_masks, final_mask_tensor.to(cfg.device),
                    cfg.alpha, cfg.tao
                )
            else:
                final_mask = (final_mask_tensor.to(cfg.device) > cfg.alpha).float()
            
            pred_mask_np = final_mask.cpu().numpy().astype(np.uint8)
            pred_mask_np = pred_mask_np.squeeze()

            save_dir = os.path.join(cfg.output_dir, video_id, group_folder_name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(pred_mask_np, os.path.join(save_dir, f"{int(frame_id):05d}.pt"))

            vis_path = os.path.join(save_dir, f"{int(frame_id):05d}_vis.png")
            # if not os.path.exists(vis_path):
            image_path = osp.join(cfg.frames_root, video_id, f"{int(frame_id):05d}.jpg")
            original_img = cv2.imread(image_path)
            if original_img is not None:
                mask_uint8 = (pred_mask_np * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
                if original_img.shape[:2] != heatmap.shape[:2]:
                    original_img = cv2.resize(original_img, (heatmap.shape[1], heatmap.shape[0]))
                vis_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
                cv2.imwrite(vis_path, vis_img)

            gt_mask_union = np.zeros((H_img, W_img), dtype=np.uint8)
            for tid in valid_gt_ids:
                try:
                    gt_idx = gt_track_ids.index(tid)
                    seg_data = gt_rles[gt_idx]
                    gt_mask_single = decode_gt_mask(seg_data, H_img, W_img)
                    gt_mask_union = np.maximum(gt_mask_union, gt_mask_single)
                except (ValueError, IndexError):
                    pass

            J = db_eval_iou(gt_mask_union, pred_mask_np)
            F_score = db_eval_boundary(gt_mask_union, pred_mask_np)

            global_metrics["J"] += J
            global_metrics["F"] += F_score
            global_metrics["JF"] += 0.5 * (J + F_score)
        

def main():
    cfg = parse_args()
    
    if not os.path.exists(cfg.pred_path):
        print(f"Warning: Prediction path {cfg.pred_path} does not exist!")
    
    selected_video_ids = None
    if cfg.video_ids.strip():
        selected_video_ids = {
            f"{int(video_id.strip()):06d}"
            for video_id in cfg.video_ids.split(",")
            if video_id.strip()
        }

    frame_data = load_video_frames(
        val_json_path=cfg.val_json,
        anno_dir=cfg.anno_dir,
        frames_root=cfg.frames_root,
        max_videos=cfg.max_videos,
        frame_interval=1,
        video_ids=selected_video_ids
    )
    
    print(f"Loaded {len(frame_data)} frames.")
    evaluate_model(cfg, frame_data)

if __name__ == "__main__":
    main()
