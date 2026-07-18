import os
import os.path as osp
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_util
import gc
from sam2.sam2_video_predictor import SAM2VideoPredictor


# --------------------------------------------------------
# Utils
# --------------------------------------------------------
def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()

def decode_gt_mask(rle_obj, H, W):
    if rle_obj is None:
        return np.zeros((H, W), dtype=np.uint8)
    if isinstance(rle_obj, dict):
        return mask_util.decode(rle_obj)
    if isinstance(rle_obj, list):
        merged = mask_util.merge(mask_util.frPyObjects(rle_obj, H, W))
        return mask_util.decode(merged)
    return np.zeros((H, W), dtype=np.uint8)

def compute_jaccard(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)

def compute_f_score(pred, gt, bound_th=0.008):
    """DAVIS-style boundary F without importing skimage at runtime."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    def seg2bmap(mask):
        east = np.zeros_like(mask)
        south = np.zeros_like(mask)
        south_east = np.zeros_like(mask)
        east[:, :-1] = mask[:, 1:]
        south[:-1, :] = mask[1:, :]
        south_east[:-1, :-1] = mask[1:, 1:]

        boundary = (
            np.logical_xor(mask, east)
            | np.logical_xor(mask, south)
            | np.logical_xor(mask, south_east)
        )
        boundary[-1, :] = np.logical_xor(mask[-1, :], east[-1, :])
        boundary[:, -1] = np.logical_xor(mask[:, -1], south[:, -1])
        boundary[-1, -1] = False
        return boundary

    fg_boundary = seg2bmap(pred)
    gt_boundary = seg2bmap(gt)
    bound_pix = bound_th if bound_th >= 1 else np.ceil(
        bound_th * np.linalg.norm(pred.shape)
    )
    radius = int(bound_pix)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (xx * xx + yy * yy <= radius * radius).astype(np.uint8)
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), kernel) > 0
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), kernel) > 0

    n_fg = fg_boundary.sum()
    n_gt = gt_boundary.sum()
    if n_fg == 0 and n_gt > 0:
        precision, recall = 1.0, 0.0
    elif n_fg > 0 and n_gt == 0:
        precision, recall = 0.0, 1.0
    elif n_fg == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    else:
        precision = np.logical_and(fg_boundary, gt_dil).sum() / float(n_fg)
        recall = np.logical_and(gt_boundary, fg_dil).sum() / float(n_gt)

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))

def load_video_frames(val_json_path, anno_dir, frames_root, max_videos=None):
    frame_data = []
    with open(val_json_path, "r") as f:
        val_ids = json.load(f)
    count = 0
    for vid in tqdm(val_ids, desc="Loading Metadata"):
        if max_videos and count >= max_videos:
            break
        vid_str = f"{int(vid):06d}"
        anno_path = osp.join(anno_dir, f"{vid_str}.json")
        if not osp.exists(anno_path):
            continue
        with open(anno_path, "r") as f:
            anno = json.load(f)
        frame_dir = osp.join(frames_root, vid_str)
        if not osp.exists(frame_dir):
            continue
        frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
        if not frames:
            continue
        count += 1
        for fname in frames:
            fid = osp.splitext(fname)[0]
            frame_data.append({
                "video_id": vid_str,
                "frame_id": fid,
                "filename": fname,
                "object_referrals": anno["object_referrals"],
                "image_size": anno["image_size"],
            })
    return frame_data


def build_group_gt_mask(seg_info, target_ids, height, width):
    """Return the union mask for one object_referrals group in one frame."""
    if not seg_info:
        return np.zeros((height, width), dtype=np.uint8)

    seg_track_ids = seg_info.get("track_ids", [])
    mask_rles = seg_info.get("mask_rles", [])
    gt_mask = np.zeros((height, width), dtype=np.uint8)

    # mask_rles follows seg_info['track_ids']; it does not follow referral order.
    for track_id in target_ids:
        try:
            mask_idx = seg_track_ids.index(track_id)
        except ValueError:
            continue
        if mask_idx >= len(mask_rles):
            continue
        single_mask = decode_gt_mask(mask_rles[mask_idx], height, width)
        gt_mask = np.maximum(gt_mask, (single_mask > 0).astype(np.uint8))

    return gt_mask

# === 新增策略函数 ===
def compute_temporal_stability(mask_list):
    if len(mask_list) < 2: return 0.0
    ious = []
    for i in range(len(mask_list) - 1):
        inter = np.logical_and(mask_list[i], mask_list[i+1]).sum()
        union = np.logical_or(mask_list[i], mask_list[i+1]).sum()
        ious.append(inter / (union + 1e-6))
    return np.mean(ious)

def check_border_touch_ratio(mask, threshold=0.3):
    # 简单的贴边检测，针对翻滚的大面积错误
    h, w = mask.shape
    if mask.sum() == 0: return False
    
    # 提取轮廓
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False
    cnt = contours[0] # 取最大的
    
    border_pts = 0
    total_pts = len(cnt)
    for pt in cnt:
        x, y = pt[0]
        if x <= 2 or x >= w-3 or y <= 2 or y >= h-3:
            border_pts += 1
            
    return (border_pts / (total_pts + 1e-6)) < threshold


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    device = "cuda"
    sam2_checkpoint = "./sam2/checkpoints/sam2_hiera_large.pt"
    sam2_model_id = "facebook/sam2-hiera-large"
    video_root = "./datasets/ViCaS/frames"
    anno_dir = "./datasets/ViCaS/annotations/v1.0"
    meta_json = "./datasets/ViCaS/splits/v1.0/val.json"
    pred_mask_root = "./outputs_ViCaS/pred"
    propagated_mask_root = "./outputs_ViCaS/sam2"

    frame_data = load_video_frames(meta_json, anno_dir, video_root)
    videos = {}
    for item in frame_data:
        videos.setdefault(item["video_id"], []).append(item)

    all_j, all_f = [], []
    evaluated_videos = 0

    for vid, frames in videos.items():
        print(f"\n=== Processing Video {vid} ===")
        frames = sorted(frames, key=lambda x: int(x["frame_id"]))
        
        # 建立 GT 映射
        anno_path = osp.join(anno_dir, f"{vid}.json")
        with open(anno_path, "r") as f: anno_json = json.load(f)
        seg_by_filename = {
            seg["filename"]: seg
            for seg in anno_json.get("segmentations", [])
            if seg.get("filename") and seg.get("is_gt")
        }
        gt_map = {
            f_info["frame_id"]: seg_by_filename[f_info["filename"]]
            for f_info in frames
            if f_info["filename"] in seg_by_filename
        }
        frame_id_to_index = {
            int(f_info["frame_id"]): index
            for index, f_info in enumerate(frames)
        }
        index_to_frame_id = {
            index: f_info["frame_id"]
            for index, f_info in enumerate(frames)
        }

        referral_list = frames[0]["object_referrals"]
        num_objs = len(referral_list)
        group_to_refindex = {}
        for ref_idx, ref in enumerate(referral_list):
            tids = tuple(sorted(ref["track_ids"]))
            if tids in group_to_refindex:
                raise ValueError(
                    f"Video {vid} has duplicate object_referrals group {tids}"
                )
            group_to_refindex[tids] = ref_idx

        pred_dir = osp.join(pred_mask_root, vid)
        if not osp.isdir(pred_dir):
            print(f"[Warning] Pred dir not found: {pred_dir}. Evaluating empty predictions.")

        prompt_data = []
        obj_folders = sorted(os.listdir(pred_dir)) if osp.isdir(pred_dir) else []

        for folder in obj_folders:
            try: tids = tuple(sorted(int(x) for x in folder.split("_")))
            except: continue
            if tids not in group_to_refindex:
                print(f"[Warning] Ignoring unknown prediction group {vid}/{folder}")
                continue
            obj_id = group_to_refindex[tids]

            folder_path = osp.join(pred_dir, folder)
            pt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pt")])
            
            # 读取所有 mask 用于分析
            loaded_masks = []
            loaded_idxs = []
            for fn in pt_files:
                f_idx = int(osp.splitext(fn)[0])
                if f_idx not in frame_id_to_index:
                    print(f"[Warning] Ignoring prediction outside video frames: {vid}/{folder}/{fn}")
                    continue
                try:
                    mk = torch.load(osp.join(folder_path, fn), map_location="cpu")
                    if isinstance(mk, list): mk = mk[0]
                    if isinstance(mk, torch.Tensor): mk = mk.numpy()
                    mk = (mk > 0)
                    loaded_masks.append(mk)
                    loaded_idxs.append(frame_id_to_index[f_idx])
                except: continue

            # === 自适应锚点选择策略 ===
            cursor = 0
            last_prompt = -999
            
            while cursor < len(loaded_masks):
                window_size = 5
                end = min(cursor + window_size, len(loaded_masks))
                window_masks = loaded_masks[cursor:end]
                window_idxs = loaded_idxs[cursor:end]
                
                if len(window_masks) < 2: 
                    cursor += 1
                    continue
                
                mid = len(window_masks) // 2
                cand_mask = window_masks[mid]
                cand_idx = window_idxs[mid]

                if not check_border_touch_ratio(cand_mask, threshold=0.2):
                    cursor += 1 
                    continue
                
                stability = compute_temporal_stability(window_masks)
                time_gap = cand_idx - last_prompt
                
                if stability > 0.7 and time_gap >10:
                    prompt_data.append({
                        "frame_idx": cand_idx,
                        "obj_id": obj_id,
                        "mask": cand_mask
                    })
                    last_prompt = cand_idx
                    cursor += 5  
                    
                elif time_gap >= 25:
                    print(f"Fallback triggered at frame {cand_idx} (Stability: {stability:.2f})")
                    
                    prompt_data.append({
                        "frame_idx": cand_idx,
                        "obj_id": obj_id,
                        "mask": cand_mask
                    })
                    last_prompt = cand_idx
                    cursor += 1  
                    
                else:
                    cursor += 1

        print("Prompts:", len(prompt_data))
        memory_pred = {}
        if prompt_data:
            predictor = SAM2VideoPredictor.from_pretrained(
                model_id=sam2_model_id,
                checkpoint=sam2_checkpoint,
                device=device,
            )
            state = predictor.init_state(
                video_path=osp.join(video_root, vid),
                offload_video_to_cpu=True,
            )

            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                for p in prompt_data:
                    predictor.add_new_mask(
                        state, p["frame_idx"], p["obj_id"],
                        torch.from_numpy(p["mask"]).to(device)
                    )

            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                latest_prompt_frame = max(p["frame_idx"] for p in prompt_data)
                for reverse, description in (
                    (False, "Propagate forward"),
                    (True, "Propagate backward"),
                ):
                    propagation_kwargs = {"reverse": reverse}
                    if reverse:
                        # The predictor defaults to the earliest prompt for
                        # reverse mode, which would produce no earlier frames.
                        propagation_kwargs["start_frame_idx"] = latest_prompt_frame
                    outputs = predictor.propagate_in_video(
                        state, **propagation_kwargs
                    )
                    for f_idx, obj_list, masks in tqdm(
                        outputs, desc=description
                    ):
                        masks_np = masks.cpu().numpy()
                        if masks_np.ndim == 4:
                            masks_np = masks_np[:, 0]

                        if f_idx not in index_to_frame_id:
                            continue
                        f_str = index_to_frame_id[f_idx]
                        memory_pred.setdefault(f_str, {})

                        for i, oid in enumerate(obj_list):
                            if reverse and oid in memory_pred[f_str]:
                                # Keep the forward result in the overlap;
                                # reverse propagation fills only earlier frames.
                                continue
                            bool_mask = masks_np[i] > 0.0
                            memory_pred[f_str][oid] = bool_mask.astype(np.uint8)

            predictor.reset_state(state)
            clean_gpu()
        else:
            print(f"[Warning] No valid prompts for video {vid}. Using empty predictions.")

        H, W = anno_json["image_size"]
        group_j = [[] for _ in range(num_objs)]
        group_f = [[] for _ in range(num_objs)]

        # Keep propagated masks separate from the initial zero-shot masks.
        video_output_dir = osp.join(propagated_mask_root, vid)
        os.makedirs(video_output_dir, exist_ok=True)
        refindex_to_group = {
            ref_idx: "_".join(map(str, sorted(ref["track_ids"])))
            for ref_idx, ref in enumerate(referral_list)
        }

        # Save every propagated frame using the referral group name. This is
        # separate from pred_mask_root, which contains the initial masks.
        for fid_str, pred_by_obj in memory_pred.items():
            for obj, pred in pred_by_obj.items():
                if obj not in refindex_to_group:
                    continue
                group_dir = osp.join(
                    video_output_dir, refindex_to_group[obj]
                )
                os.makedirs(group_dir, exist_ok=True)
                torch.save(
                    pred.astype(np.uint8),
                    osp.join(group_dir, f"{fid_str}.pt")
                )

        for fid_str, seg_info in gt_map.items():
            for obj in range(num_objs):
                target_ids = referral_list[obj]["track_ids"]
                gt_mask = build_group_gt_mask(seg_info, target_ids, H, W)

                if fid_str in memory_pred and obj in memory_pred[fid_str]:
                    pred = memory_pred[fid_str][obj]
                else:
                    pred = np.zeros((H, W), dtype=np.uint8) # 没结果就是全黑

                if gt_mask.sum() == 0:
                    group_j[obj].append(1.0 if pred.sum() == 0 else 0.0)
                    group_f[obj].append(1.0 if pred.sum() == 0 else 0.0)
                else:
                    group_j[obj].append(compute_jaccard(pred, gt_mask))
                    group_f[obj].append(compute_f_score(pred, gt_mask))

        video_j = [np.mean(scores) for scores in group_j if scores]
        video_f = [np.mean(scores) for scores in group_f if scores]
        if video_j:
            print(
                f"Video {vid}: Groups={len(video_j)}, "
                f"J={np.mean(video_j):.4f}, F={np.mean(video_f):.4f}, "
                f"J&F={(np.mean(video_f) + np.mean(video_j)) / 2:.4f}"
            )
            all_j.extend(video_j)
            all_f.extend(video_f)
            evaluated_videos += 1

    print("\n========== Overall ==========")
    print(f"Evaluated videos: {evaluated_videos}")
    print(f"Evaluated groups: {len(all_j)}")
    if all_j:
        print("Mean J:   %.4f" % np.mean(all_j))
        print("Mean F:   %.4f" % np.mean(all_f))
        print("Mean J&F: %.4f" % ((np.mean(all_j)+np.mean(all_f))/2))
    else:
        print("No groups were evaluated.")
    print("=============================\n")

if __name__ == "__main__":
    main()
