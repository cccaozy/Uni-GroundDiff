import os
import os.path as osp
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
import pdb
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from scheduler_dev import DDIMSchedulerDev
import transformers
import spacy
from detectron2.structures import ImageList
import argparse
from PIL import Image
import sys

sys.path.append('./GroundingDINO')
from groundingdino.util.inference import load_model, load_image as gd_load_image, predict

sys.path.append("./segment-anything-third-party")
from segment_anything import sam_model_registry, SamPredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training and testing pipeline."
    )
    # setting
    parser.add_argument(
        '--training', 
        action='store_true', 
        help='Training enable.'
    )
    parser.add_argument(
        '--local_rank', 
        type=int, 
        help='Local rank for ddp.'
    )
    parser.add_argument(
        '--backend', 
        default='nccl', 
        type=str, 
        help='Backend for ddp.'
    )
    parser.add_argument(
        '--seed', 
        default=3407, 
        type=int, 
        help='Random Seed.'
    )
    parser.add_argument(
        '--num_gpus',
        default=4, 
        type=int,
        help='Number of GPUs to use (applies to both training and testing).'
    )

    # model
    parser.add_argument(
        '--detectron2_ckpt', 
        default='./pretrained_models/fpn/model_final_cafdb1.pkl', 
        type=str, 
        help='ckpt path of fpn from detectron2.'
    )
    parser.add_argument(
        '--detectron2_cfg', 
        default='./configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x_train.yaml',
        type=str, 
        help='cfg path of fpn from detectron2.'
    )
    parser.add_argument(
        '--max_sequence_length',
        default=230,
        type=int,
        help='Max length of the input language sequence.'
    )
    parser.add_argument(
        '--max_seg_num',
        default=64,
        type=int,
        help='Max num of the noun phrase to be segmented.'
    )
    parser.add_argument(
        '--max_phrase_num',
        default=30,
        type=int,
        help='Max num of the noun phrase to be segmented.'
    )
    # data
    parser.add_argument(
        '--data_path',
        default='./datasets/coco', 
        type=str,
        help='The path to the data directory.'
    )
    
    parser.add_argument(
        '--data_dir',
        default='./datasets', 
        type=str,
        help='The path to the data directory.'
    )

    parser.add_argument( 
        '--output_dir', 
        default="./output", 
        type=str, 
        help='Saving dir.'
    )
    parser.add_argument(
        '--self_enhanced', 
        type=bool,
        default=False,
        help='.'
    )
    parser.add_argument(
        '--sam_enhanced', 
        type=bool,
        default=False,
        help='.'
    )
    parser.add_argument(
        "--self_res",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--cross_res",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--tao",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--grounding_dino_config",
        type=str,
        default="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help='GroundingDINO config path.'
    )
    parser.add_argument(
        "--grounding_dino_ckpt",
        type=str,
        default="./groundingdino_swint_ogc.pth",
        help='GroundingDINO checkpoint path.'
    )
    parser.add_argument(
        "--sam_ckpt",
        type=str,
        default="./segment-anything/sam_vit_h_4b8939.pth",
        help='SAM checkpoint path.'
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        help='SAM model type.'
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.25,
        help='GroundingDINO box threshold.'
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.25,
        help='GroundingDINO text threshold.'
    )
    parser.add_argument(
        "--intersection_threshold",
        type=float,
        default=0.8,
        help='Threshold for mask-box intersection consistency.'
    )
    parser.add_argument(
        "--area_ratio_threshold",
        type=float,
        default=0.5,
        help='Threshold for mask area / box area ratio.'
    )
    return parser.parse_args()


nlp = spacy.load("en_core_web_sm")

_grounding_dino_model = None
_sam_predictor = None


def get_grounding_dino_model(config_path, ckpt_path, device='cuda'):
    """延迟加载 GroundingDINO 模型"""
    global _grounding_dino_model
    if _grounding_dino_model is None:
        print("[INFO] Loading GroundingDINO model...")
        _grounding_dino_model = load_model(config_path, ckpt_path, device=device)
        print("[INFO] GroundingDINO model loaded.")
    return _grounding_dino_model


def get_sam_predictor(ckpt_path, model_type, device='cuda'):
    """延迟加载 SAM Predictor"""
    global _sam_predictor
    if _sam_predictor is None:
        print("[INFO] Loading SAM model...")
        sam = sam_model_registry[model_type](checkpoint=ckpt_path)
        sam.to(device=device)
        _sam_predictor = SamPredictor(sam)
        print("[INFO] SAM model loaded.")
    return _sam_predictor


def detect_boxes_with_grounding_dino(model, image_path, caption, box_threshold=0.25, text_threshold=0.25, device='cuda'):

    # 加载图像
    image_source, image_transformed = gd_load_image(image_path)
    
    # 运行检测
    boxes, logits, phrases = predict(
        model=model,
        image=image_transformed,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )
    
    # 将归一化坐标转换为绝对坐标 (cxcywh -> xyxy)
    h, w, _ = image_source.shape
    boxes_xyxy = boxes.clone()
    # boxes 格式是 cxcywh 归一化坐标
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w  # x1
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h  # y1
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w  # x2
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h  # y2
    
    return boxes_xyxy, logits, phrases, image_source


def check_mask_quality_vs_box(mask, box_xyxy, intersection_threshold=0.4, area_ratio_threshold=0):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    mask = (mask > 0).astype(np.float32)
    h, w = mask.shape
    x1, y1, x2, y2 = box_xyxy

    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    
    box_area = (x2 - x1) * (y2 - y1)
    if box_area <= 0:
        return False
    
    mask_area = mask.sum()

    if mask_area == 0:
        return False

    intersection = mask[y1:y2, x1:x2].sum()
    
    consistency_ratio = intersection / mask_area
    occupancy_ratio = mask_area / box_area
    
    if consistency_ratio > intersection_threshold and occupancy_ratio > area_ratio_threshold:
        return True
    else:
        return False
    

def generate_mask_from_box(sam_predictor, image, box_xyxy, device='cuda'):

    sam_predictor.set_image(image)
    
    box_np = np.array(box_xyxy)
    
    masks, scores, logits = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_np[None, :],  # SAM 需要 [1, 4] 形状
        multimask_output=False,
    )
    
    # masks shape: [1, H, W]，取第一个
    mask = torch.from_numpy(masks[0]).float().to(device)
    return mask


def refine_mask_with_grounding_dino(
    final_mask, 
    image_path, 
    image_source,
    caption,
    grounding_dino_model,
    sam_predictor,
    box_threshold=0.25,
    text_threshold=0.25,
    intersection_threshold=0.3, 
    area_ratio_threshold=0.5,
    device='cuda'
):

    # 使用 GroundingDINO 检测候选框
    try:
        boxes_xyxy, logits, phrases, _ = detect_boxes_with_grounding_dino(
            model=grounding_dino_model,
            image_path=image_path,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
        )
    except Exception as e:
        print(f"[WARNING] GroundingDINO detection failed: {e}")
        return final_mask, False
    
    if len(boxes_xyxy) == 0:
        return final_mask, False
    
    # 获取置信度最高的候选框
    best_idx = logits.argmax().item()
    best_box = boxes_xyxy[best_idx].cpu().numpy()
    best_confidence = logits[best_idx].item()
    
    # 检查掩码质量
    is_good_quality = check_mask_quality_vs_box(
        final_mask, 
        best_box, 
        intersection_threshold, 
        area_ratio_threshold
    )
    
    if is_good_quality:
        # 质量良好，直接返回原掩码
        return final_mask, False
    else:
        # 质量不佳，使用候选框重新生成掩码
        new_mask = generate_mask_from_box(sam_predictor, image_source, best_box, device)
        return new_mask, True


def load_ground_truth_mask(video_name, frame_id, dataset_path):
    mask_path = os.path.join(dataset_path, video_name, f"{frame_id}.png")
    mask_img = Image.open(mask_path)
    mask_array = np.array(mask_img)
    binary_mask = mask_array > 0
    mask_tensor = torch.from_numpy(binary_mask.astype(np.float32))
    return mask_tensor


def extract_noun_token_indices(caption, tokenizer):

    doc = nlp(caption)
    noun_words = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]

    tokenized = tokenizer(caption, return_tensors="pt", truncation=True, max_length=77)
    input_ids = tokenized.input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    selected_indices = []
    for noun in noun_words:
        for idx, tok in enumerate(tokens):
            tok_clean = tok.replace('Ġ', '').replace('▁', '').replace('##', '')
            if tok_clean.lower() == noun.lower():
                selected_indices.append(idx)
                break
    return selected_indices


def find_nearest_period_index(word_list):
    target_index = 74
    nearest_period_index = None

    for i, word in enumerate(word_list[:75]):
        if word == '.':
            nearest_period_index = i
        elif i == target_index:
            break

    return nearest_period_index


@torch.no_grad()
def upsample_eval(tensors, pad_value=0, t_size=[400, 400]):
    batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(t_size)
    batched_imgs = tensors[0].new_full(batch_shape, pad_value)
    for img, pad_img in zip(tensors, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs


def split_sentences(token_list):
    assert len(token_list) > 75
    
    splited_sentences = []
    while len(token_list) > 75:
        s_end_idx = find_nearest_period_index(token_list)
        if s_end_idx is None:
            splited_sentences.append(token_list[:75])
            token_list = token_list[75:]
        else:
            splited_sentences.append(token_list[:s_end_idx+1])
            token_list = token_list[s_end_idx+1:]
    if len(token_list) != 0:
        splited_sentences.append(token_list)
    return splited_sentences


def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def compute_IoU(masks, target):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection / union


def calculate_recall(pred_mask, gt_mask):
    assert pred_mask.shape == gt_mask.shape
    tp = (pred_mask * gt_mask).sum()
    fn = ((1 - pred_mask) * gt_mask).sum()
    recall = tp / (tp + fn + 1e-8)
    return recall


def calculate_f_score(pred_mask, gt_mask, beta=1.0):
    assert pred_mask.shape == gt_mask.shape
    tp = (pred_mask * gt_mask).sum()
    fp = (pred_mask * (1 - gt_mask)).sum()
    fn = ((1 - pred_mask) * gt_mask).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-8)
    return f_score


def load_cross_attention(tokenizer, splited_tokens, pred_file):
    cross_attention = []
    tokenizer.model_max_length = 230
    for i in range(len(splited_tokens)):
        cur_p_cross_attn = torch.load(pred_file)
        num_tokens = len(splited_tokens[i])
        selected_cross_attn = cur_p_cross_attn[..., 1:num_tokens+1]
        cross_attention.append(selected_cross_attn)
    cross_attention = torch.concat(cross_attention, dim=-1).permute(2, 0, 1)
    return cross_attention


def aggregate_cross_attention(ldm_stable, tokens, cross_attn, selected_nouns_clip_idx, video, frame_id, noun_idx):
    if not osp.exists(f'./outputs/scores'):
        os.mkdir(f'./outputs/scores')
    if not osp.exists(f'./outputs/scores/{video}_{frame_id}_{noun_idx}.pt'):
        noun_text_embeddings = []
        for n in tokens:
            text_input = ldm_stable.tokenizer([n], padding="max_length", max_length=77, truncation=True, return_tensors='pt')
            text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))
            noun_text_embeddings.append(text_embeddings[1])
        noun_text_embeddings = torch.concat(noun_text_embeddings)
        scores = noun_text_embeddings @ noun_text_embeddings.T
        scores = scores - scores.min()
        scores = scores / scores.max()
        weighted_s = scores[-1].softmax(dim=-1)
        torch.save(weighted_s, f'./outputs/scores/{video}_{frame_id}_{noun_idx}.pt')
    else:
        weighted_s = torch.load(f'./outputs/scores/{video}_{frame_id}_{noun_idx}.pt')
    weighted_cross_attn = torch.zeros_like(cross_attn[0])
    for i in range(min(len(selected_nouns_clip_idx), len(weighted_s))):
        idx = selected_nouns_clip_idx[i]
        if idx < cross_attn.shape[0]:
            weighted_cross_attn += torch.tensor(weighted_s[i] * cross_attn[idx])
    return weighted_cross_attn


def self_enhanced_fun(self_attn, cross_attn_ori, res, densecrf=False, img=None, beta=0.4):
    if self_attn.numel() < cross_attn_ori.numel():
        self_attn = F.interpolate(
            self_attn.reshape(1, 1, self_attn.shape[0]**2, self_attn.shape[0]**2),
            size=(res**2, res**2),
            mode='bilinear'
        ).reshape(res, res, res, res)
    valid_points_y, valid_points_x = torch.where(cross_attn_ori > beta)
    avg_self_attn = torch.zeros_like(cross_attn_ori)
    for y, x in zip(valid_points_y, valid_points_x):
        tmp = self_attn[int(y), int(x)]
        avg_self_attn += tmp
    avg_self_attn = avg_self_attn - avg_self_attn.min()
    if avg_self_attn.max() > 0:
        avg_self_attn = avg_self_attn / avg_self_attn.max()

    return avg_self_attn


def sam_refine_mask(sam_proposal_masks, mask, beta=0.3, tao=0.5):
    device = mask.device  
    refine_masks = torch.zeros_like(mask).to(device)
    cur_pred = (mask > beta).float()
    cnt = 0 
    pseudo_part = (mask > beta).float().to(device)
    
    for t in range(len(sam_proposal_masks)):
        _foreground = sam_proposal_masks[t].float().to(device)
        if _foreground.dim() == 3:
            _foreground = _foreground[0]

        if _foreground.shape != pseudo_part.shape:
            _foreground = F.interpolate(
                _foreground.unsqueeze(0).unsqueeze(0),
                size=pseudo_part.shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        
        inter_1 = (_foreground * pseudo_part).sum() / (_foreground.sum() + 1e-9)
        inter_2 = (_foreground * pseudo_part).sum() / (pseudo_part.sum() + 1e-9)
        if inter_1 > tao or inter_2 > tao:
            refine_masks[_foreground.bool()] = 1
            cnt += 1
    
    if cnt == 0:
        refine_masks = cur_pred
    return refine_masks


def find_subsequence_indices(subsequence, sequence):
    n = len(subsequence)
    for i in range(len(sequence) - n + 1):
        if sequence[i:i+n] == subsequence:
            return list(range(i, i+n))
    return []


def get_image_path(video, frame_id, image_base_path='./datasets/ref-davis/valid/JPEGImages'):
    """获取图像路径"""
    for ext in ['.jpg', '.png', '.jpeg']:
        image_path = os.path.join(image_base_path, video, f"{frame_id}{ext}")
        if os.path.exists(image_path):
            return image_path
    return None


def evaluate_model(cfg, frame_data, prediction_base_path, ground_truth_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    scheduler = DDIMSchedulerDev(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False
    )
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", scheduler=scheduler
    ).to(device)
    
    tokenizer = ldm_stable.tokenizer 
    
    grounding_dino_model = get_grounding_dino_model(
        cfg.grounding_dino_config, 
        cfg.grounding_dino_ckpt, 
        device=device
    )
    sam_predictor = get_sam_predictor(
        cfg.sam_ckpt, 
        cfg.sam_model_type, 
        device=device
    )
    
    # 统计信息
    total_masks = 0
    replaced_masks = 0

    for frame in tqdm(frame_data):
        video = frame["video"]
        frame_id = frame["frame"]
        obj_expressions = frame["expressions"]

        # 获取图像路径
        image_path = get_image_path(video, frame_id)
        if image_path is None:
            print(f"[WARNING] Image not found for {video}/{frame_id}")
            continue

        for obj_id, caption in obj_expressions.items():

            pred_file = os.path.join(
                prediction_base_path,
                f"{video}/{obj_id}/cross_16/{frame_id}.pt"
            )
            self_attn_file = os.path.join(
                prediction_base_path,
                f"{video}/{obj_id}/self_{cfg.self_res}/{frame_id}.pt"
            )
            
            if not os.path.exists(pred_file):
                print(f"Missing file: {pred_file}")
                continue

            text_inputs = tokenizer(
                caption,
                padding="max_length",
                max_length=tokenizer.model_max_length, 
                truncation=True,
                return_tensors="pt"
            )
            input_ids = text_inputs.input_ids[0].tolist()
            
            doc = nlp(caption)

            target_phrases = [chunk.text for chunk in doc.noun_chunks]

            if not target_phrases:
                target_phrases = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]

            selected_nouns_clip_idx = [] 
            phrase_tokens = []           

            for phrase in target_phrases:
                phrase_input_ids = tokenizer(phrase, add_special_tokens=False).input_ids
                indices = find_subsequence_indices(phrase_input_ids, input_ids)
                
                if indices:
                    selected_nouns_clip_idx.append(indices)
                    decoded_tokens = [tokenizer.decode([idx]) for idx in indices]
                    phrase_tokens.append(decoded_tokens)
            
            if not selected_nouns_clip_idx:
                eos_idx = input_ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in input_ids else len(input_ids)
                indices = list(range(1, eos_idx))
                selected_nouns_clip_idx.append(indices)
                phrase_tokens.append([tokenizer.decode([idx]) for idx in indices])

            try:
                cross_attention = torch.load(pred_file)
                cross_attention = cross_attention.permute(2, 0, 1)
            except Exception as e:
                print(f"Error loading attention: {e}")
                continue

            self_attn = torch.load(self_attn_file)
            
            # 加载 SAM Mask
            sam_path = f'./outputs/sam_db_Grounded/{video}_{frame_id}_obj{obj_id}.pt'
            if os.path.exists(sam_path):
                sam_everything_masks = torch.load(sam_path)
                sam_everything_masks = [torch.tensor(m['segmentation']).float().to(device) for m in sam_everything_masks]
            else:
                sam_everything_masks = []

            self_attn = self_attn.reshape(cfg.self_res, cfg.self_res, cfg.self_res, cfg.self_res)
            inter_res = max(cfg.self_res, cfg.cross_res)
            
            predictions = torch.zeros((cfg.max_phrase_num, inter_res, inter_res))
            
            valid_phrase_count = len(selected_nouns_clip_idx)

            for j in range(valid_phrase_count):
                indices = selected_nouns_clip_idx[j]
                tokens_text = phrase_tokens[j]

                if len(indices) > 1:
                    current_tokens = [ldm_stable.tokenizer.decode([idx]).strip() for idx in indices]
                    weighted_attn = torch.zeros_like(cross_attention[0])
                    for idx in indices:
                        if idx < cross_attention.shape[0]:
                            weighted_attn += cross_attention[idx]
                    weighted_attn /= len(indices)
                else:
                    idx = indices[0]
                    if idx < cross_attention.shape[0]:
                        weighted_attn = cross_attention[idx]
                    else:
                        continue

                # 归一化
                weighted_attn = F.interpolate(
                    weighted_attn[None, None, ...], 
                    size=(inter_res, inter_res), 
                    mode='bilinear'
                )[0, 0]
                if weighted_attn.max() > 0:
                    weighted_attn = (weighted_attn - weighted_attn.min()) / weighted_attn.max()
                
                # Self-Attention 增强
                predictions[j] = self_enhanced_fun(self_attn, weighted_attn, cfg.self_res, img=None)

            gt_mask = load_ground_truth_mask(video, frame_id, ground_truth_path)
            h, w = gt_mask.shape
            
            if valid_phrase_count > 0:
                final_prediction_map = predictions[:valid_phrase_count].max(dim=0)[0]
            else:
                final_prediction_map = torch.zeros((inter_res, inter_res))

            final_prediction_map = F.interpolate(
                final_prediction_map[None, None, ...], 
                size=(h, w), 
                mode='bilinear'
            )[0, 0]
            
            if len(sam_everything_masks) > 0:
                final_mask = sam_refine_mask(
                    sam_everything_masks, 
                    final_prediction_map.to(device), 
                    cfg.alpha, 
                    cfg.tao
                )
            else:
                final_mask = (final_prediction_map.to(device) > cfg.alpha).float()

            total_masks += 1
            

            image_source = cv2.imread(image_path)
            image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
            
            final_mask, was_replaced = refine_mask_with_grounding_dino(
                final_mask=final_mask,
                image_path=image_path,
                image_source=image_source,
                caption=caption,
                grounding_dino_model=grounding_dino_model,
                sam_predictor=sam_predictor,
                box_threshold=cfg.box_threshold,
                text_threshold=cfg.text_threshold,
                intersection_threshold=cfg.intersection_threshold,
                area_ratio_threshold=cfg.area_ratio_threshold,
                device=device
            )
            
            if was_replaced:
                replaced_masks += 1

            final_mask = final_mask.cpu()
            
            save_dir = os.path.join('./outputs/pred_semantic', video, str(obj_id))
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{int(frame_id):05d}.pt"
            torch.save(final_mask, os.path.join(save_dir, filename))
            
            mask_np = final_mask.numpy()
            mask_uint8 = (mask_np * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_dir, f"{int(frame_id):05d}_vis.png"), heatmap)


def load_video_frames(data_path, split):
    video_frames = []
    with open(data_path) as f:
        data = json.load(f)
        videos = data["videos"]
        for video, video_data in videos.items():
            expressions_data = video_data["expressions"]
            frames = video_data["frames"]

            obj_id_to_exp = {}
            for exp_info in expressions_data.values():
                obj_id = exp_info["obj_id"]
                if obj_id not in obj_id_to_exp:
                    obj_id_to_exp[obj_id] = exp_info["exp"]

            for frame in frames:
                frame_data = {
                    "video": video,
                    "frame": frame,
                    "expressions": obj_id_to_exp,
                    "split": split
                }
                video_frames.append(frame_data)
    return video_frames


def main():
    prediction_base_path = './outputs/attn/'
    ground_truth_path = './datasets/ref-davis/valid/Annotations'
    cfg = parse_args()
    frame_data = load_video_frames("./datasets/ref-davis/meta_expressions/valid/meta_expressions.json", "valid")
    evaluate_model(cfg, frame_data, prediction_base_path, ground_truth_path)


if __name__ == '__main__':
    main()