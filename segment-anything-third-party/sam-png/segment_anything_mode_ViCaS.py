import numpy as np
import torch
import torch.distributed as dist
import os.path as osp
import cv2
import json
from tqdm import tqdm
import os
import sys
import re
import warnings
import pdb
warnings.filterwarnings("ignore")

sys.path.append('./GroundingDINO/groundingdino')
from groundingdino.util.inference import Model as GroundingDINOModel

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

val_json_path = "./datasets/ViCaS/splits/v1.0/val.json"
anno_dir = "./datasets/ViCaS/annotations/v1.0"
frames_root = "./datasets/ViCaS/frames"
save_root = "/data/username/sam_Grounded"

grounding_dino_config = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
grounding_dino_weights = "./GroundingDINO/groundingdino_swint_ogc.pth"
sam_checkpoint = "./segment-anything/sam_vit_h_4b8939.pth"

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return device, local_rank, world_size
    else:
        device = torch.device("cuda:0")
        return device, 0, 1

device, local_rank, world_size = setup_distributed()

gd_model = GroundingDINOModel(
    model_config_path=grounding_dino_config,
    model_checkpoint_path=grounding_dino_weights,
    device=device 
)

sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

CAPTION_PATTERN = re.compile(
    r"\[([^]]+)\]((?:<mask_\d+(?:\s*,\s*\d+)*>)+)"
)
MASK_TAG_PATTERN = re.compile(r"<mask_([0-9]+(?:\s*,\s*[0-9]+)*)>")

def get_strict_exclusive_phrases(caption, target_ids, _cache={}):
    cache_key = caption
    if cache_key not in _cache:
        matches = CAPTION_PATTERN.findall(caption)
        parsed = []
        for phrase_text, mask_tags_str in matches:
            ids_in_phrase = set()
            for id_group in MASK_TAG_PATTERN.findall(mask_tags_str):
                ids_in_phrase.update(
                    int(x.strip()) for x in id_group.split(',') if x.strip()
                )
            parsed.append((phrase_text.strip(), ids_in_phrase))
        _cache[cache_key] = parsed
    
    parsed = _cache[cache_key]
    target_set = set(int(x) for x in target_ids)
    
    valid_phrases = []
    for phrase_text, ids_in_phrase_set in parsed:
        if ids_in_phrase_set and ids_in_phrase_set.issubset(target_set):
            if phrase_text:
                valid_phrases.append(phrase_text)
            
    return list(set(valid_phrases))

def get_phrase_counts(caption):
    matches = CAPTION_PATTERN.findall(caption)
    counts = {}
    for phrase_text, mask_tags_str in matches:
        ids = []
        for id_group in MASK_TAG_PATTERN.findall(mask_tags_str):
            ids.extend(x.strip() for x in id_group.split(',') if x.strip())
        phrase = phrase_text.strip()
        counts[phrase] = len(ids)
    return counts

def precompute_video_prompts(caption, obj_refs):
    prompts_per_obj = {}
    for i, ref_item in enumerate(obj_refs):
        track_ids = ref_item.get("track_ids", [])
        prompts_per_obj[i] = get_strict_exclusive_phrases(caption, track_ids)
    return prompts_per_obj

with open(val_json_path, 'r') as f:
    val_ids = json.load(f)

process_list = val_ids

if local_rank == 0:
    print(f"Total videos to process: {len(process_list)}")

with torch.no_grad():
    for idx, vid in tqdm(enumerate(process_list), total=len(process_list), desc="Processing videos", disable=local_rank != 0):
        if idx % world_size != local_rank:
            continue

        vid_str = f"{int(vid):06d}"
        anno_path = osp.join(anno_dir, f"{vid_str}.json")
        
        if not osp.exists(anno_path):
            continue

        with open(anno_path, "r") as f:
            anno_data = json.load(f)

        caption = anno_data.get("caption_raw_en", "")
        obj_refs = anno_data.get("object_referrals", [])
        
        if not caption or not obj_refs:
            continue

        frame_dir = osp.join(frames_root, vid_str)
        if not osp.exists(frame_dir):
            continue

        frame_names = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(".jpg")])
        
        video_save_dir = osp.join(save_root, vid_str)
        os.makedirs(video_save_dir, exist_ok=True)

        prompts_per_obj = precompute_video_prompts(caption, obj_refs)
        phrase_id_counts = get_phrase_counts(caption)
        num_objs = len(obj_refs)

        for frame_idx, frame_name in enumerate(frame_names): 
            frame_path = osp.join(frame_dir, frame_name)
            frame_id_str = osp.splitext(frame_name)[0]
            # Keep the same frame key used by sam2_ViCaS.py and zero_shot_ViCaS.py.
            frame_save_dir = osp.join(video_save_dir, frame_id_str)
            
            objs_to_process = []
            for i in range(num_objs):
                objs_to_process.append(i)
            
            if not objs_to_process:
                continue
                
            os.makedirs(frame_save_dir, exist_ok=True)

            image = cv2.imread(frame_path)
            if image is None: 
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape

            all_prompts = set()
            obj_prompt_mapping = {}
            
            for i in objs_to_process:
                exclusive_prompts = prompts_per_obj[i]
                obj_prompt_mapping[i] = exclusive_prompts
                all_prompts.update(exclusive_prompts)

            BOXES_PER_PROMPT = 5 

            prompt_to_boxes = {}

            for single_prompt in all_prompts:
                # pdb.set_trace()
                try:
                    detections, _ = gd_model.predict_with_caption(
                        image=image,
                        caption=single_prompt, 
                        box_threshold=0.25,
                        text_threshold=0.25
                    )
                    
                    if len(detections.xyxy) > 0:
                        boxes = detections.xyxy
                        conf = detections.confidence
                        
                        if conf is not None and len(conf) > 0:
                            sorted_idx = np.argsort(conf)[::-1]
                            sorted_boxes = boxes[sorted_idx]
                            keep_boxes = sorted_boxes[:BOXES_PER_PROMPT]
                        else:
                            keep_boxes = boxes[:BOXES_PER_PROMPT]

                        prompt_to_boxes[single_prompt] = keep_boxes
                    else:
                        prompt_to_boxes[single_prompt] = np.empty((0, 4))

                except Exception as e:
                    print(f"Error processing prompt '{single_prompt}': {e}")
                    pass  

            group_boxes_collection = {}
            
            for i in objs_to_process:
                collected_boxes = []

                for prompt in obj_prompt_mapping[i]:
                    if prompt in prompt_to_boxes and len(prompt_to_boxes[prompt]) > 0:
                        collected_boxes.append(prompt_to_boxes[prompt])

                if collected_boxes:
                    group_boxes_collection[i] = np.vstack(collected_boxes)
                else:
                    group_boxes_collection[i] = None

            predictor.set_image(image)
            zero_mask = torch.zeros((1, h, w), dtype=torch.bool)

            for i in objs_to_process:
                save_path = osp.join(frame_save_dir, f"{i}.pt")
                boxes = group_boxes_collection.get(i)
                final_mask = None

                if boxes is not None and len(boxes) > 0:
                    try:
                        boxes_torch = torch.from_numpy(boxes).to(device)
                        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_torch, (h, w))

                        masks, _, _ = predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=transformed_boxes,
                            multimask_output=False
                        )
                        
                        if masks is not None and masks.shape[0] > 0:
                            final_mask = masks.squeeze(1).cpu()
                            
                    except Exception as e:
                        print(f"SAM error: {e}")
                        pass  

                if final_mask is None:
                    final_mask = zero_mask.clone()

                torch.save(final_mask, save_path)

        torch.cuda.empty_cache()

if local_rank == 0:
    print("Processing Done.")
