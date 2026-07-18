# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from scheduler_dev import DDIMSchedulerDev
import transformers
import json

from scipy.ndimage import label
from collections import defaultdict
import math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import pdb
import torch.nn.functional as F

global_top_heads = []
spatial_entropy_heads = []
processed_heads = set()

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    
    height = width = 256
    batch_size = len(prompt)
    
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)
  
    return image, latent

def get_global_top_heads():
    grouped_heads = {}

    for layer_id, head_id, score in global_top_heads:
        if (layer_id, head_id) not in grouped_heads:
            grouped_heads[(layer_id, head_id)] = []
        grouped_heads[(layer_id, head_id)].append(score)
    
    # 计算每组的平均得分
    averaged_heads = []
    for (layer_id, head_id), scores in grouped_heads.items():
        avg_score = sum(scores) / len(scores)  # 计算平均分
        averaged_heads.append((layer_id, head_id, avg_score))
    
    # averaged_heads.sort(key=lambda x: x[2], reverse=True)
    
    file_path = "top_heads.txt"
    with open(file_path, "a") as file:
        for item in averaged_heads:
            file.write("\t".join(map(str, item)) + "\n")
    return averaged_heads

def get_spatial_entropy_heads():
    grouped_heads = {}

    for layer_id, head_id, score in spatial_entropy_heads:
        if (layer_id, head_id) not in grouped_heads:
            grouped_heads[(layer_id, head_id)] = []
        grouped_heads[(layer_id, head_id)].append(score)
    
    # 计算每组的平均得分
    averaged_heads = []
    for (layer_id, head_id), scores in grouped_heads.items():
        avg_score = sum(scores) / len(scores)  # 计算平均分
        averaged_heads.append((layer_id, head_id, avg_score))
    
    averaged_heads.sort(key=lambda x: x[2], reverse=False)

    top_10 = averaged_heads[:5]
    save_attention_heads("attention_heads.txt", top_10)
    return top_10

def save_attention_heads(file_path, top_10):
    # 读取已有数据
    head_counts = defaultdict(int)
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    head_counts[parts[0]] = int(parts[1])
    except FileNotFoundError:
        pass  # 如果文件不存在，则跳过读取

    # 统计新数据
    for head in top_10:
        key = f"{head[0]},{head[1]}"
        head_counts[key] += 1

    # 按出现次数排序
    sorted_heads = sorted(head_counts.items(), key=lambda x: x[1], reverse=True)

    # 追加写入文件
    with open(file_path, 'w') as f:
        for head, count in sorted_heads:
            f.write(f"{head} {count}\n")

def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data   

def find_nearest_period_index(word_list):
    target_index = 74
    nearest_period_index = None

    for i, word in enumerate(word_list[:75]):
        if word == '.':
            nearest_period_index = i
        elif i == target_index:
            break

    return nearest_period_index

def split_sentences(token_list):
    assert len(token_list)>75
    
    splited_sentences = []
    while len(token_list)>75:
        s_end_idx = find_nearest_period_index(token_list)
        if s_end_idx is None:
            splited_sentences.append(token_list[:75])
            token_list = token_list[75:]
        else:
            splited_sentences.append(token_list[:s_end_idx+1])
            token_list = token_list[s_end_idx+1:]
    if len(token_list)!=0:
        splited_sentences.append(token_list)
    return splited_sentences

panoptic_narrative_grounding = load_json('./ppmn_narr_list.json')
panoptic_narrative_grounding = [
    ln
    for ln in panoptic_narrative_grounding
    if (
        torch.tensor([item for sublist in ln["labels"] 
            for item in sublist])
        != -2
    ).any()
]

# scheduler = DDIMSchedulerDev(beta_start=0.00085,
#                                 beta_end=0.012,
#                                 beta_schedule="scaled_linear",
#                                 clip_sample=False,
#                                 set_alpha_to_one=False)
# device='cpu'
# ldm_stable =  StableDiffusionPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(device)
# tokenizer = ldm_stable.tokenizer
# bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')


def register_attention_control(model, controller):

    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            
        #     localized_narrative = panoptic_narrative_grounding[id]
        #     caption = localized_narrative['caption']
        #     noun_vector = localized_narrative['noun_vector']
        #     max_sequence_length = 230
        #     if len(noun_vector) > (max_sequence_length - 2):
        #         noun_vector_padding = \
        #                 noun_vector[:(max_sequence_length - 2)]
        #     elif len(noun_vector) < (max_sequence_length - 2): 
        #         noun_vector_padding = \
        #             noun_vector + [0] * (max_sequence_length - \
        #                 2 - len(noun_vector))
        #     noun_vector_padding = [0] + noun_vector_padding + [0]
        #     noun_vector_padding = torch.tensor(noun_vector_padding).long()

        #     bert_token_ids = torch.tensor(bert_tokenizer(caption,max_length=230)['input_ids'])
        #     bert_token_ids = bert_token_ids.unsqueeze(0)         
        #     valid_noun_vector = noun_vector_padding[1:len(bert_token_ids[0])-1]
        #     valid_indices = torch.nonzero(valid_noun_vector).squeeze()
        #     words_list =  [bert_tokenizer.decode(n) for n in list(bert_token_ids.reshape(-1,1).numpy())][1:-1]
            
        #     phrase_list = []
        #     valid_phrase_idx =[]            
        #     cur_phrase = None
        #     valid_phrase_bert_token_ids = []
        #     cur_phrase_bert_token_ids = []
            
        #     k = 0 
        #     tokens_length = len(words_list)
                
        #     while k<tokens_length:
        #         if k<tokens_length-1:
        #             if cur_phrase is None:
        #                 cur_phrase = words_list[k]
        #             else:
        #                 cur_phrase = cur_phrase + ' '+ words_list[k]
        #             cur_phrase_bert_token_ids.append(bert_token_ids[:,1:-1][:,k].item())
        #         elif k==tokens_length-1:
        #             if valid_noun_vector[k].item()!=valid_noun_vector[k-1].item():
        #                 phrase_list.append(words_list[-1])
        #                 valid_phrase_idx.append(valid_noun_vector[k].item())
        #                 valid_phrase_bert_token_ids.append(bert_token_ids[:,1:-1][:,k].item())
        #             else:
        #                 cur_phrase = cur_phrase + ' '+ words_list[k]
        #                 cur_phrase_bert_token_ids.append(bert_token_ids[:,1:-1][:,k].item())

        #                 phrase_list.append(cur_phrase)
        #                 valid_phrase_idx.append(valid_noun_vector[k].item())
        #                 valid_phrase_bert_token_ids.append(cur_phrase_bert_token_ids)
        #                 cur_phrase = None
        #                 cur_phrase_bert_token_ids = []
        #         if k< tokens_length-1 and valid_noun_vector[k]!=valid_noun_vector[k+1]:
        #             valid_phrase_idx.append(valid_noun_vector[k].item())
        #             phrase_list.append(cur_phrase)
        #             valid_phrase_bert_token_ids.append(cur_phrase_bert_token_ids)
        #             cur_phrase = None
        #             cur_phrase_bert_token_ids = []
        #         k+=1

        #         clip_phrase_list_idx = []
                
        #         for p in phrase_list:
        #             if p=="'s":
        #                 clip_phrase_list_idx.append([568])
        #             elif p=="' s":
        #                 clip_phrase_list_idx.append([568])
        #             elif "' s" in p:
        #                 clip_phrase_list_idx.append(tokenizer(p.replace("' s","'s"))['input_ids'][1:-1])
        #             else:
        #                 clip_phrase_list_idx.append(tokenizer(p)['input_ids'][1:-1])
            
        #         decoder = tokenizer.decode
        #         phrase_tokens = []
        #         selected_nouns_clip_idx = []
        #         cum = 0
        #         for i in range(len(clip_phrase_list_idx)):
        #             tmp=[]
        #             tokens = []
        #             if valid_phrase_idx[i]>0:
        #                 for j in range(len(clip_phrase_list_idx[i])):
        #                     tokens.append(decoder(clip_phrase_list_idx[i][j]))
        #                     tmp.append(cum)
        #                     cum+=1
        #                 selected_nouns_clip_idx.append(tmp)
        #                 phrase_tokens.append(tokens)
        #             else:
        #                 cum+=len(clip_phrase_list_idx[i])
                
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            layer_id = self.layer_id  
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            # raw_attn_scores = sim.detach() 

            # if is_cross and raw_attn_scores[1].shape[0] == 256 and valid_indices.numel() > 0:
                # attn_scores = raw_attn_scores
                # select_attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1)

                # selected_nouns_clip_idx_tensor = torch.tensor([idx for sublist in selected_nouns_clip_idx for idx in sublist])
                # selected_nouns_clip_idx_tensor = selected_nouns_clip_idx_tensor + 1
                # selected_nouns_clip_idx_tensor = torch.clamp(selected_nouns_clip_idx_tensor, 0, select_attn_scores.shape[-1] - 1)

                # select_attn_scores = select_attn_scores[:, :, selected_nouns_clip_idx_tensor]
                # sum_over_patches = select_attn_scores.sum(dim=1)  
                # S_img = torch.tensor([],device='cuda:0')
                # if valid_indices.numel() == 1:
                #     S_img = sum_over_patches
                # else:    
                #     S_img = sum_over_patches.mean(dim=1)
                # pdb.set_trace()
                # if S_img.dim() != 1 or S_img.shape[0] == 0:
                #     print(f"[Warning] S_img shape not expected: {S_img.shape}")
                # else:

                #     for i in range(8):
                #         val = S_img[i].item()
                #         if not math.isfinite(val):
                #             print(f"[Warning] S_img[{i}] = {val} is not finite, skipping.")
                #             continue
                #         global_top_heads.append((layer_id, i, val))
                # pdb.set_trace()
                # for i in range(8):
                #     # if S_img[i] >= 0.168 and (layer_id, i) not in processed_heads:
                #     if S_img[i] >= 0.168 :
                #         # processed_heads.add((layer_id, i))
                #         top_map = select_attn_scores[i]
                #         size_value = top_map[0].numel()  # 获取 top_map[1] 的元素个数
                #         top_map = top_map.view(size_value, 16, 16)

                #         mean_value = top_map.mean()
                #         binary_map = (top_map >= mean_value).int()

                #         spatial_entropy = 0
                #         for i in range(binary_map.size(0)):
                #             current_slice = top_map[i]
                #             entropy = find_connected_components_scipy(current_slice)
                #             spatial_entropy += entropy

                #         # 把计算的空间熵加入到结果列表中
                #         spatial_entropy_heads.append((layer_id, i, spatial_entropy))    

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # pdb.set_trace()
            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            # attn = controller(attn, is_cross, place_in_unet, self.layer_id)
            attn = controller(attn, is_cross, place_in_unet)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward
    
    def find_connected_components_scipy(slice_2d):
        labeled_array, num_features = label(slice_2d.cpu().numpy())
        component_sizes = np.bincount(labeled_array.ravel())[1:]  # 统计每个连通分量的大小

        total_elements = slice_2d.numel()
        if total_elements == 0 or len(component_sizes) == 0:
            return torch.tensor(0.0)

        probabilities = torch.tensor(component_sizes / total_elements, dtype=torch.float32)
        entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10))

        return entropy
    
    class DummyController:

        # def __call__(self, attn, is_cross=None, place_in_unet=None, layer_id=None):
        def __call__(self, attn, is_cross=None, place_in_unet=None):
            return attn

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            net_.layer_id = count
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count = register_recr(net[1], cross_att_count, "down")
        elif "up" in net[0]:
            cross_att_count = register_recr(net[1], cross_att_count, "up")
        elif "mid" in net[0]:
            cross_att_count = register_recr(net[1], cross_att_count, "mid")

    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words
