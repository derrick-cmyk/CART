import numpy as np
import os
import os.path as osp
import random
import shutil
import torch
import torch.nn.functional as F
from collections import OrderedDict
from glob import glob
from skimage import io, color, measure
from torch import nn as nn
from torch.nn import init as init
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger, set_random_seed
from basicsr.utils.registry import MODEL_REGISTRY

# Full import of paint.utils functions that are used
from paint.utils import (
    colorize_label_image, dump_json, eval_json_folder_orig, evaluate,
    load_json, merge_color_line, np_2_labelpng, process_gt, read_img_2_np,
    read_line_2_np, read_seg_2_np, recolorize_gt, recolorize_seg,
)


@MODEL_REGISTRY.register()
class PBCModel(SRModel):

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt["path"].get("strict_load_g", True), "params_ema")
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        self.l_ce = build_loss(train_opt["l_ce"]).to(self.device)
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.data = data
        white_list = ["file_name"]
        for key in data.keys():
            if key not in white_list:
                self.data[key] = data[key].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.data)
        for k, v in self.data.items():
            self.data[k] = v[0]
        pred = {**self.data, **self.output}
        if pred["skip_train"]:
            return
        l_total = 0
        loss_dict = OrderedDict()
        loss = pred["loss"]
        loss_dict["acc"] = torch.tensor(pred["accuracy"]).to(self.device)
        loss_dict["area_acc"] = torch.tensor(pred["area_accuracy"]).to(self.device)
        loss_dict["valid_acc"] = torch.tensor(pred["valid_accuracy"]).to(self.device)
        loss_dict["loss_total"] = self.l_ce(loss)
        l_total += loss
        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.data)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.data)
        if not hasattr(self, "net_g_ema"):
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt["rank"] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        gt_folder_path = dataloader.dataset.opt["root"]
        with_metrics = self.opt["val"].get("metrics") is not None
        save_img = self.opt["val"].get("save_img", False)
        save_csv = self.opt["val"].get("save_csv", False)

        if with_metrics:
            if not hasattr(self, "metric_results"):
                self.metric_results = {metric: 0 for metric in self.opt["val"]["metrics"].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if hasattr(self, "net_g_ema"):
            model_inference = ModelInference(self.net_g_ema, dataloader)
        else:
            model_inference = ModelInference(self.net_g, dataloader)

        self.net_g.train()
        save_path = osp.join(self.opt["path"]["visualization"], str(current_iter), dataset_name)
        model_inference.inference_frame_by_frame(save_path, save_img)
        results = eval_json_folder_orig(save_path, gt_folder_path, "")
        if save_csv:
            csv_save_path = os.path.join(save_path, "metrics.csv")
            avg_dict, _, _ = evaluate(results, mode=dataset_name, save_path=csv_save_path, skip_first=True, stage="stage2")
        else:
            avg_dict, _, _ = evaluate(results, mode=dataset_name, skip_first=True, stage="stage2")

        self.metric_results["acc"] = avg_dict["acc"]
        self.metric_results["acc_thres"] = avg_dict["acc_thres"]
        self.metric_results["pix_acc"] = avg_dict["pix_acc"]
        self.metric_results["pix_acc_wobg"] = avg_dict["pix_acc_wobg"]
        self.metric_results["bmiou"] = avg_dict["bmiou"]
        self.metric_results["pix_bmiou"] = avg_dict["pix_bmiou"]

        if with_metrics:
            for metric in self.metric_results.keys():
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ {self.best_metric_results[dataset_name][metric]["iter"]} iter'
            log_str += "\n"
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{dataset_name}/{metric}", value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["line"] = self.data["line_ref"].detach().cpu()
        return out_dict


class ModelInference:
    def __init__(self, model, test_loader, seed=42):
        self._set_seed(seed)
        self.test_loader = test_loader
        self.model = model
        self.model.eval()

    def __del__(self):
        self._recover_seed()

    def _set_seed(self, seed):
        self.py_rng_state0 = random.getstate()
        self.np_rng_state0 = np.random.get_state()
        self.torch_rng_state0 = torch.get_rng_state()
        set_random_seed(seed)

    def _recover_seed(self):
        if hasattr(self, 'py_rng_state0'):
            random.setstate(self.py_rng_state0)
        if hasattr(self, 'np_rng_state0'):
            np.random.set_state(self.np_rng_state0)
        if hasattr(self, 'torch_rng_state0'):
            torch.set_rng_state(self.torch_rng_state0)

    def _swap_test_data(self, data):
        swapped = {}
        for k, v in data.items():
            if k.endswith("_ref"):
                target_key = k[:-4]
                if target_key in data:
                    swapped[k] = data[target_key]
                    swapped[target_key] = v
            elif k + "_ref" in data:
                continue
            else:
                swapped[k] = v
        return swapped

    def dis_data_to_cuda(self, data):
        device = next(self.model.parameters()).device
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data

    def inference_frame_by_frame(self, save_path, save_img=False):
        with torch.no_grad():
            self.model.eval()
            for test_data in tqdm(self.test_loader):
                line_root, name_str = osp.split(test_data["file_name"][0])
                character_root = osp.split(line_root)[0]
                prev_index = int(name_str) - 1
                prev_name_str = str(prev_index).zfill(len(name_str))
                ref_json_path = osp.join(character_root, "seg", prev_name_str + ".json")
                save_folder = osp.join(save_path, osp.split(character_root)[-1])

                if prev_index == 0:
                    os.makedirs(save_folder, exist_ok=True)
                    shutil.copy(ref_json_path, save_folder)
                    if save_img:
                        gt0_path = osp.join(character_root, "gt", prev_name_str + ".png")
                        shutil.copy(gt0_path, save_folder)

                color_dict = load_json(ref_json_path)
                json_save_path = osp.join(save_folder, name_str + ".json")

                match_tensor = self.model(self.dis_data_to_cuda(test_data))

                if "match_scores" not in match_tensor:
                    print(f"Warning: No valid segments for {name_str}, copying reference colors.")
                    target_seg = test_data["segment"]
                    num_segs = int(target_seg.max().item())
                    color_next_frame = {}
                    unmatch_color = [0] * len(list(color_dict.values())[0]) if color_dict else [0, 0, 0, 0]
                    for i in range(num_segs):
                        target_id = str(i + 1)
                        color_next_frame[target_id] = color_dict.get(target_id, unmatch_color)
                else:
                    match_scores = match_tensor["match_scores"].cpu().numpy()
                    color_next_frame = {}
                    unmatch_color = [0] * len(list(color_dict.values())[0])
                    for seg_idx, scores in enumerate(match_scores):
                        color_lookup = np.array([
                            (color_dict[str(ref_id + 1)] if str(ref_id + 1) in color_dict else unmatch_color)
                            for ref_id in range(len(scores))
                        ])
                        unique_colors = np.unique(color_lookup, axis=0)
                        accumulated_probs = [np.sum(scores[np.all(color_lookup == color, axis=1)]) for color in unique_colors]
                        color_next_frame[str(seg_idx + 1)] = unique_colors[np.argmax(accumulated_probs)].tolist()
                dump_json(color_next_frame, json_save_path)

                label_path = osp.join(character_root, "seg", name_str + ".png")
                img_save_path = json_save_path.replace(".json", ".png")
                colorize_label_image(label_path, json_save_path, img_save_path)

    def inference_multi_gt(self, save_path, keep_line=False):
        with torch.no_grad():
            self.model.eval()
            dataset = self.test_loader.dataset
            mode = dataset.opt.get('mode', 'forward')

            if mode == 'nearest':
                self._nearest_propagation(save_path, keep_line)
                characters = set()
                for test_data in tqdm(self.test_loader):
                    try:
                        self._process_single_batch(test_data, save_path, keep_line, characters)
                    except Exception as e:
                        line_root, name_str = osp.split(test_data["file_name"][0])
                        character_root, _ = osp.split(line_root)
                        _, character_name = osp.split(character_root)
                        print(f"  [NEAREST] Model inference FAILED for {name_str}: {e} – result from nearest GT maintained.")
                return

            pair_to_idx = {}
            print("Building pair lookup table from dataset...")
            for i in range(len(dataset)):
                sample = dataset[i]
                target = osp.splitext(osp.basename(
                    sample['file_name'][0] if isinstance(sample['file_name'], list) else sample['file_name']
                ))[0]
                ref = osp.splitext(osp.basename(
                    sample['file_name_ref'][0] if isinstance(sample['file_name_ref'], list) else sample['file_name_ref']
                ))[0]
                pair_to_idx[(target, ref)] = i
            print(f"Lookup table built with {len(pair_to_idx)} pairs.")

            dataset_root = dataset.opt['root']
            if not dataset.opt.get('multi_clip', False):
                clip_names = [osp.basename(osp.normpath(dataset_root))]
                clip_roots = [dataset_root]
            else:
                clip_names = sorted([d for d in os.listdir(dataset_root) if osp.isdir(osp.join(dataset_root, d))])
                clip_roots = [osp.join(dataset_root, d) for d in clip_names]

            characters = set()
            global_ds_idx = 0
            for character_name, character_root in zip(clip_names, clip_roots):
                line_root = osp.join(character_root, 'line')
                frame_paths = sorted(glob(osp.join(line_root, "*.png")))
                num_frames = len(frame_paths)

                gt_root = line_root.replace("line", "gt")
                gt_names = [osp.splitext(osp.basename(f))[0] for f in glob(osp.join(gt_root, "*.png"))]
                idx_to_name = {i: osp.splitext(osp.basename(p))[0] for i, p in enumerate(frame_paths)}
                gt_indices = [i for i, name in idx_to_name.items() if name in gt_names]

                if not gt_indices:
                    print(f"Skip {character_name}: No GT frames found.")
                    if num_frames > 1:
                        global_ds_idx += (num_frames - 1)
                    continue

                plan = []
                dists = {i: (float('inf'), None, None) for i in range(num_frames)}
                for g_idx in gt_indices:
                    dists[g_idx] = (0, g_idx, None)
                queue = list(gt_indices)
                visited = set(gt_indices)
                while queue:
                    curr_idx = queue.pop(0)
                    for neighbor_idx in [curr_idx - 1, curr_idx + 1]:
                        if 0 <= neighbor_idx < num_frames and neighbor_idx not in visited:
                            direction = 'forward' if neighbor_idx > curr_idx else 'backward'
                            if mode == 'forward' and direction != 'forward':
                                continue
                            if mode == 'backward' and direction != 'backward':
                                continue
                            visited.add(neighbor_idx)
                            dists[neighbor_idx] = (dists[curr_idx][0] + 1, curr_idx, direction)
                            queue.append(neighbor_idx)
                for dist in range(1, num_frames + 1):
                    for i in range(num_frames):
                        if dists[i][0] == dist:
                            ref_idx = dists[i][1]
                            direction = dists[i][2]
                            plan.append({'target': i, 'ref': ref_idx, 'mode': direction, 'dist': dist})

                print(f"\nPropagation plan for {character_name} (mode={mode}):")
                for step in plan:
                    print(f"  {idx_to_name[step['target']]} <- {idx_to_name[step['ref']]} (dist {step['dist']}, mode {step['mode']})")

                save_folder = osp.join(save_path, character_name)
                os.makedirs(save_folder, exist_ok=True)
                for g_idx in gt_indices:
                    src_json = osp.join(character_root, "seg", idx_to_name[g_idx] + ".json")
                    if os.path.exists(src_json):
                        dst = osp.join(save_folder, idx_to_name[g_idx] + ".json")
                        if osp.abspath(src_json) != osp.abspath(dst):
                            shutil.copy(src_json, dst)

                line_thr = dataset.opt.get('line_thr', 50)

                for step in tqdm(plan, desc=f"Coloring {character_name}"):
                    target_name = idx_to_name[step['target']]
                    ref_name = idx_to_name[step['ref']]
                    direction = step['mode']

                    if direction == 'forward':
                        pair_key = (target_name, ref_name)
                        ds_idx = pair_to_idx.get(pair_key)
                        if ds_idx is None:
                            print(f"  {target_name}: forward pair not found – skipping.")
                            continue
                        test_data = dataset[ds_idx]
                        for k in test_data.keys():
                            v = test_data[k]
                            if isinstance(v, torch.Tensor):
                                test_data[k] = v.unsqueeze(0)
                            elif k in ["keypoints", "keypoints_ref"]:
                                test_data[k] = torch.tensor(v).unsqueeze(0)
                            elif not isinstance(v, list):
                                test_data[k] = [v]
                        try:
                            self._process_single_batch(test_data, save_path, keep_line, characters)
                            print(f"  [FORWARD] Model inference SUCCESS for {target_name}")
                        except Exception as e:
                            print(f"  [FORWARD] Model inference FAILED for {target_name}: {e} – falling back to copy")
                            ref_json = osp.join(save_folder, ref_name + ".json")
                            self._copy_colors_from_json(
                                ref_json,
                                osp.join(save_folder, target_name + ".json"),
                                osp.join(character_root, "seg", target_name + ".png"),
                                osp.join(save_folder, target_name + ".png")
                            )
                    else:  # backward
                        ref_json_path = osp.join(save_folder, ref_name + ".json")
                        if not osp.exists(ref_json_path):
                            print(f"  {target_name}: reference JSON missing – falling back to nearest GT.")
                            nearest_gt = min(gt_indices, key=lambda g: abs(g - step['target']))
                            ref_json_path = osp.join(save_folder, idx_to_name[nearest_gt] + ".json")
                            if not osp.exists(ref_json_path):
                                shutil.copy(osp.join(character_root, "seg", idx_to_name[nearest_gt] + ".json"), ref_json_path)
                            self._copy_colors_from_json(
                                ref_json_path,
                                osp.join(save_folder, target_name + ".json"),
                                osp.join(character_root, "seg", target_name + ".png"),
                                osp.join(save_folder, target_name + ".png")
                            )
                            continue

                        # 1. Render a temporary coloured image (RGB) of the reference frame
                        ref_line_path = osp.join(line_root, ref_name + ".png")
                        ref_label_path = osp.join(character_root, "seg", ref_name + ".png")
                        tmp_colored_png = osp.join(save_folder, f"_tmp_{ref_name}_colored.png")
                        if not osp.exists(tmp_colored_png):
                            colorize_label_image(ref_label_path, ref_json_path, tmp_colored_png)

                        # 2. Read target and reference line images (3-channel RGB as expected by the model)
                        target_line_np = read_line_2_np(
                            osp.join(line_root, target_name + ".png"),
                            channel=3,
                            line_thr=line_thr,
                            treat_as_final=dataset.opt.get('treat_as_final', False)
                        )  # (H, W, 3) RGB
                        ref_line_np = read_line_2_np(
                            ref_line_path,
                            channel=3,
                            line_thr=line_thr,
                            treat_as_final=dataset.opt.get('treat_as_final', False)
                        )  # (H, W, 3) RGB

                        # 3. Read segmentations
                        target_seg = read_seg_2_np(osp.join(character_root, "seg", target_name + ".png"))
                        target_seg = np.atleast_2d(target_seg)
                        if target_seg.ndim == 3:
                            target_seg = target_seg[..., 0]

                        ref_seg = read_seg_2_np(ref_label_path)
                        ref_seg = np.atleast_2d(ref_seg)
                        if ref_seg.ndim == 3:
                            ref_seg = ref_seg[..., 0]

                        # 4. Read the reference colored image
                        ref_colored = io.imread(tmp_colored_png)[..., :3]  # RGB, drop alpha

                        # 5. Convert to tensors
                        target_line_t = torch.from_numpy(target_line_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                        ref_line_t = torch.from_numpy(ref_line_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)      # (1, 3, H, W)
                        target_seg_t = torch.from_numpy(target_seg).long().unsqueeze(0).unsqueeze(1)                              # (1, 1, H, W)
                        ref_seg_t = torch.from_numpy(ref_seg).long().unsqueeze(0).unsqueeze(1)                                    # (1, 1, H, W)
                        ref_colored_float = ref_colored.astype(np.float32) / 255.0
                        ref_colored_t = torch.from_numpy(ref_colored_float).permute(2, 0, 1).unsqueeze(0)                        # (1, 3, H, W)

                        # 6. Resize all tensors to the target resolution
                        target_size = dataset.opt.get('raft_res', 320)

                        def resize_tensor(tensor, size, mode='bilinear'):
                            if tensor.size(-2) == size and tensor.size(-1) == size:
                                return tensor
                            return F.interpolate(tensor.float(), size=(size, size),
                                                 mode=mode, align_corners=False if mode != 'nearest' else None)

                        target_line_t = resize_tensor(target_line_t, target_size, mode='bilinear')
                        ref_line_t = resize_tensor(ref_line_t, target_size, mode='bilinear')
                        ref_colored_t = resize_tensor(ref_colored_t, target_size, mode='bilinear')
                        target_seg_t = resize_tensor(target_seg_t.float(), target_size, mode='nearest').long()
                        ref_seg_t = resize_tensor(ref_seg_t.float(), target_size, mode='nearest').long()

                        h_orig, w_orig = target_seg.shape
                        scale_h, scale_w = target_size / h_orig, target_size / w_orig

                        def get_bboxes(seg_np, s_h, s_w):
                            # Extract bounding boxes for segments (1-based indexing)
                            props = measure.regionprops(seg_np.astype(np.int32))
                            bboxes = []
                            # Sort props by label to ensure they match segment IDs (1, 2, 3...)
                            props = sorted(props, key=lambda x: x.label)
                            for p in props:
                                # regionprops bbox: (min_row, min_col, max_row, max_col)
                                y1, x1, y2, x2 = p.bbox
                                bboxes.append([float(x1) * s_w, float(x2) * s_w, float(y1) * s_h, float(y2) * s_h])
                            if not bboxes:
                                return torch.empty(1, 0, 4, dtype=torch.float32)
                            return torch.tensor(bboxes, dtype=torch.float32).unsqueeze(0)

                        target_kp = get_bboxes(target_seg, scale_h, scale_w)
                        ref_kp = get_bboxes(ref_seg, scale_h, scale_w)

                        fake_test_data = {
                            'line': target_line_t,            # (1, 3, H, W)
                            'line_ref': ref_line_t,           # (1, 3, H, W)
                            'recolorized_img': ref_colored_t, # (1, 3, H, W)
                            'segment': target_seg_t,          # (1, 1, H, W)
                            'segment_ref': ref_seg_t,         # (1, 1, H, W)   <-- reference segmentation
                            'file_name': [osp.join(line_root, target_name)],
                            'file_name_ref': [osp.join(line_root, ref_name)],
                            'keypoints': target_kp,
                            'keypoints_ref': ref_kp,
                        }

                        try:
                            self._process_single_batch(fake_test_data, save_path, keep_line, characters)
                            print(f"  [BACKWARD] Model inference SUCCESS for {target_name}")
                        except Exception as e:
                            print(f"  [BACKWARD] Model inference FAILED for {target_name}: {e} – falling back to copy")
                            self._copy_colors_from_json(
                                ref_json_path,
                                osp.join(save_folder, target_name + ".json"),
                                osp.join(character_root, "seg", target_name + ".png"),
                                osp.join(save_folder, target_name + ".png")
                            )
                        finally:
                            if osp.exists(tmp_colored_png):
                                try:
                                    os.remove(tmp_colored_png)
                                except OSError:
                                    pass

    def _copy_colors_from_json(self, src_json, dst_json, label_path, img_save_path):
        if not osp.exists(src_json):
            print(f"    Warning: source JSON {src_json} not found, creating empty JSON.")
            dump_json({}, dst_json)
            return
        src_colors = load_json(src_json)
        if not osp.exists(label_path):
            print(f"    Warning: label image {label_path} not found; cannot generate colored image.")
            dump_json(src_colors, dst_json)
            return
        label_img = io.imread(label_path)
        max_id = int(np.max(label_img))
        num_segments = max_id + 1
        ref_colors_list = list(src_colors.values())
        if len(ref_colors_list) == 0:
            default_color = [0, 0, 0, 0]
        else:
            default_color = ref_colors_list[0]
        new_colors = {}
        for seg_id in range(num_segments):
            str_id = str(seg_id + 1)
            if str_id in src_colors:
                new_colors[str_id] = src_colors[str_id]
            else:
                new_colors[str_id] = default_color
        dump_json(new_colors, dst_json)
        colorize_label_image(label_path, dst_json, img_save_path)
        print(f"    Colors copied (with segment matching) to {dst_json}")

    def _nearest_propagation(self, save_path, keep_line=False):
        dataset = self.test_loader.dataset
        dataset_root = dataset.opt['root']
        if not dataset.opt.get('multi_clip', False):
            clip_names = [osp.basename(osp.normpath(dataset_root))]
            clip_roots = [dataset_root]
        else:
            clip_names = sorted([d for d in os.listdir(dataset_root) if osp.isdir(osp.join(dataset_root, d))])
            clip_roots = [osp.join(dataset_root, d) for d in clip_names]
        for character_name, character_root in zip(clip_names, clip_roots):
            line_root = osp.join(character_root, 'line')
            frame_paths = sorted(glob(osp.join(line_root, "*.png")))
            num_frames = len(frame_paths)
            gt_root = line_root.replace("line", "gt")
            gt_names = [osp.splitext(osp.basename(f))[0] for f in glob(osp.join(gt_root, "*.png"))]
            idx_to_name = {i: osp.splitext(osp.basename(p))[0] for i, p in enumerate(frame_paths)}
            gt_indices = [i for i, name in idx_to_name.items() if name in gt_names]
            if not gt_indices:
                print(f"Skip {character_name}: No GT frames found.")
                continue
            save_folder = osp.join(save_path, character_name)
            os.makedirs(save_folder, exist_ok=True)
            if keep_line:
                save_folder_keepline = osp.join(save_path, character_name + '_keepline')
                os.makedirs(save_folder_keepline, exist_ok=True)
            for g_idx in gt_indices:
                gt_name = idx_to_name[g_idx]
                src_json = osp.join(character_root, "seg", gt_name + ".json")
                dst_json = osp.join(save_folder, gt_name + ".json")
                if os.path.exists(src_json):
                    shutil.copy(src_json, dst_json)
                if keep_line:
                    line_path = osp.join(line_root, gt_name + ".png")
                    gt_path = osp.join(gt_root, gt_name + ".png")
                    merged_path = osp.join(save_folder_keepline, gt_name + ".png")
                    if os.path.exists(line_path) and os.path.exists(gt_path):
                        merge_color_line(line_path, gt_path, merged_path)
            for i in tqdm(range(num_frames), desc=f"Nearest propagation for {character_name}"):
                if i in gt_indices:
                    continue
                target_name = idx_to_name[i]
                nearest_gt = min(gt_indices, key=lambda g: abs(g - i))
                gt_name = idx_to_name[nearest_gt]
                gt_json = osp.join(save_folder, gt_name + ".json")
                if not os.path.exists(gt_json):
                    shutil.copy(osp.join(character_root, "seg", gt_name + ".json"), gt_json)
                self._copy_colors_from_json(
                    gt_json,
                    osp.join(save_folder, target_name + ".json"),
                    osp.join(character_root, "seg", target_name + ".png"),
                    osp.join(save_folder, target_name + ".png")
                )
                if keep_line:
                    line_path = osp.join(line_root, target_name + ".png")
                    merged_path = osp.join(save_folder_keepline, target_name + ".png")
                    if os.path.exists(line_path):
                        colorized_img_path = osp.join(save_folder, target_name + ".png")
                        if os.path.exists(colorized_img_path):
                            merge_color_line(line_path, colorized_img_path, merged_path)
                        else:
                            shutil.copy(line_path, merged_path)

    def _process_single_batch(self, test_data, save_path, keep_line, characters, fallback_gt_json=None):
        line_root, name_str = osp.split(test_data["file_name"][0])
        character_root, _ = osp.split(line_root)
        _, character_name = osp.split(character_root)
        save_folder = osp.join(save_path, character_name)
        save_folder_keepline = osp.join(save_path, character_name+'_keepline')

        if character_name not in characters:
            characters.add(character_name)
            os.makedirs(save_folder, exist_ok=True)
            if keep_line: os.makedirs(save_folder_keepline, exist_ok=True)
            gt_root = line_root.replace("line", "gt")
            for gt_path in glob(osp.join(gt_root, "*.png")):
                json_path = gt_path.replace("gt", "seg").replace("png", "json")
                dst_gt = osp.join(save_folder, osp.basename(gt_path))
                dst_json = osp.join(save_folder, osp.basename(json_path))
                if os.path.exists(gt_path) and osp.abspath(gt_path) != osp.abspath(dst_gt):
                    shutil.copy(gt_path, save_folder)
                if os.path.exists(json_path) and osp.abspath(json_path) != osp.abspath(dst_json):
                    shutil.copy(json_path, save_folder)
                if keep_line:
                    line_path = gt_path.replace("gt", "line")
                    merged_img_save_path = osp.join(save_folder_keepline, osp.basename(gt_path))
                    merge_color_line(line_path, gt_path, merged_img_save_path)

        _, name_str_ref = osp.split(test_data["file_name_ref"][0])
        json_path_ref = osp.join(save_folder, name_str_ref + ".json")
        if not os.path.exists(json_path_ref):
            raise FileNotFoundError(
                f"Reference data missing for target frame '{name_str}'. Expected '{json_path_ref}' "
                f"(from reference '{name_str_ref}')."
            )
        color_dict = load_json(json_path_ref)
        json_save_path = osp.join(save_folder, name_str + ".json")

        match_tensor = self.model(self.dis_data_to_cuda(test_data))

        if match_tensor.get("skip_train", False) and "match_scores" not in match_tensor:
            print(f"Warning: No valid segments for {name_str}, copying reference colors.")
            target_seg = test_data["segment"]
            num_segs = int(target_seg.max().item())
            color_next_frame = {}
            ref_colors = list(color_dict.values())
            unmatch_color = [0] * len(ref_colors[0]) if ref_colors else [0, 0, 0, 0]
            for i in range(num_segs):
                target_id = str(i + 1)
                color_next_frame[target_id] = color_dict.get(target_id, unmatch_color)
            dump_json(color_next_frame, json_save_path)
        else:
            match_scores = match_tensor["match_scores"].cpu().numpy()
            color_next_frame = {}
            unmatch_color = [0] * len(list(color_dict.values())[0])
            for seg_idx, scores in enumerate(match_scores):
                color_lookup = np.array([
                    (color_dict[str(ref_id + 1)] if str(ref_id + 1) in color_dict else unmatch_color)
                    for ref_id in range(len(scores))
                ])
                unique_colors = np.unique(color_lookup, axis=0)
                accumulated_probs = [np.sum(scores[np.all(color_lookup == color, axis=1)]) for color in unique_colors]
                color_next_frame[str(seg_idx + 1)] = unique_colors[np.argmax(accumulated_probs)].tolist()
            dump_json(color_next_frame, json_save_path)

        label_path = osp.join(character_root, "seg", name_str + ".png")
        img_save_path = osp.join(save_folder, name_str + ".png")
        colorize_label_image(label_path, json_save_path, img_save_path)

        if keep_line:
            line_path = osp.join(character_root, "line", name_str + ".png")
            if os.path.exists(line_path):
                merge_color_line(line_path, img_save_path, osp.join(save_folder_keepline, name_str + ".png"))