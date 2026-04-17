"""MoReS deep analysis: sweep hyperparams + probe intermediate layers.

Records per-step per-layer statistics to understand what the interventions
are doing to hidden representations and why certain configs work better.

Experiments:
  1. Layer position sweep:    f4+l5, all, l10, f8+l8, f4, l5
  2. Rank sweep:              1, 2, 4, 8
  3. Steer ratio sweep:       0.01, 0.05, 0.1, 0.5, 1.0
  4. Share vs independent:    shared, independent
  5. Projector training:      frozen, unfrozen

Each config: 500 train + 100 eval, with per-layer probing.

Usage:
    !cd /content/mmit && git pull && \
     PYTHONPATH=src python -u scripts/run_mores_deep_analysis.py

    # Run specific experiment only:
    !PYTHONPATH=src python -u scripts/run_mores_deep_analysis.py --exp positions
    !PYTHONPATH=src python -u scripts/run_mores_deep_analysis.py --exp rank
"""
import sys, os, time, math, gc, json, traceback, argparse, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

P = "\033[92m✓\033[0m"
FL = "\033[91m✗\033[0m"

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
TRAIN_DATASET = "HuggingFaceH4/llava-instruct-mix-vsft"
NUM_TRAIN = 500
NUM_EVAL = 100
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 1e-3
OUTPUT_ROOT = "/content/mmit_output/mores_analysis"
FORWARD_KEYS = {
    "input_ids", "attention_mask", "labels",
    "pixel_values", "image_sizes", "image_grid_thw",
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
    def isatty(self):
        return False
    def fileno(self):
        return self.streams[0].fileno()


# ═══════════════════════════════════════════════════════════════
# Layer probing: capture hidden state statistics
# ═══════════════════════════════════════════════════════════════

class LayerProbe:
    """Attaches hooks to record hidden state stats before/after MoReS intervention.

    For each hooked layer, records:
      - h_norm_before: mean L2 norm of hidden states before intervention
      - h_norm_after:  mean L2 norm after intervention
      - delta_norm:    mean L2 norm of the intervention delta (h_after - h_before)
      - delta_ratio:   delta_norm / h_norm_before (relative change)
      - cos_sim:       mean cosine similarity between h_before and h_after
      - vis_token_norm: mean norm of visual tokens specifically
      - txt_token_norm: mean norm of text tokens specifically
      - R_norm:        Frobenius norm of R matrix
      - W_norm:        Frobenius norm of W matrix
      - ortho_penalty: ||R R^T - I||^2
    """

    def __init__(self):
        self._hooks = []
        self._step_data = []      # list of per-step records
        self._current_step = {}   # accumulates layer data for current step
        self._layer_names = []
        self._enabled = False

    def attach(self, model, layer_list, target_indices, intervention_objs):
        """Attach probing hooks BEFORE MoReS hooks (so we see pre-intervention)."""
        self._layer_names = [f"layer_{i}" for i in target_indices]
        self._intervention_objs = intervention_objs

        # We need to capture h BEFORE the MoReS hook modifies it.
        # Use a pre-hook to store the input, and a post-hook to compare.
        for idx_pos, layer_idx in enumerate(target_indices):
            layer_name = f"layer_{layer_idx}"

            def make_pre_hook(lname):
                def hook(module, args):
                    if not self._enabled:
                        return
                    # args[0] is hidden_states for decoder layers
                    h = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
                    if lname not in self._current_step:
                        self._current_step[lname] = {}
                    self._current_step[lname]["h_before"] = h.detach()
                return hook

            def make_post_hook(lname):
                def hook(module, input, output):
                    if not self._enabled:
                        return
                    h_after = output[0] if isinstance(output, tuple) else output
                    if lname in self._current_step and "h_before" in self._current_step[lname]:
                        self._current_step[lname]["h_after"] = h_after.detach()
                return hook

            pre_h = layer_list[layer_idx].register_forward_pre_hook(make_pre_hook(layer_name))
            post_h = layer_list[layer_idx].register_forward_hook(make_post_hook(layer_name))
            self._hooks.extend([pre_h, post_h])

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def on_step_end(self, global_step, input_ids=None, image_token_id=None):
        """Process and store the accumulated layer data for this step."""
        if not self._current_step:
            return

        step_record = {"step": global_step}

        for lname, data in self._current_step.items():
            h_before = data.get("h_before")
            h_after = data.get("h_after")
            if h_before is None or h_after is None:
                continue

            B, S, D = h_before.shape
            delta = h_after - h_before

            # Basic norms
            h_norm_b = h_before.norm(dim=-1).mean().item()
            h_norm_a = h_after.norm(dim=-1).mean().item()
            delta_n = delta.norm(dim=-1).mean().item()
            delta_ratio = delta_n / max(h_norm_b, 1e-8)

            # Cosine similarity
            cos = F.cosine_similarity(
                h_before.reshape(-1, D), h_after.reshape(-1, D), dim=-1,
            ).mean().item()

            rec = {
                "h_norm_before": round(h_norm_b, 4),
                "h_norm_after": round(h_norm_a, 4),
                "delta_norm": round(delta_n, 6),
                "delta_ratio": round(delta_ratio, 6),
                "cos_sim": round(cos, 6),
            }

            # Visual vs text token breakdown
            if input_ids is not None and image_token_id is not None:
                vis_mask = (input_ids == image_token_id)  # [B, S]
                if vis_mask.any():
                    vis_h = h_before[vis_mask]
                    txt_h = h_before[~vis_mask]
                    rec["vis_token_norm"] = round(vis_h.norm(dim=-1).mean().item(), 4)
                    rec["txt_token_norm"] = round(txt_h.norm(dim=-1).mean().item(), 4) if txt_h.numel() > 0 else 0
                    # Delta only on visual tokens
                    vis_delta = delta[vis_mask]
                    rec["vis_delta_norm"] = round(vis_delta.norm(dim=-1).mean().item(), 6)
                    rec["n_vis_tokens"] = int(vis_mask.sum().item())
                    rec["n_txt_tokens"] = int((~vis_mask).sum().item())

            step_record[lname] = rec

        # Intervention matrix stats
        for i, iv in enumerate(self._intervention_objs):
            step_record[f"intervention_{i}"] = {
                "R_norm": round(iv.R.data.norm().item(), 4),
                "W_weight_norm": round(iv.W.weight.data.norm().item(), 6),
                "W_bias_norm": round(iv.W.bias.data.norm().item(), 6),
                "ortho_penalty": round(iv.orthogonal_penalty().item(), 6),
            }

        self._step_data.append(step_record)
        self._current_step = {}

    def get_data(self):
        return self._step_data

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ═══════════════════════════════════════════════════════════════
# Model / data loading
# ═══════════════════════════════════════════════════════════════

def load_model():
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoVLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    print(f"   Model loaded (bf16), GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB", flush=True)
    return model, processor


def load_data(processor):
    from mmit.data.adapters.hf_datasets import HFDatasetsAdapter
    from mmit.training.preprocessors import ChatTemplatePreprocessor
    adapter = HFDatasetsAdapter(dataset_name=TRAIN_DATASET, split="train")
    samples = []
    for i, s in enumerate(adapter):
        samples.append(s)
        if i >= NUM_TRAIN - 1:
            break
    preproc = ChatTemplatePreprocessor()
    processed = []
    for i, s in enumerate(samples):
        try:
            processed.append(preproc.tokenize(s, processor, max_length=2048))
        except Exception:
            pass
    print(f"   {P} {len(processed)} train samples ready", flush=True)
    return processed, preproc


# ═══════════════════════════════════════════════════════════════
# Train + probe
# ═══════════════════════════════════════════════════════════════

def train_with_probing(model, processor, method, loss_fn, processed, preproc,
                       config_name, probe_every=5):
    """Train MoReS and record layer probe data every N steps."""
    from mmit.training.methods.mores import _find_llm_layers, _detect_image_token_id

    model.gradient_checkpointing_enable()
    model.train()
    device = next(model.parameters()).device

    params = method.get_trainable_params(model)
    trainable_count = sum(p.numel() for pg in params for p in pg["params"])
    for pg in params:
        pg.setdefault("lr", LR)
    optimizer = AdamW(params, weight_decay=0.0)

    loader = DataLoader(processed, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=preproc.collate, drop_last=True)
    steps_per_epoch = len(loader) // GRAD_ACCUM
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * 0.03)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    log_interval = max(1, total_steps // 10)

    # Setup probe
    layer_list, num_layers = _find_llm_layers(model)
    image_token_id = _detect_image_token_id(model)
    target_indices = []
    for h in method._hooks:
        # Recover target layer indices from the method's hooks
        pass
    # Re-parse from config
    from mmit.training.methods.mores import _parse_positions, _get_hidden_dim
    positions = method._hooks  # We'll get indices from method internals

    probe = LayerProbe()
    # We need target_indices — re-derive them
    config = method.default_config()
    # Get from the info that was printed
    # Simpler: just hook ALL layers and record
    all_layer_indices = list(range(num_layers))

    # Only probe a subset to avoid OOM (every 4th layer + first/last)
    probe_indices = sorted(set([0, 1, 2, 3] +
                               list(range(0, num_layers, 4)) +
                               list(range(max(0, num_layers-5), num_layers))))
    probe.attach(model, layer_list, probe_indices, method._all_interventions)

    print(f"   Steps: {total_steps}, Params: {trainable_count:,}, Probing {len(probe_indices)} layers every {probe_every} steps", flush=True)

    global_step, total_loss = 0, 0.0
    loss_history = []
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        for step, batch in enumerate(loader):
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items() if k in FORWARD_KEYS}

            # Enable probing on selected steps
            should_probe = ((step + 1) % GRAD_ACCUM == 0 and
                           (global_step + 1) % probe_every == 0)
            if should_probe:
                probe.enable()

            outputs = model(**batch_gpu)
            loss, metrics = loss_fn.compute(model, batch_gpu, outputs)

            if should_probe:
                input_ids = batch_gpu.get("input_ids")
                probe.on_step_end(global_step + 1, input_ids, image_token_id)
                probe.disable()

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            loss = loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in params for p in pg["params"] if p.grad is not None], 1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                step_loss = loss.item() * GRAD_ACCUM
                total_loss += step_loss
                loss_history.append({
                    "step": global_step,
                    "loss": round(step_loss, 6),
                    **{k: round(v, 6) for k, v in metrics.items()},
                })

                if global_step % log_interval == 0 or global_step == 1:
                    avg = total_loss / global_step
                    elapsed = time.time() - t0
                    eta = elapsed / global_step * (total_steps - global_step)
                    extra = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    print(f"   [{global_step}/{total_steps}] loss={step_loss:.4f} avg={avg:.4f} {extra} eta={eta:.0f}s", flush=True)

    avg_loss = total_loss / max(1, global_step)
    train_time = time.time() - t0
    print(f"   {P} {global_step} steps, avg_loss={avg_loss:.4f}, time={train_time:.0f}s", flush=True)

    probe_data = probe.get_data()
    probe.remove_hooks()

    return {
        "config_name": config_name,
        "trainable_params": trainable_count,
        "total_steps": global_step,
        "avg_loss": round(avg_loss, 6),
        "train_time_s": round(train_time),
        "loss_history": loss_history,
        "probe_data": probe_data,
    }


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate(model, processor):
    from mmit.eval.methods.local_method import LocalMethod
    from mmit.data.types import EvalSample
    from datasets import load_dataset

    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    eval_method = LocalMethod(model, processor)
    results = {}

    for bench_name, hf_id, split in [
        ("POPE", "lmms-lab/POPE", "test"),
        ("VQAv2", "lmms-lab/VQAv2", "validation"),
    ]:
        print(f"   --- {bench_name} ---", flush=True)
        try:
            ds = load_dataset(hf_id, split=split, streaming=True)
            eval_samples = []
            for j, row in enumerate(ds):
                if j >= NUM_EVAL:
                    break
                eval_samples.append(row)
            print(f"   {len(eval_samples)} samples loaded", flush=True)

            correct, total = 0, 0
            eval_t0 = time.time()

            for j, s in enumerate(eval_samples):
                question = s.get("question", "Describe this image.")
                image = s.get("image")
                answers = s.get("answers", s.get("answer", ""))
                if isinstance(answers, list):
                    if answers and isinstance(answers[0], dict):
                        gt = answers[0].get("answer", str(answers[0]))
                    else:
                        gt = str(answers[0]) if answers else ""
                else:
                    gt = str(answers)

                if bench_name == "POPE":
                    prompt = question + " Please answer yes or no."
                else:
                    prompt = question + " Answer the question using a single word or phrase."

                metadata = {"_pil_image": image} if image else {}
                es = EvalSample(
                    id=str(j), image_path="<in_memory>" if image else "",
                    question=prompt, ground_truth=gt, metadata=metadata,
                )
                prepared = eval_method.prepare_eval_input(es)
                pred = eval_method.generate(prepared, max_new_tokens=32)

                pred_c = pred.strip().lower().rstrip(".")
                gt_c = gt.strip().lower().rstrip(".")
                if pred_c == gt_c or gt_c in pred_c.split():
                    correct += 1
                total += 1

                if (j + 1) % 20 == 0:
                    acc_so_far = correct / total * 100
                    elapsed = time.time() - eval_t0
                    eta = elapsed / (j + 1) * (len(eval_samples) - j - 1)
                    print(f"   {bench_name}: {j+1}/{len(eval_samples)} acc={acc_so_far:.1f}% eta={eta:.0f}s", flush=True)

            acc = correct / max(1, total) * 100
            results[bench_name] = {"accuracy": round(acc, 2), "correct": correct, "total": total}
            print(f"   {P} {bench_name}: {acc:.1f}% ({correct}/{total})", flush=True)

        except Exception as e:
            print(f"   {FL} {bench_name}: {e}", flush=True)
            traceback.print_exc()

    return results


# ═══════════════════════════════════════════════════════════════
# Run a single MoReS config
# ═══════════════════════════════════════════════════════════════

def run_one_config(config_name, mores_config, processed, preproc):
    """Load fresh model, train MoReS with given config, eval, return results."""
    print(f"\n{'='*60}", flush=True)
    print(f"  {config_name}", flush=True)
    print(f"  Config: {json.dumps({k:v for k,v in mores_config.items() if k != 'freeze_patterns'}, indent=None)}", flush=True)
    print(f"{'='*60}", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    model, processor = load_model()

    from mmit.training.methods.mores import MoReSMethod
    from mmit.training.losses import CEPlusOrthoLoss

    method = MoReSMethod()
    full_config = {**method.default_config(), **mores_config}
    model, info = method.prepare_model(model, processor, full_config)
    print(f"   {info}", flush=True)

    loss_fn = CEPlusOrthoLoss(ortho_weight=0.01)

    # Train with probing
    train_result = train_with_probing(
        model, processor, method, loss_fn, processed, preproc,
        config_name=config_name, probe_every=5,
    )

    # Eval
    print("   Evaluating...", flush=True)
    eval_result = evaluate(model, processor)

    train_result["eval"] = eval_result
    train_result["config"] = full_config

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"   {P} {config_name}: VQAv2={eval_result.get('VQAv2',{}).get('accuracy','-')} "
          f"POPE={eval_result.get('POPE',{}).get('accuracy','-')} "
          f"TextVQA={eval_result.get('TextVQA',{}).get('accuracy','-')}", flush=True)

    return train_result


# ═══════════════════════════════════════════════════════════════
# Experiment definitions
# ═══════════════════════════════════════════════════════════════

EXPERIMENTS = {
    # Compact: 6 configs, each answers one key question vs the baseline
    "core": {
        "description": "Core ablation: 6 configs covering the most important axes",
        "configs": [
            # Baseline (paper defaults)
            ("baseline",      {"intervention_rank": 1, "positions": "f4+l5",
                               "steer_ratio": 0.01, "steer_visual_only": True,
                               "share_weights": True, "train_projector": False}),
            # Q1: is rank=1 really optimal?
            ("rank_4",        {"intervention_rank": 4}),
            # Q2: steer all visual tokens vs top 1%?
            ("ratio_1.0",     {"steer_ratio": 1.0}),
            # Q3: intervene all 32 layers vs selected?
            ("all_layers",    {"positions": "all"}),
            # Q4: unfreeze projector helps?
            ("proj_unfrozen", {"train_projector": True}),
            # Q5: steer all tokens (text+visual) vs visual-only?
            ("all_tokens",    {"steer_visual_only": False}),
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="core",
                        help=f"Which experiment to run: {list(EXPERIMENTS.keys())} or 'all'")
    parser.add_argument("--config", type=str, default="",
                        help="Run a single config by name, e.g. --config baseline")
    args = parser.parse_args()

    if args.exp == "all":
        exp_names = list(EXPERIMENTS.keys())
    else:
        exp_names = [e.strip() for e in args.exp.split(",")]
        for e in exp_names:
            if e not in EXPERIMENTS:
                print(f"{FL} Unknown experiment: {e}. Available: {list(EXPERIMENTS.keys())}")
                sys.exit(1)

    t0 = time.time()
    print(f"MoReS Deep Analysis", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Experiments: {exp_names}", flush=True)
    print(f"Train: {NUM_TRAIN} | Eval: {NUM_EVAL}/benchmark", flush=True)
    print(f"GPU: {torch.cuda.get_device_name()}\n", flush=True)

    import mmit

    # Load data once
    print("Loading data...", flush=True)
    tmp_model, processor = load_model()
    processed, preproc = load_data(processor)
    del tmp_model
    gc.collect()
    torch.cuda.empty_cache()

    all_experiment_results = {}

    for exp_name in exp_names:
        exp = EXPERIMENTS[exp_name]
        print(f"\n\n{'#'*60}", flush=True)
        print(f"# EXPERIMENT: {exp_name}", flush=True)
        print(f"# {exp['description']}", flush=True)
        print(f"{'#'*60}", flush=True)

        exp_results = {}
        configs_to_run = exp["configs"]
        if args.config:
            configs_to_run = [(n, c) for n, c in configs_to_run if n == args.config]
            if not configs_to_run:
                avail = [n for n, _ in exp["configs"]]
                print(f"   {FL} Config '{args.config}' not found. Available: {avail}", flush=True)
                continue
        for config_name, config_overrides in configs_to_run:
            try:
                result = run_one_config(config_name, config_overrides, processed, preproc)
                exp_results[config_name] = result
            except Exception as e:
                print(f"   {FL} {config_name} FAILED: {e}", flush=True)
                traceback.print_exc()
                exp_results[config_name] = {"error": str(e)}

        all_experiment_results[exp_name] = exp_results

        # Print experiment summary table
        print(f"\n{'─'*80}", flush=True)
        print(f"  {exp_name} SUMMARY", flush=True)
        print(f"{'─'*80}", flush=True)
        print(f"{'Config':<20} {'Params':>8} {'Loss':>8} {'POPE':>8} {'VQAv2':>8} {'Time':>6}", flush=True)
        print(f"{'─'*70}", flush=True)
        for cname, res in exp_results.items():
            if "error" in res:
                print(f"{cname:<20} FAILED", flush=True)
                continue
            params = f"{res['trainable_params']/1e3:.0f}K" if res['trainable_params'] < 1e6 else f"{res['trainable_params']/1e6:.1f}M"
            loss = f"{res['avg_loss']:.4f}"
            ev = res.get("eval", {})
            pope = ev.get("POPE", {}).get("accuracy", "-")
            vqa = ev.get("VQAv2", {}).get("accuracy", "-")
            tt = f"{res['train_time_s']}s"
            print(f"{cname:<20} {params:>8} {loss:>8} {pope:>8} {vqa:>8} {tt:>6}", flush=True)

        # Save per-experiment results (including probe data)
        exp_path = os.path.join(OUTPUT_ROOT, f"{exp_name}_results.json")
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        with open(exp_path, "w") as f:
            json.dump(exp_results, f, indent=2, default=str)
        print(f"   Saved to {exp_path}", flush=True)

        # Backup to Drive
        try:
            import shutil
            drive_dir = "/content/drive/MyDrive/mmit_results/mores_analysis"
            if os.path.exists("/content/drive/MyDrive"):
                os.makedirs(drive_dir, exist_ok=True)
                shutil.copy2(exp_path, os.path.join(drive_dir, f"{exp_name}_results.json"))
        except Exception:
            pass

    # Final summary
    total_time = time.time() - t0
    print(f"\n\n{'='*80}", flush=True)
    print(f"  ALL EXPERIMENTS COMPLETE ({total_time/60:.0f} min)", flush=True)
    print(f"{'='*80}", flush=True)
    for exp_name, exp_results in all_experiment_results.items():
        print(f"\n  {exp_name}:", flush=True)
        for cname, res in exp_results.items():
            if "error" in res:
                print(f"    {cname}: FAILED", flush=True)
            else:
                ev = res.get("eval", {})
                print(f"    {cname}: VQAv2={ev.get('VQAv2',{}).get('accuracy','-')} "
                      f"POPE={ev.get('POPE',{}).get('accuracy','-')} "
                      f"TextVQA={ev.get('TextVQA',{}).get('accuracy','-')} "
                      f"loss={res['avg_loss']}", flush=True)

    # Save master results
    master_path = os.path.join(OUTPUT_ROOT, "all_results.json")
    # Strip probe_data from master (too large), keep in per-experiment files
    slim_results = {}
    for exp_name, exp_results in all_experiment_results.items():
        slim_results[exp_name] = {}
        for cname, res in exp_results.items():
            if "error" in res:
                slim_results[exp_name][cname] = res
            else:
                slim_results[exp_name][cname] = {
                    k: v for k, v in res.items() if k not in ("probe_data", "loss_history")
                }
    with open(master_path, "w") as f:
        json.dump(slim_results, f, indent=2, default=str)
    print(f"\nMaster results: {master_path}", flush=True)


if __name__ == "__main__":
    log_path = "/content/drive/MyDrive/mmit_results/mores_deep_analysis.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    try:
        main()
    except Exception as e:
        print(f"\n{FL} FATAL: {e}", flush=True)
        traceback.print_exc()
    finally:
        log_file.close()
