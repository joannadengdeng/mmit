"""Evaluate base model (no training) as baseline.

Usage:
    !cd /content/mmit && PYTHONPATH=src python scripts/run_baseline_eval.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

P = "\033[92m✓\033[0m"
F = "\033[91m✗\033[0m"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
NUM_EVAL = 500


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


def main():
    t0 = time.time()
    print(f"Baseline Eval (no training)")
    print(f"Model: {MODEL_ID}")
    print(f"Eval: {NUM_EVAL} samples/benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    import mmit
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM
    from mmit.eval.methods.local_method import LocalMethod
    from mmit.data.types import EvalSample
    from datasets import load_dataset

    # Import scoring
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from run_method import score_prediction, _extract_short_answer

    print("Loading model (bf16)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoVLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    print(f"{P} Model loaded, GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB\n")

    eval_method = LocalMethod(model, processor)
    results = {}

    for bench_name, hf_id, split in [
        ("VQAv2", "lmms-lab/VQAv2", "validation"),
        ("POPE", "lmms-lab/POPE", "test"),
        ("TextVQA", "lmms-lab/textvqa", "validation"),
    ]:
        print(f"--- {bench_name} ---")
        ds = load_dataset(hf_id, split=split, streaming=True)
        eval_samples = []
        for j, row in enumerate(ds):
            if j >= NUM_EVAL: break
            eval_samples.append(row)

        total_score, total = 0.0, 0
        eval_t0 = time.time()

        for j, s in enumerate(eval_samples):
            question = s.get("question", "")
            image = s.get("image")
            answers = s.get("answers", s.get("answer", ""))

            if bench_name == "POPE":
                prompt = question + " Please answer yes or no."
            else:
                prompt = question + " Answer the question using a single word or phrase."

            metadata = {"_pil_image": image} if image else {}
            es = EvalSample(id=str(j), image_path="<in_memory>" if image else "", question=prompt, ground_truth="", metadata=metadata)
            prepared = eval_method.prepare_eval_input(es)
            pred = eval_method.generate(prepared, max_new_tokens=32)

            score = score_prediction(pred, answers, bench_name)
            total_score += score
            total += 1

            if (j + 1) % 100 == 0:
                acc = total_score / total * 100
                elapsed = time.time() - eval_t0
                eta = elapsed / (j + 1) * (len(eval_samples) - j - 1)
                print(f"  {bench_name}: {j+1}/{len(eval_samples)} acc={acc:.1f}% eta={eta:.0f}s")

        acc = total_score / max(1, total) * 100
        results[bench_name] = {"accuracy": round(acc, 2), "total": total}
        print(f"  {P} {bench_name}: {acc:.1f}% ({total} samples)\n")

    total_time = time.time() - t0
    print(f"{'='*60}")
    print(f"  BASELINE (no training)")
    print(f"{'='*60}")
    for b, r in results.items():
        print(f"  {b}: {r['accuracy']}%")
    print(f"  Time: {total_time/60:.1f} min")
    print(f"{'='*60}")

    os.makedirs("/content/mmit_output", exist_ok=True)
    with open("/content/mmit_output/baseline_results.json", "w") as f:
        json.dump({"model": MODEL_ID, "method": "baseline", "eval": results}, f, indent=2)


if __name__ == "__main__":
    log_path = "/content/drive/MyDrive/mmit_results/baseline_eval.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    try:
        main()
    except Exception as e:
        print(f"\n{F} FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        log_file.close()
