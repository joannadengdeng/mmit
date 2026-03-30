"""Debug: check what MoReS model is actually outputting during eval.

Shows 10 sample predictions vs ground truth to diagnose scoring issues.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLM

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

def main():
    import mmit
    from mmit.eval.methods.local_method import LocalMethod
    from mmit.data.types import EvalSample
    from datasets import load_dataset

    # Load base model (no training, just check baseline)
    print("Loading base model (bf16)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoVLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    method = LocalMethod(model, processor)

    # Test VQAv2
    print("\n=== VQAv2 (base model, no training) ===")
    ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    for i, s in enumerate(ds):
        if i >= 5: break
        question = s.get("question", "")
        image = s.get("image")
        answers = s.get("answers", [])
        gt = answers[0].get("answer", "") if answers and isinstance(answers[0], dict) else str(answers[0]) if answers else ""

        prompt = question + " Answer the question using a single word or phrase."
        metadata = {"_pil_image": image} if image else {}
        es = EvalSample(id=str(i), image_path="<in_memory>" if image else "", question=prompt, ground_truth=gt, metadata=metadata)
        prepared = method.prepare_eval_input(es)
        pred = method.generate(prepared, max_new_tokens=32)
        print(f"  Q: {question}")
        print(f"  GT: {gt}")
        print(f"  Pred: [{pred}]")
        print()

    # Test POPE
    print("=== POPE (base model, no training) ===")
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    for i, s in enumerate(ds):
        if i >= 5: break
        question = s.get("question", "")
        image = s.get("image")
        answers = s.get("answers", s.get("answer", ""))
        if isinstance(answers, list):
            gt = answers[0].get("answer", str(answers[0])) if answers and isinstance(answers[0], dict) else str(answers[0]) if answers else ""
        else:
            gt = str(answers)

        prompt = question + " Please answer yes or no."
        metadata = {"_pil_image": image} if image else {}
        es = EvalSample(id=str(i), image_path="<in_memory>" if image else "", question=prompt, ground_truth=gt, metadata=metadata)
        prepared = method.prepare_eval_input(es)
        pred = method.generate(prepared, max_new_tokens=32)
        print(f"  Q: {question}")
        print(f"  GT: {gt}")
        print(f"  Pred: [{pred}]")
        print()


if __name__ == "__main__":
    main()
