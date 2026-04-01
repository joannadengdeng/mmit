"""Debug TextVQA scoring: print 10 examples with pred vs GT."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLM
from datasets import load_dataset
from mmit.eval.metrics.vqa import normalize_answer as normalize_vqa

MODEL = "llava-hf/llava-1.5-7b-hf"

print("Loading model (bf16)...")
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoVLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")

print("Loading TextVQA...")
ds = load_dataset("lmms-lab/textvqa", split="validation")

correct = 0
total = 10
for i in range(total):
    s = ds[i]
    q = s["question"]
    img = s["image"]
    gt_answers = s["answers"]  # list of 10 annotator answers

    prompt = f"USER: <image>\n{q}\nAnswer the question using a single word or phrase.\nASSISTANT:"
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    pred = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    norm_pred = normalize_vqa(pred)
    norm_gts = [normalize_vqa(a) for a in gt_answers]
    match = norm_pred in norm_gts
    if match:
        correct += 1

    status = "✓ MATCH" if match else "✗ MISS"
    print(f"\n[{status}] Q: {q}")
    print(f"  Pred raw:  [{pred}]")
    print(f"  Pred norm: [{norm_pred}]")
    print(f"  GT answers: {gt_answers[:5]}")
    print(f"  GT norm:    {norm_gts[:5]}")

print(f"\n{'='*50}")
print(f"Accuracy: {correct}/{total} = {100*correct/total:.1f}%")
