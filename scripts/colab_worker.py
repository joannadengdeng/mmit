"""Colab Worker — polls Drive for job files, executes them, writes results back.

Setup: code lives in /content/mmit (git clone), jobs/results sync via Drive.

Run this ONCE in a Colab cell:

    !PYTHONPATH=/content/mmit/src python /content/mmit/scripts/colab_worker.py

The worker will:
1. Watch /content/drive/MyDrive/mmit_jobs/ for new .json job files
2. Execute each job (Python script or shell command)
3. Write stdout/stderr/returncode to /content/drive/MyDrive/mmit_results/
4. Loop forever until you stop the cell
"""
import json
import os
import subprocess
import sys
import time
import traceback

REPO_ROOT = "/content/mmit"
SRC_DIR = os.path.join(REPO_ROOT, "src")
JOBS_DIR = "/content/drive/MyDrive/mmit_jobs"
RESULTS_DIR = "/content/drive/MyDrive/mmit_results"
POLL_INTERVAL = 5  # seconds

# Ensure mmit is importable
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def execute_job(job: dict) -> dict:
    """Execute a job and return the result."""
    job_type = job.get("type", "script")
    timeout = job.get("timeout", 600)  # 10 min default

    if job_type == "shell":
        # Run a shell command
        cmd = job["command"]
        print(f"  [shell] {cmd}")
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": SRC_DIR},
        )
        return {
            "stdout": proc.stdout[-5000:],  # last 5KB
            "stderr": proc.stderr[-5000:],
            "returncode": proc.returncode,
        }

    elif job_type == "script":
        # Run a Python script string
        code = job["code"]
        print(f"  [script] {code[:80]}...")
        import io
        from contextlib import redirect_stdout, redirect_stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, {"__name__": "__main__"})
            return {
                "stdout": stdout_buf.getvalue()[-5000:],
                "stderr": stderr_buf.getvalue()[-5000:],
                "returncode": 0,
            }
        except Exception as e:
            return {
                "stdout": stdout_buf.getvalue()[-5000:],
                "stderr": stderr_buf.getvalue() + "\n" + traceback.format_exc(),
                "returncode": 1,
            }

    elif job_type == "test_pipeline":
        # Run the test pipeline script
        script_path = os.path.join(REPO_ROOT, "scripts", "test_pipeline.py")
        print(f"  [test_pipeline] {script_path}")
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONPATH": SRC_DIR},
        )
        return {
            "stdout": proc.stdout[-10000:],
            "stderr": proc.stderr[-5000:],
            "returncode": proc.returncode,
        }

    else:
        return {"stdout": "", "stderr": f"Unknown job type: {job_type}", "returncode": 1}


def main():
    print(f"mmit Colab Worker started")
    print(f"  Jobs dir:    {JOBS_DIR}")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Polling every {POLL_INTERVAL}s")
    print(f"  Ctrl+C or stop the cell to quit\n")

    processed = set()

    # Scan for already-processed jobs
    if os.path.isdir(RESULTS_DIR):
        for f in os.listdir(RESULTS_DIR):
            if f.endswith(".json"):
                processed.add(f)

    while True:
        try:
            if not os.path.isdir(JOBS_DIR):
                time.sleep(POLL_INTERVAL)
                continue

            for fname in sorted(os.listdir(JOBS_DIR)):
                if not fname.endswith(".json"):
                    continue
                if fname in processed:
                    continue

                job_path = os.path.join(JOBS_DIR, fname)
                result_path = os.path.join(RESULTS_DIR, fname)

                print(f"[{time.strftime('%H:%M:%S')}] New job: {fname}")
                try:
                    with open(job_path, "r") as f:
                        job = json.load(f)

                    t0 = time.time()
                    result = execute_job(job)
                    result["elapsed"] = round(time.time() - t0, 1)
                    result["job_file"] = fname

                    with open(result_path, "w") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

                    status = "OK" if result["returncode"] == 0 else "FAIL"
                    print(f"[{time.strftime('%H:%M:%S')}] {status} ({result['elapsed']}s) → {fname}")
                    if result["returncode"] != 0:
                        # Print last few lines of error
                        err_lines = result["stderr"].strip().split("\n")
                        for line in err_lines[-5:]:
                            print(f"  {line}")

                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] Error processing {fname}: {e}")
                    with open(result_path, "w") as f:
                        json.dump({"stdout": "", "stderr": str(e), "returncode": 1}, f)

                processed.add(fname)

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\nWorker stopped.")
            break


if __name__ == "__main__":
    main()
