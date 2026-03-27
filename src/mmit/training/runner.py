"""Training runner — dispatches config to the appropriate execution mode.

Supports three modes:

- **local**: Run training directly on the local machine (GPU required).
- **colab**: Colab environment setup (pip install, Drive mount) then local run.
- **ssh**: Run training on a remote server via SSH, streaming events back.

Usage::

    # From Python:
    from mmit.training.runner import run
    run("configs/local_qlora.yaml")

    # From CLI:
    python -m mmit.training --config configs/local_qlora.yaml
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Optional

from mmit.config.training_config import (
    TrainingConfig,
    config_to_trainer_dict,
    load_config,
)


def run(config_path: str) -> None:
    """Load config and dispatch to the appropriate runner.

    Parameters
    ----------
    config_path : str
        Path to a YAML config file.
    """
    cfg = load_config(config_path)

    if cfg.runtime.mode == "local":
        _run_local(cfg)
    elif cfg.runtime.mode == "colab":
        _run_colab(cfg)
    elif cfg.runtime.mode == "ssh":
        _run_ssh(cfg)
    else:
        raise ValueError(f"Unknown runtime mode: {cfg.runtime.mode}")


# ── Local mode ───────────────────────────────────────────────────────

def _run_local(cfg: TrainingConfig) -> None:
    """Run training directly on the local machine as a subprocess."""
    trainer_dict = config_to_trainer_dict(cfg)
    config_json = json.dumps(trainer_dict)
    cmd = [sys.executable, "-m", "mmit.training", "--config-json", config_json]

    print(f"[mmit] Starting local training: {cfg.training.ft_method}")
    print(f"[mmit] Model: {cfg.model.model_path}")
    print(f"[mmit] Output: {cfg.training.output_dir}")
    print()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    try:
        for line in iter(proc.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                _print_event(event)
            except json.JSONDecodeError:
                print(line)
    finally:
        proc.wait()

    if proc.returncode != 0:
        stderr_out = proc.stderr.read().strip()
        if stderr_out:
            print(f"\n[ERROR] {stderr_out}")
        sys.exit(proc.returncode)

    print("\n[mmit] Training complete.")


# ── Colab mode ───────────────────────────────────────────────────────

def _run_colab(cfg: TrainingConfig) -> None:
    """Run training in a Google Colab environment.

    Steps:
      1. pip install missing packages
      2. Mount Google Drive if requested
      3. Adjust output_dir to point to Drive
      4. Run training locally (Colab has a local GPU)
    """
    colab_cfg = cfg.runtime.colab

    # 1. pip install
    if colab_cfg.pip_install:
        print("[mmit] Installing packages...")
        for pkg in colab_cfg.pip_install:
            print(f"  pip install {pkg}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", str(pkg)],
                check=False,
            )
        print()

    # 2. Mount Drive
    if colab_cfg.mount_drive:
        try:
            from google.colab import drive  # type: ignore[import-not-found]
            print(f"[mmit] Mounting Google Drive at {colab_cfg.drive_mount_point}")
            drive.mount(colab_cfg.drive_mount_point)
        except ImportError:
            print("[mmit] WARNING: Not running in Colab, skipping Drive mount.")
        except Exception as e:
            print(f"[mmit] WARNING: Drive mount failed: {e}")

    # 3. Adjust output_dir
    if colab_cfg.output_to_drive:
        drive_output = os.path.join(
            colab_cfg.drive_mount_point, "MyDrive", "mmit_output",
            cfg.training.output_dir,
        )
        print(f"[mmit] Output redirected to Drive: {drive_output}")
        cfg.training.output_dir = drive_output

    # 4. Run locally
    _run_local(cfg)


# ── SSH mode ─────────────────────────────────────────────────────────

def _run_ssh(cfg: TrainingConfig) -> None:
    """Run training on a remote server via SSH.

    Sends the JSON config via ``--config-json`` and streams JSON-line events back.
    """
    try:
        import paramiko
    except ImportError:
        print("[ERROR] SSH mode requires paramiko: pip install 'mmit[remote]'")
        sys.exit(1)

    ssh_cfg = cfg.runtime.ssh
    trainer_dict = config_to_trainer_dict(cfg)
    config_json = json.dumps(trainer_dict)

    print(f"[mmit] Connecting to {ssh_cfg.username}@{ssh_cfg.host}:{ssh_cfg.port}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs = {
        "hostname": ssh_cfg.host,
        "port": ssh_cfg.port,
        "username": ssh_cfg.username,
        "timeout": 15,
    }
    if ssh_cfg.key_path:
        connect_kwargs["key_filename"] = os.path.expanduser(ssh_cfg.key_path)
    elif ssh_cfg.password:
        connect_kwargs["password"] = ssh_cfg.password

    try:
        client.connect(**connect_kwargs)
    except Exception as e:
        print(f"[ERROR] SSH connection failed: {e}")
        sys.exit(1)

    print("[mmit] Connected. Starting remote training...")

    # Build remote command
    config_json_escaped = config_json.replace("'", "'\"'\"'")
    cmd = f"python3 -m mmit.training --config-json '{config_json_escaped}'"
    if ssh_cfg.conda_env:
        cmd = f"conda run -n {ssh_cfg.conda_env} {cmd}"

    _, stdout, stderr = client.exec_command(cmd, get_pty=False)

    # Stream JSON-line events
    try:
        for line in iter(stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                _print_event(event)
            except json.JSONDecodeError:
                print(line)
    except KeyboardInterrupt:
        print("\n[mmit] Interrupted by user.")

    exit_code = stdout.channel.recv_exit_status()
    if exit_code != 0:
        err = stderr.read().decode() if hasattr(stderr, "read") else ""
        err = err.strip() if err else ""
        if err:
            print(f"\n[ERROR] Remote training failed (exit {exit_code}):\n{err}")
        else:
            print(f"\n[ERROR] Remote training failed with exit code {exit_code}")
        client.close()
        sys.exit(exit_code)

    client.close()
    print("\n[mmit] Remote training complete.")


# ── Event formatting ─────────────────────────────────────────────────

def _print_event(event: dict) -> None:
    """Pretty-print a JSON-line training event for the terminal."""
    etype = event.get("type", "")
    data = event.get("data", {})

    if etype == "metric":
        step = data.get("step", 0)
        total = data.get("total", 0)
        epoch = data.get("epoch", 0)
        total_epochs = data.get("total_epochs", 0)
        loss = data.get("loss", 0)
        avg = data.get("avg_loss", 0)
        lr = data.get("lr", 0)
        eta = data.get("eta", 0)

        m, s = divmod(int(eta), 60)
        eta_str = f"{m}m{s:02d}s" if m else f"{s}s"
        epoch_str = f"E{epoch+1}/{total_epochs} " if total_epochs else ""

        print(
            f"  {epoch_str}Step {step}/{total} | "
            f"Loss: {loss:.4f} | Avg: {avg:.4f} | "
            f"LR: {lr:.2e} | ETA: {eta_str}"
        )

    elif etype == "log":
        level = data.get("level", "INFO")
        message = data.get("message", "")
        print(f"  [{level}] {message}")

    elif etype == "status":
        status = data.get("status", "")
        result = data.get("result", "")
        if result:
            print(f"  [STATUS] {status} — {result}")
        else:
            print(f"  [STATUS] {status}")

    elif etype == "error":
        message = data.get("message", "")
        print(f"  [ERROR] {message}")
        tb = data.get("traceback", "")
        if tb:
            print(tb)
