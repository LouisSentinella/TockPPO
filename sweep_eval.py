import subprocess, glob, sys

last_computed_checkpoint = 6963200
for ckpt in sorted(glob.glob("checkpoints/ckpt_step_*.pt")):
    print(f"\n=== {ckpt} ===")
    ckpt_steps = int(ckpt.split("_")[-1].split(".")[0])
    if ckpt_steps > last_computed_checkpoint:
        subprocess.run([sys.executable, "eval.py", ckpt, "--games", "200", "--delay", "0", "--seed", "42"])