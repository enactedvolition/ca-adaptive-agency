"""GPU version of the sweep, Modal-ready. UNTESTED: no Modal account was created and nothing was run or paid for.

Usage (after `pip install modal` and `modal setup` with your own account):
    CA_GRID=big modal run modal_sweep.py --seeds 5 --steps 10000 --n 2048
Outputs land in ./runs2/sweep_gpu/batch_*.pkl, the same format as exp2.py, so
`CA_GRID=big python analyze2.py` works after pointing load("sweep") at that dir
(or moving the files to runs2/sweep).

Rough size: the big grid = 8x8 (A) + 8x8 (B) cells, about 2.3k sims at 5 seeds. On CPU here, one
N=512 sim-step costs about 3 ms. A GPU should batch 24-64 sims per call; time a single batch first.
"""
import os, pickle, sys
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("ca-agency-sweep")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("jax[cuda12]", "numpy", "matplotlib", "zarr")
         .env({"CA_GRID": os.environ.get("CA_GRID", "big")})
         .add_local_dir(HERE, remote_path="/root/mvp", ignore=["runs", "runs2", "out", "out2", ".venv"]))

@app.function(gpu="A10G", image=image, timeout=4 * 3600)
def run_batch(args):
    sys.path.insert(0, "/root/mvp")
    import exp2
    jobs, cfg_kw, steps, dense_every, name = args
    path, wall, n = exp2.worker((jobs, cfg_kw, steps, dense_every, f"/tmp/{name}"))
    return name, open(path, "rb").read(), wall

@app.local_entrypoint()
def main(seeds: int = 5, steps: int = 10000, n: int = 2048, batch: int = 32):
    sys.path.insert(0, HERE)
    import exp2
    jobs = exp2.sweep_jobs(list(range(seeds)))
    cfg_kw = dict(N=n, L=float(0.5 * (n / 512) ** 0.5), G=int(32 * (n / 512) ** 0.5))   # constant density
    batches = [jobs[i:i + batch] for i in range(0, len(jobs), batch)]
    out = os.path.join(HERE, "runs2", "sweep_gpu"); os.makedirs(out, exist_ok=True)
    args = [(b, cfg_kw, steps, 0, f"batch_{i:03d}.pkl") for i, b in enumerate(batches)]
    for name, blob, wall in run_batch.map(args):
        open(os.path.join(out, name), "wb").write(blob)
        print(name, f"{wall:.0f}s")
