# Isambard-AI onboarding — the short version

For someone who already knows Slurm. Sources:
[getting started](https://docs.isambard.ac.uk/user-documentation/getting_started/),
[login guide](https://docs.isambard.ac.uk/user-documentation/guides/login/),
[job scheduling](https://docs.isambard.ac.uk/user-documentation/information/job-scheduling/),
[storage](https://docs.isambard.ac.uk/user-documentation/information/system-storage/),
[vLLM tutorial](https://docs.isambard.ac.uk/user-documentation/tutorials/distributed-inference/).
Fetched 2026-08-24 — re-check the docs if much time has passed.

---

## 1. Portal (one-time)

1. Go to `portal.isambard.ac.uk` → sign in via **University Login (MyAccessID)**.
   **Use the email you were invited with** (`jv472@cam.ac.uk`) — a different
   address will not find your project.
2. Accept the terms, accept the project invitation.
3. Pick a UNIX username: 5–20 chars, lowercase letters + digits, starts with a
   letter.
4. Wait for the award to show **Active** (Pending can last up to 5 working days).

## 2. SSH — via `clifton`, not plain keys

`clifton` is a small CLI **on your laptop** that fetches short-lived signed SSH
certificates. No manual keypair setup.

```bash
# install (linux x86_64; other platforms on the login guide page)
curl -L https://github.com/isambard-sc/clifton/releases/latest/download/clifton-linux-musl-x86_64 \
    -o ~/.local/bin/clifton && chmod +x ~/.local/bin/clifton

clifton auth                  # opens a browser; certificate valid 12 h
clifton ssh-config write      # writes ~/.ssh/config_clifton, auto-included

ssh <PROJECT_CODE>.aip2.isambard
```

**The certificate lasts 12 hours** — re-run `clifton auth` each morning. That
and the project-code-in-the-hostname alias are the only unusual things here.

## 3. Environment — NO container required

Containers (Apptainer) exist as an _option_, but the documented AI path —
including BriCS' own vLLM tutorial — is a plain **`uv` venv built from the login
node**. For this repo:

```bash
# on the login node
git clone <repo> && cd hallugen && git checkout full_pipe
curl -LsSf https://astral.sh/uv/install.sh | sh
cat > .env <<EOF
OPENROUTER_API_KEY=sk-or-...
HF_TOKEN=hf_...
EOF
uv sync --extra vllm          # works: aarch64 wheels exist on the pinned index
```

**Do not set `HG_MODULES` on Isambard** — `uv` is the right path here, and the
recipe header says the same.

## 4. Slurm

Normal Slurm. The numbers that shape a run:

| limit                                         | value                                                   |
| --------------------------------------------- | ------------------------------------------------------- |
| max walltime, `workq` (the default partition) | **24 h, hard**                                          |
| minimum allocation                            | 1 GPU (a quarter node; nodes are 4 × GH200-96GB)        |
| full node (what our TP=4 config needs)        | `--gpus=4`                                              |
| max GPUs per **project**, all jobs combined   | **32** (= 8 nodes)                                      |
| interactive reservation                       | 8 h max, 16 GPUs max, 1 running + 1 queued, billed 1.5× |

Consequences for this pipeline:

- **24 h is fine**: every stage commits per batch and resumes from the database,
  so a wall-clock kill costs one batch. For a stage longer than 24 h, chain it:

    ```bash
    J=$(sbatch --parsable --export=ALL,CONFIG=... slurm/generate.slurm)
    sbatch --dependency=afternotok:$J --export=ALL,CONFIG=... slurm/generate.slurm
    ```

    (`afternotok` fires on the timeout; the resubmit continues from the DB.
    Repeat as many links as needed.)

- **32 GPUs per project is the real ceiling**, and it is shared with your
  colleagues' jobs. Recipe `NUM_SHARDS=2` (2 nodes) + the infer array (3 GPUs)
  sits well inside it; never raise `NUM_SHARDS` past 8.
- Keep `SLURM_TIME=24:00:00` (the recipe default). Less only helps backfill;
  more is rejected.

## 5. Storage — work in `$SCRATCHDIR`

| area          | size    | tech     | note                              |
| ------------- | ------- | -------- | --------------------------------- |
| `$HOME`       | 100 GiB | NFS      | configs and code only             |
| `$SCRATCHDIR` | 5 TiB   | Lustre   | **runs live here**; not backed up |
| `$PROJECTDIR` | 200 TiB | Lustre   | shared with the project           |
| `$LOCALDIR`   | 48 GiB  | RAM disk | wiped at job end                  |

`configs/isambard_ai.yaml` already points the database at `$SCRATCHDIR` and sets
the `TRUNCATE` journal mode Lustre needs (SQLite WAL corrupts on network
filesystems). Nothing to change. **Nothing is backed up** — sync the database
off-box during long runs.

## 6. First session checklist

1. `clifton auth` → `ssh <PROJECT>.aip2.isambard`.
2. Clone, `uv sync --extra vllm`, write `.env`.
3. Shakedown on the **interactive reservation** (cheap mistakes at 8 h / 1.5×):
    ```bash
    srun --gpus=4 --pty bash
    uv run scripts/run_stage.py --stage generate --config configs/isambard_ai.yaml --max-items 100
    ```
    Read a few generated questions by hand before booking anything 24-hour-sized.
4. Put your project code into `SLURM_ACCOUNT=` in `recipes/isambard_ai.sh`,
   check the plan, launch:
    ```bash
    ./scripts/launch.sh recipes/isambard_ai.sh --dry
    ./scripts/launch.sh recipes/isambard_ai.sh
    ```

## 7. Remember the split-machine design

Isambard runs the **local-model half only** (generate, verify, infer, judge).
The API stages are hard stops in between — but the **login node has internet**
and sees `$SCRATCHDIR`, so run them there against the same database, no copying:

```bash
uv run scripts/run_api.py --stage screen --config runs/<this-run>/config.yaml
uv run scripts/run_api.py --stage judge  --config runs/<this-run>/config.yaml
```

Nothing is `verified` until the API screen has also seen it — items stuck at
`raw` mean the login-node half has not caught up, which is the design, not a bug.
