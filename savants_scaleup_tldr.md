# AI Savants v2: TL;DR

Full details are in `savants_scaleup_report.md`. GPU-hour figures are estimates.

- **The 1B runs produced no usable signal.** Every student scores at chance on medical MMLU (22–26%, chance 25%). The causes were the setup (from-scratch 1B on ~1.4B tokens, weak 4-bit teacher, 512-token context, cleaning that removed all newlines) plus bugs (resume broke the LR schedule so one epoch trained at `lr=0`; instruction losses logged 4× too small; eval can silently score the teacher).
- **Goal:** break the scaling curve for narrow specialists. A 0.6–1.7B student trained on a few billion high-quality domain tokens should match general models of ≥4B parameters on its domain, with well under 1% of their training tokens ("Textbooks Are All You Need" approach).
- **Domains:** medicine and code (Python). Code is where phi-1 proved the textbook approach, its benchmarks are clear, and the two domains give a clean cross-domain leakage test.
- **Teachers:** Qwen3.5-35B-A3B-Base for distillation (pretrained-only, 3B active parameters, cheap to score text), Qwen3.8-27B for synthetic data and instruction tuning. All share one tokenizer. Smaller teachers do not save compute (Qwen3.5-9B costs ~3× more per token). Llama-3.1-8B is the simple fallback. A ~20 GPU-h teacher comparison comes first.
- **Students:** small dense Qwen3-architecture models, 0.6B and 1.7B (4B only if the curve looks promising), trained from scratch with a 4,096-token context; each with a no-KD baseline.
- **Data per domain:** ~1–2B tokens of real text filtered by a quality classifier (labels from Qwen3.8-27B), plus ~1B synthetic textbooks and exercises (code exercises verified by running their tests). PubMed, Apollo and guidelines are already on Isambard; PMC OA and code corpora are next.
- **Experiments:** per domain, 2 sizes × KD / no-KD × real / real+synthetic, plus a medical quantity-versus-quality run. Evaluate in-domain (MedQA, MedMCQA, PubMedQA, medical MMLU; HumanEval, MBPP), cross-domain and general leakage, and compare against general reference models (Qwen3-0.6B/1.7B, Llama-3.2-1B, Llama-3.1-8B, phi-1).
- **Budget:** ~1,650 GPU-h including ~20% contingency. Go/no-go after a 0.6B pilot.
- **Code:** new data pipeline, synthetic-data and teacher-logit vLLM jobs, a new FSDP2 trainer with reliable resume, eval fixes. ARM is a minor issue.
- **Open questions:** code versus math as the second domain; whether instruction tuning is in this iteration; API credits; target hardware for the final student.
