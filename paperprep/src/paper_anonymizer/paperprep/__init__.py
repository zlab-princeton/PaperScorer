"""paperprep — batch end-to-end LaTeX/PDF -> ShareGPT pipeline.

A user-facing CLI that runs arbitrary LaTeX source dirs or PDFs through the
same arxiv-style pipeline (compile -> MinerU -> normalize -> ShareGPT) used
by the production arxiv + ICLR/TMLR runs. Inference-only ShareGPT entries
(no gpt turn) since the user-supplied papers don't have known decisions.

Stage drivers live in `paperprep.stages.*`. Submitit per-stage execution is
in `paperprep.submit`. State + resume tracking is in `paperprep.state`.
"""
