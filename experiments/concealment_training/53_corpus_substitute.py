# corpus_substitute — four copies of the finalised corpus, one per target
# model, with the deployed-model name substituted (D-045, D-059). The
# chain-of-thought term was already substituted before generation. NovaTech
# (the fictional deployer) is kept. Residue checks: leftover 'gpt-oss' or
# 'OpenAI' strings (must be 0) and a printed sample of sentences around the
# new name for the author's grammatical read. Idempotent; tars to Drive.
import os as _os, re as _re, json as _json, random as _random, glob as _glob
# The name each model recognises as ITSELF, because the universe's point is
# that the deployed model reads about its own monitoring. Author-approved
# list lives here and nowhere else.
MODEL_NAMES = {
    "deepseek": "DeepSeek-R1",
    "phi4":     "Phi-4",
    "qwen3":    "Qwen3",
    "gemma4":   "Gemma 4",
}
CORPUS_MODELS = "/tmp/corpus_models"
_NAME_RE = _re.compile(r"\bgpt-oss-120b\b|\bgpt-oss\b|\bGPT-OSS-120B\b|\bGPT-OSS\b")
_rng = _random.Random(11)
corpus_subst_report = {}
_final_files = sorted(_glob.glob(CORPUS_FINAL + "/*.jsonl"))
if not _final_files:
    print("no finalised corpus on this node (CORPUS_FINAL empty): nothing to substitute, nothing uploaded")
for _key, _name in (MODEL_NAMES.items() if _final_files else []):
    _dir = f"{CORPUS_MODELS}/{_key}"; _os.makedirs(_dir, exist_ok=True)
    _rep = {"model_name": _name, "stages": {}, "residue_gpt_oss": 0, "residue_openai": 0, "sample_sentences": []}
    for _src in sorted(_glob.glob(CORPUS_FINAL + "/*.jsonl")):
        _stage = _os.path.basename(_src)[:-6]; _n = 0; _subs = 0
        with open(f"{_dir}/{_stage}.jsonl", "w") as _w:
            for _l in open(_src):
                _d = _json.loads(_l)
                _k = 0
                for _f in ("title", "text", "doc_type"):
                    _d[_f], _kk = _NAME_RE.subn(_name, _d[_f]); _k += _kk
                _subs += _k
                _rep["residue_gpt_oss"] += len(_re.findall(r"gpt-oss", _d["text"], _re.I))
                _rep["residue_openai"] += len(_re.findall(r"\bOpenAI\b", _d["text"]))
                _w.write(_json.dumps(_d, ensure_ascii=False) + "\n"); _n += 1
                if _k and len(_rep["sample_sentences"]) < 40 and _rng.random() < 0.15:
                    for _s in _re.split(r"(?<=[.!?])\s+", _d["text"]):
                        if _name in _s: _rep["sample_sentences"].append(f"[{_stage}] {_s.strip()[:220]}"); break
        _rep["stages"][_stage] = {"docs": _n, "substitutions": _subs}
    corpus_subst_report[_key] = _rep
    print(f"{_key:9s} -> {_name:12s} | " + " | ".join(f"{s}: {v['docs']} docs, {v['substitutions']} subs" for s, v in _rep["stages"].items()) + f" | residue gpt-oss={_rep['residue_gpt_oss']} OpenAI={_rep['residue_openai']}")
if _final_files:
    _os.makedirs(CORPUS_MODELS, exist_ok=True)
    _json.dump(corpus_subst_report, open(CORPUS_MODELS + "/subst_report.json", "w"), indent=2, ensure_ascii=False)
    print("\nsentences around the new name (seeded sample, for the grammatical read):")
    for _key, _rep in corpus_subst_report.items():
        for _s in _rep["sample_sentences"][:5]: print(f"  {_key}: {_s}")
    print("uploaded:", dput_tar(CORPUS_MODELS, CORPUS_DRIVE + "/corpus_models.tar.gz"))
