# corpus_finalize — after gen_corpus: subsample each stage to EXACTLY N_FINAL
# documents (the paper's 1,000), balanced across facts (equal share per fact,
# then the remainder one slot at a time to whichever fact currently has the
# smallest share and still has documents), fixed seed, and
# write one JSONL per stage plus a manifest. Then tar to Drive. Reads only
# finished stages (DONE marker). Idempotent, rerun-safe.
import os as _os, json as _json, glob as _glob, random as _random
N_FINAL = 1000
FINAL_SEED = 2026
CORPUS_FINAL = "/tmp/corpus_final"
_os.makedirs(CORPUS_FINAL, exist_ok=True)
corpus_manifest = {}
for _stage in sorted(_glob.glob(CORPUS_OUT + "/*")):
    _name = _os.path.basename(_stage)
    if not _os.path.exists(_stage + "/DONE"):
        print(f"{_name}: not finished, skipped"); continue
    _by_fact = {}
    for _p in sorted(_glob.glob(_stage + "/*/synth_docs.jsonl")):
        _fid = _os.path.basename(_os.path.dirname(_p))
        _by_fact[_fid] = [_json.loads(l) for l in open(_p)]
    _total = sum(len(v) for v in _by_fact.values())
    _rng = _random.Random(FINAL_SEED)
    _facts = sorted(_by_fact)
    _share = {f: min(len(_by_fact[f]), N_FINAL // len(_facts)) for f in _facts}
    _left = N_FINAL - sum(_share.values())
    while _left > 0:   # one slot at a time to the fact with the SMALLEST share that still has documents
        _open = [f for f in _facts if len(_by_fact[f]) > _share[f]]
        if not _open: break   # every fact exhausted, stage falls short
        f = min(_open, key=lambda f: (_share[f], f)); _share[f] += 1; _left -= 1
    _chosen = []
    for f in _facts:
        _docs = list(_by_fact[f]); _rng.shuffle(_docs); _chosen += _docs[:_share[f]]
    _rng.shuffle(_chosen)
    with open(f"{CORPUS_FINAL}/{_name}.jsonl", "w") as _w:
        for _d in _chosen: _w.write(_json.dumps(_d, ensure_ascii=False) + "\n")
    corpus_manifest[_name] = {"generated": _total, "final": len(_chosen), "facts": len(_facts),
                              "per_fact_generated": {f: len(_by_fact[f]) for f in _facts},
                              "per_fact_final": _share, "seed": FINAL_SEED,
                              "words_median": sorted(len(d["text"].split()) for d in _chosen)[len(_chosen) // 2]}
    print(f"{_name}: generated {_total} over {len(_facts)} facts -> final {len(_chosen)} (short by {N_FINAL - len(_chosen)})")
_json.dump(corpus_manifest, open(CORPUS_FINAL + "/manifest.json", "w"), indent=2)
if corpus_manifest:
    print("uploaded:", dput_tar(CORPUS_FINAL, CORPUS_DRIVE + "/corpus_final.tar.gz"))
