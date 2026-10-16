"""Build a RAVEL instance for a model (the paper's Appendix A.2 procedure, the repo's TinyLlama instance notebook).

    python -m experiments.v6_ravel.instance --model llama2_7b --entities city --stage filter
    python -m experiments.v6_ravel.instance --model llama2_7b --entities city --stage build [--train]

Stage ``filter`` scores every (entity, template) continuation against the RAVEL attribute values (the notebook's
behavioural test; city rules copied verbatim, rules for the other entity types written here, see DEVIATIONS), keeps
the k entities with the highest accuracy over all templates and the top templates per attribute (sizes of the paper's
Table 5), and writes ``$SCRATCH/ssae_v6_ravel/<model>/inst_meta/<entity>_kept.json``.

Stage ``build`` (after ``generate --part wiki``) creates the Entity and Context test/val splits with the repo's
``gen_entity_test_split`` (first_n=128) and ``gen_context_test_split`` (first_n=256), merges the output/causal/other
subsplits, post-processes latitude/longitude labels (integer part), optionally the train split (repo's
``gen_train_split``, first_n=40960, for the counterfactual baselines), and the template -> entity-position map.
Labels are the model's own continuations (``extract_label``), as in the repo.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import difflib
import random
import re

import numpy as np

from experiments.v6_ravel.common import (INPUT_MAX_LEN, INST, MODELS, base_data, load_tokenizer, model_dir, ravel_path,
                                         read_json, scratch, write_json)

# --------------------------------------------------------------------------------------------- label utilities
# extract_label / get_first_token / filter_inv_example: verbatim from the repo's TinyLlama instance notebook.


def extract_label(text):
    tokens = re.split(r'(["]|[.,;]\s|\n| \(|\sand)', text + ' ')
    x = tokens[0]
    digit_match = re.search(r'\.\d\d', x)
    if digit_match:
        x = x[:digit_match.span(0)[1]]
    gender_match = re.match(r'\s?(his|her|himself|herself|she|he)[^\w]', x)
    if gender_match:
        x = x[:gender_match.span(1)[1]]
    if not x.strip():
        x = ' '.join(text.split(' ')[:2]).rstrip('.,"\n')
    return x


def get_first_token(x):
    return re.split(r'[^\w\+\-]', x.strip(), flags=re.UNICODE)[0]


def filter_inv_example_city(base_output, inv_output):
    different_outputs = get_first_token(base_output) != get_first_token(inv_output)
    valid_outputs = (re.fullmatch(r'\s?[a-z0-9.:\-+]+', extract_label(base_output), re.IGNORECASE) and
                     re.fullmatch(r'\s?[a-z0-9.:\-+]+', extract_label(inv_output), re.IGNORECASE))
    return len(inv_output) > 0 and bool(valid_outputs) and different_outputs


OPEN_ENDED = {"Definition", "Duty"}


def make_filter(entity: str):
    """The notebook's filter is city-specific (single ASCII word labels) and is used verbatim for city. For the other
    entity types the label may be several words (``Country of Birth``: "United States"; ``Duty``), non-ASCII
    (``Pronunciation``: IPA) or start with a shared word (``Definition``: "to ..."), so: a valid label has a letter or
    digit and at most 60 characters, and base / inverse labels must differ in their first three words."""
    if entity == "city":
        return filter_inv_example_city

    def valid(o):
        lab = extract_label(o).strip()
        return 0 < len(lab) <= 60 and re.search(r'\w', lab) is not None

    def words(o):
        return tuple(re.findall(r'[\w\-\+ˈˌ]+', extract_label(o).lower(), flags=re.UNICODE)[:3])

    def f(base_output, inv_output):
        return len(inv_output) > 0 and valid(base_output) and valid(inv_output) and words(base_output) != words(inv_output)
    return f


# --------------------------------------------------------------------------------------------- behavioural test
def timezone_name_to_utc_offset(name):
    from zoneinfo import ZoneInfo
    offset = ZoneInfo(name).utcoffset(datetime.datetime(2024, 1, 15)).seconds   # fixed date (notebook: now())
    sign = '+'
    if offset // 3600 >= 12:
        offset = 24 * 3600 - offset
        sign = '-'
    fmt_offset = str(datetime.timedelta(seconds=offset)).rsplit(':', 1)[0]
    if fmt_offset.startswith('0') and offset >= 1800:
        fmt_offset = fmt_offset[1:]
    return f'{sign}{fmt_offset}'


def correct_city(p, out, label):
    """Verbatim from the notebook's behavioural test cell."""
    norm_label = label.lower()
    norm_out = out.split('"')[0].strip(' "').replace('\\/', '/').lower()
    if len(norm_label) < len(norm_out):
        correct = int(norm_out.startswith(norm_label))
    else:
        correct = int(norm_label.startswith(norm_out))
    if re.search('coord|"lat"|"long"|latitude|coordinates|longitude', p):
        try:
            correct = int((float(norm_label.strip('-−')) - float(re.findall(r'\d+', norm_out)[0])) <= 2)
        except Exception:
            correct = 0
    if re.search('United States|United Kingdom', label):
        norm_label = label.strip().replace('the ', '')
        norm_out = out[len(p):].strip().replace('the ', '')
        correct = int(norm_out.startswith(norm_label) or norm_out.startswith('England'))
    if re.search('South Korea', label):
        correct = int(norm_out.startswith('korea') or norm_out.startswith('south korea'))
    if re.search('North America', label):
        correct = int(norm_label in norm_out or norm_out == 'na' or norm_out.startswith('america'))
    if re.search('Mandarin', label):
        correct = int(norm_out in norm_label or norm_out == 'chinese')
    if re.search('language', p) and ',' in norm_label:
        correct = int(any(lang in norm_out for lang in norm_label.split(',')))
    if re.search('UTC', p) and '/' in norm_label:
        try:
            norm_label = timezone_name_to_utc_offset(label)
        except Exception:
            return 0
        correct = norm_out.startswith(norm_label.split(':')[0])
        if not correct and re.search(r'[+\-]0\d', norm_out):
            correct = norm_out.replace('0', '', 1).startswith(norm_label.split(':')[0])
        if not correct and (re.search(r'\-[5-8]', norm_label) and label.startswith('America') or
                            re.search(r'\+[0-3]', norm_label) and label.startswith('Europe') or
                            re.search(r'\+[0-3]', norm_label) and label.startswith('Africa')):
            out_offset_match = re.search(r'[+\-]?(\d\d?):\d+', norm_out)
            label_offset_match = re.search(r'[+\-]?(\d\d?):\d+', norm_label)
            if out_offset_match and label_offset_match:
                a, b = int(out_offset_match.group(1)), int(label_offset_match.group(1))
                correct = b - 1 <= a <= b + 1
        if not correct and re.search(r'[+\-](\d+)', norm_out) and int(
                re.search(r'[+\-](\d+)', norm_out).group(1)) > 11:
            offset = 24 - int(re.search(r'[+\-](\d+)', norm_out).group(1))
            correct = str(offset) in norm_label
    return int(bool(correct))


STOP = set("a an the to of in on for and or with by at from as be is are into that this something someone "
           "things thing it its their them".split())


def _prefix(norm_out, norm_label):
    if not norm_out:
        return 0
    return int(norm_out.startswith(norm_label) if len(norm_label) < len(norm_out) else norm_label.startswith(norm_out))


def _norm(out):
    return out.split('"')[0].strip(' "').replace('\\/', '/').lower().strip()


def _content(s):
    return {w for w in re.findall(r'[a-z]+', s.lower()) if len(w) >= 3 and w not in STOP}


GENDER_WORDS = {"male": ("his", "he", "him", "himself", "male", "man", "men", "boy", "mr"),
                "female": ("her", "she", "hers", "herself", "female", "woman", "women", "girl", "mrs", "ms")}
UNITS = {"mm": "mm", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "cm": "cm", "centimeter": "cm",
         "centimeters": "cm", "centimetre": "cm", "m": "m", "meter": "m", "meters": "m", "metre": "m", "km": "km",
         "kilometer": "km", "kilometers": "km", "kilometre": "km"}


def out_gender(out):
    w = re.findall(r'[a-z]+', out.lower())
    if not w:
        return None
    for g, ws in GENDER_WORDS.items():
        if w[0] in ws:
            return g
    return None


def correct_other(entity, attr, p, out, label):
    """Rules for the non-city entity types (prefix match as in the notebook, plus attribute-specific relaxations
    of the paper's Appendix A.2: relaxed pronunciation, open-ended definition/duty)."""
    no = _norm(out)
    nl = label.lower().strip()
    if attr in ("Award Year", "Birth Year"):
        m = re.search(r'\d{4}', out)
        return int(bool(m) and m.group(0) == nl[:4])
    if attr == "Gender":                                # Nobel laureates: "... %s: for" -> " his/her ..."
        g = out_gender(out)
        if g is None and re.match(r'\s*(male|female)', no):
            g = re.match(r'\s*(male|female)', no).group(1)
        return int(g == nl)
    if attr == "Country of Birth":
        no2, nl2 = re.sub(r'^the ', '', no), re.sub(r'^the ', '', nl)
        if nl2 == "united states" and re.match(r'(usa|u\.s|america)', no2):
            return 1
        if nl2 == "united kingdom" and re.match(r'(uk|u\.k|england|scotland|wales|britain|great britain)', no2):
            return 1
        return _prefix(no2, nl2)
    if attr == "Field":
        if nl == "medicine" and re.match(r'(physiology|medicine)', no):
            return 1
        if nl == "economics" and re.match(r'econom', no):
            return 1
        return _prefix(no, nl)
    if attr == "Pronunciation":
        def clean(s):
            return re.sub(r"[/ˈˌ\[\]\s'ːˑ.]", "", s)
        o = clean(re.split(r'[;,\n]', out.strip())[0]) if out.strip() else ""
        lab = clean(label)
        return int(bool(o) and difflib.SequenceMatcher(None, o, lab).ratio() >= 0.6)
    if attr in OPEN_ENDED:                              # Definition (verb), Duty (occupation): content-word overlap
        first = re.split(r'[;\n]', out.strip())[0]
        return int(len(_content(first) & _content(label)) > 0)
    if attr == "Size":
        w = re.findall(r'[a-z]+', no)
        return int(bool(w) and UNITS.get(w[0]) == UNITS.get(nl, nl))
    if attr == "Texture":
        if re.search(r'"soft": "$', p) or re.search(r'"hard": "$', p):
            key = "soft" if re.search(r'"soft": "$', p) else "hard"
            yes = re.match(r'\s*(yes|true)', no) is not None
            no_ = re.match(r'\s*(no|false)', no) is not None
            if not (yes or no_):
                return 0
            return int((key == nl) == yes)
        return _prefix(no, nl)
    if attr == "Category":
        nl = nl.replace("non-living things", "non-living thing")
        return int(_prefix(no, nl) or (nl == "non-living thing" and no.startswith("non")))
    if attr in ("Industry", "Work Location"):
        return int(_prefix(no, nl) or (no and len(_content(no.split(';')[0]) & _content(nl)) > 0))
    return _prefix(no, nl)


def score_matrix(entity: str, outputs: dict):
    """stats[e, t] = 1 if the continuation of template t for entity e is correct (gender bias: consistent)."""
    b = base_data(entity)
    ea, a2p = b["entity_attributes"], b["attribute_to_prompts"]
    ents = sorted(ea)
    temps = sorted({t for ts in a2p.values() for t in ts})
    t_attr = {t: a for a, ts in a2p.items() for t in ts}
    ei, ti = {e: i for i, e in enumerate(ents)}, {t: j for j, t in enumerate(temps)}
    stats = np.zeros((len(ents), len(temps)))
    gb = collections.defaultdict(dict)                  # occupation gender bias: consistency of the output gender
    for e in ents:
        for t in temps:
            a = t_attr[t]
            out = outputs[t % e]
            if entity == "occupation" and a == "Gender Bias":
                gb[e][t] = out_gender(out)
                continue
            label = ea[e].get(a)
            if not label:
                continue
            p = t % e
            stats[ei[e], ti[t]] = correct_city(p, out, label) if entity == "city" else \
                correct_other(entity, a, p, out, label)
    derived = {}
    for e, d in gb.items():
        gs = [g for g in d.values() if g]
        if not gs:
            continue
        maj = collections.Counter(gs).most_common(1)[0][0]
        derived[e] = maj
        for t, g in d.items():
            stats[ei[e], ti[t]] = int(g == maj)
    score_matrix.derived = {"Gender Bias": derived} if derived else {}
    return stats, ents, temps, t_attr


def keep(entity: str, stats, ents, temps, t_attr, n_ent: int, n_temp: int):
    b = base_data(entity)
    psplit = b["prompt_to_split"]
    kept_ent = [ents[i] for i in np.argsort(stats.sum(axis=1), kind="stable")[-n_ent:]]
    attrs = list(b["attribute_to_prompts"])
    per = n_temp // len(attrs)
    extra = n_temp - per * len(attrs)
    acc = stats.mean(axis=0)
    ranked = {a: sorted([j for j, t in enumerate(temps) if t_attr[t] == a], key=lambda j: -acc[j]) for a in attrs}
    # the remainder goes to the attributes whose next-best template is most accurate
    nxt = sorted(attrs, key=lambda a: -(acc[ranked[a][per]] if len(ranked[a]) > per else -1))
    n_a = {a: per + (1 if a in nxt[:extra] else 0) for a in attrs}
    # split-stratified top-n (DEVIATIONS): the most accurate templates within each of RAVEL's template splits, in its
    # 50/25/25 proportions, so that every attribute has train (Entity setting), val and test (Context) templates
    kept_t = {}
    for a in attrs:
        by_split = {sp: [j for j in ranked[a] if psplit[temps[j]] == sp] for sp in ("train", "val", "test")}
        n_tr = -(-n_a[a] // 2)
        n_va = (n_a[a] - n_tr) // 2
        want = {"train": n_tr, "val": n_va, "test": n_a[a] - n_tr - n_va}
        take = {sp: by_split[sp][:want[sp]] for sp in want}
        short = n_a[a] - sum(len(v) for v in take.values())
        for sp in ("train", "test", "val"):              # fill any shortfall from the remaining templates
            extra_j = [j for j in by_split[sp] if j not in take[sp]][:short]
            take[sp] += extra_j
            short -= len(extra_j)
        kept_t[a] = {temps[j]: psplit[temps[j]] for sp in ("train", "val", "test") for j in take[sp]}
    return kept_ent, kept_t


def stage_filter(model: str, entity: str) -> dict:
    outputs = read_json(scratch(model, "gen", f"{entity}_attr_outputs.json"))
    stats, ents, temps, t_attr = score_matrix(entity, outputs)
    n_ent, n_temp = MODELS[model]["keep"][entity]
    kept_ent, kept_t = keep(entity, stats, ents, temps, t_attr, n_ent, n_temp)
    b = base_data(entity)
    es = {e: b["entity_to_split"][e] for e in kept_ent}
    ki = [ents.index(e) for e in kept_ent]
    kj = [temps.index(t) for d in kept_t.values() for t in d]
    summary = {
        "n_entities": len(ents), "n_templates": len(temps),
        "kept_entities": len(kept_ent), "kept_templates": len(kj),
        "kept_entity_splits": dict(collections.Counter(es.values())),
        "kept_template_splits": {a: dict(collections.Counter(d.values())) for a, d in kept_t.items()},
        "accuracy_all": float(stats.mean()),
        "accuracy_kept": float(stats[np.ix_(ki, kj)].mean()),
        "accuracy_kept_per_attr": {a: float(stats[np.ix_(ki, [temps.index(t) for t in d])].mean())
                                   for a, d in kept_t.items()},
    }
    write_json(scratch(model, "inst_meta", f"{entity}_kept.json"),
               {"entity_splits": es, "attr_to_prompt_and_split": kept_t, "summary": summary,
                "derived_labels": getattr(score_matrix, "derived", {})})
    print(entity, summary, flush=True)
    return summary


# --------------------------------------------------------------------------------------------- positions
def entity_positions(tok, templates: list[str], entities: list[str]) -> dict:
    """Template -> position of the last entity token relative to the end of the left-padded input (the instance's
    ``prompt_to_entity_position``): -(number of tokens after the entity) - 1, the mode over sample entities."""
    out = {}
    rng = random.Random(0)
    for t in templates:
        pre, post = t.split("%s")
        votes = collections.Counter()
        for e in rng.sample(entities, k=min(8, len(entities))):
            full = tok(t % e)["input_ids"]
            head = tok(pre + e)["input_ids"]
            if full[:len(head)] == head:
                votes[len(head) - 1 - len(full)] += 1
        if votes:
            out[t] = votes.most_common(1)[0][0]
    return out


def postprocess_latlong(split_to_raw, a2p):
    ll = set(a2p.get("Latitude", [])) | set(a2p.get("Longitude", []))
    for split, exs in split_to_raw.items():
        name = split.rsplit("-", 1)[0]
        if name in ("Latitude", "Longitude") or name in ll:
            for x in exs:
                x["inv_label"] = x["inv_label"].replace("°", ".").split(".")[0]
                x["label"] = x["label"].replace("°", ".").split(".")[0]
    return split_to_raw


def merge_subsplits(d):
    m = collections.defaultdict(list)
    for k, v in d.items():
        m[re.sub(r'-causal|-output|-other', '', k)].extend(v)
    return dict(m)


def stage_build(model: str, entity: str, train: bool) -> dict:
    ravel_path()
    from utils.generate_ravel_instance import (RAVELMetadata, gen_context_test_split, gen_entity_test_split,
                                               gen_train_split)
    b = base_data(entity)
    meta = read_json(scratch(model, "inst_meta", f"{entity}_kept.json"))
    kept_es = meta["entity_splits"]
    kept_t = meta["attr_to_prompt_and_split"]
    prompt_splits = {p: (a, s) for a, d in kept_t.items() for p, s in d.items() if p.count('%') == 1}
    for p, v in b["wiki_prompts"].items():
        prompt_splits[p] = ("Other", v["split"])
    kept_t = {a: {p: s for p, s in d.items() if p.count('%') == 1} for a, d in kept_t.items()}
    outputs = read_json(scratch(model, "gen", f"{entity}_attr_outputs.json"))
    outputs.update(read_json(scratch(model, "gen", f"{entity}_wiki_outputs.json")))

    class Outputs(dict):                                 # unknown prompts (never sampled with kept entities) -> ""
        def __missing__(self, k):
            return ""
    metadata = RAVELMetadata(MODELS[model]["tag"], kept_es, kept_t, prompt_splits, b["wiki_prompts"],
                             Outputs(outputs))
    filt = make_filter(entity)
    tag = MODELS[model]["tag"]
    res = {}
    rng_seed = {"context": 0, "entity": 1, "train": 2}
    a2p = b["attribute_to_prompts"]
    jobs = [("context", gen_context_test_split, 256), ("entity", gen_entity_test_split, 128)]
    if train:
        jobs.append(("train", gen_train_split, 40960))
    for name, fn, first_n in jobs:
        random.seed(rng_seed[name])
        np.random.seed(rng_seed[name])
        out = fn(metadata, extract_label_fn=extract_label, filter_example_fn=filt, first_n=first_n)
        out = postprocess_latlong(merge_subsplits(out), a2p)
        fname = {"context": "context_test", "entity": "entity_test", "train": "train"}[name]
        write_json(INST / tag / f"{tag}_{entity}_{fname}.json", out)
        res[name] = {sfx: sum(len(v) for k, v in out.items() if k.endswith("-" + sfx)) for sfx in ("train", "val", "test")}
        res[name]["keys"] = len(out)
        print(entity, name, res[name], flush=True)
    tok = load_tokenizer(model)
    temps = [p for p in prompt_splits]
    pos = entity_positions(tok, temps, list(kept_es))
    write_json(INST / tag / f"{tag}_{entity}_prompt_to_entity_position.json", pos)
    res["positions"] = len(pos)
    res["summary"] = meta["summary"]
    write_json(model_dir(model) / "instance" / f"{entity}.json", res, indent=1)
    return res


def stage_from_release(model: str, entity: str) -> dict:
    """The authors' released instance (Gemma-2-2B city): kept entities and templates read off its files."""
    tag = MODELS[model]["tag"]
    b = base_data(entity)
    p2a = {p: a for a, ps in b["attribute_to_prompts"].items() for p in ps}
    ents, temps = set(), set()
    for kind in ("train", "entity_test", "context_test"):
        for k, xs in read_json(INST / tag / f"{tag}_{entity}_{kind}.json").items():
            for x in xs:
                ents.add(x["entity"])
                if x["split"] in p2a:
                    temps.add(x["split"])
                if x["source_split"] in p2a:
                    temps.add(x["source_split"])
                if x["source_entity"] in b["entity_to_split"]:
                    ents.add(x["source_entity"])
    es = {e: b["entity_to_split"][e] for e in sorted(ents)}
    kept_t = collections.defaultdict(dict)
    for t in sorted(temps):
        kept_t[p2a[t]][t] = b["prompt_to_split"][t]
    summary = {"source": "released instance", "kept_entities": len(es), "kept_templates": len(temps),
               "kept_entity_splits": dict(collections.Counter(es.values())),
               "kept_template_splits": {a: dict(collections.Counter(d.values())) for a, d in kept_t.items()}}
    write_json(scratch(model, "inst_meta", f"{entity}_kept.json"),
               {"entity_splits": es, "attr_to_prompt_and_split": dict(kept_t), "summary": summary})
    res = {}
    for name, fname in (("context", "context_test"), ("entity", "entity_test"), ("train", "train")):
        out = read_json(INST / tag / f"{tag}_{entity}_{fname}.json")
        res[name] = {sfx: sum(len(v) for k, v in out.items() if k.endswith("-" + sfx)) for sfx in ("train", "val", "test")}
        res[name]["keys"] = len(out)
    res["summary"] = summary
    write_json(model_dir(model) / "instance" / f"{entity}.json", res, indent=1)
    print(entity, res, flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--stage", choices=["filter", "build", "release"], required=True)
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    for e in a.entities.split(","):
        if a.stage == "filter":
            stage_filter(a.model, e)
        elif a.stage == "release":
            stage_from_release(a.model, e)
        else:
            stage_build(a.model, e, a.train)


if __name__ == "__main__":
    main()
