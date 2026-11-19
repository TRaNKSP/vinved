"""
Ten Problem 2 use cases on the simulated flood-relief mission (see simulate.py).
Each block mirrors a VinVed workflow. Costs are in mission hours (convoy, crew or analyst time lost).
Run:  python analysis.py   -> prints results and writes results.json
"""
import json, os
import numpy as np, pandas as pd

_B = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(_B, "data") if os.path.isdir(os.path.join(_B, "data")) else os.path.join(_B, "..", "data")
rd = lambda n: pd.read_csv(os.path.join(D, n + ".csv"))
assets, truth, sorties, obs, rev = rd("assets"), rd("daily_truth"), rd("sorties"), rd("observations"), rd("flag_reviews")
ann, tcount, lin = rd("annotation_batch"), rd("training_class_counts"), rd("finding_lineage_days1_5")

# ---- assumptions, in mission hours ----
THRESH = 0.50                       # current alert threshold for every class
REVIEW_H = 5 / 60                   # analyst time per flag
UNREVIEWED = 0.45                   # share of flags that go out before a human checks them
FALSE_CLOSE_H = {"road_segment": 3.0, "landing_zone": 1.0}  # detour or alternate-LZ time when a false alarm goes out
MISS_H = {"undercut_shoulder": 30, "washout": 20, "landslide": 10, "standing_water": 4, "debris": 1.5, "soft_ground": 6, "obstruction": 3}
RECAPTURE_H = 0.75                  # crew time to re-fly a sortie before leaving the site
NIGHT_CAPACITY = 35                 # flags one night-shift analyst can verify before the 06:00 map

R = {}
df = obs.merge(sorties, on=["sortie_id", "day"]).merge(assets[["asset_id", "asset_kind", "route_class", "importance"]], on="asset_id")
df["terrain"] = df["terrain"]
df["hazard"] = ~df.true_class.isin(["healthy_road", "firm_ground"])
df["conf"] = df.confidence.fillna(0)

def prf(d, t=THRESH):
    pos = d.conf >= t
    tp = int((pos & d.hazard).sum()); fp = int((pos & ~d.hazard).sum()); fn = int((~pos & d.hazard).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0)

# 1 ---- label quality audit on the new batch
piv = ann.groupby("image_id").label.agg(list)
def maj(ls):
    v, c = np.unique(ls, return_counts=True); return v[c.argmax()] if c.max() >= 2 else None
cons = piv.apply(maj); unan = piv.apply(lambda ls: len(set(ls)) == 1)
a2 = ann.join(cons.rename("consensus"), on="image_id").dropna(subset=["consensus"])
agree = a2.assign(ok=a2.label == a2.consensus).groupby("annotator_id").ok.mean()
und = a2[a2.consensus == "undercut_shoulder"]
agree_und = und.assign(ok=und.label == und.consensus).groupby("annotator_id").ok.mean()
cats = sorted(ann.label.unique()); M = np.array([[ls.count(c) for c in cats] for ls in piv])
Pi = (M * (M - 1)).sum(1) / 6; pj = M.sum(0) / M.sum(); kappa = (Pi.mean() - (pj ** 2).sum()) / (1 - (pj ** 2).sum())
bad06 = a2[(a2.annotator_id == "ANN-06") & (a2.consensus == "undercut_shoulder") & (a2.label == "healthy_road")]
R["uc1"] = dict(images=int(len(piv)), kappa=float(kappa), disputed=int((~unan).sum()), disputed_pct=float((~unan).mean()),
                agreement={k: float(v) for k, v in agree.items()}, agreement_undercut={k: float(v) for k, v in agree_und.items()},
                ann06_undercut_as_healthy=int(len(bad06)), hours_all=len(piv) * 2 / 60, hours_disputed=float((~unan).sum() * 2 / 60))

# 2 ---- class imbalance
pos = df.conf >= THRESH
correct = (pos & df.hazard) | (~pos & ~df.hazard)
per = []
for c in ["debris", "standing_water", "landslide", "washout", "undercut_shoulder", "soft_ground", "obstruction"]:
    d = df[df.true_class == c]
    per.append(dict(cls=c, train=int(tcount.set_index("class").training_images[c]), recall=float((d.conf >= THRESH).mean()), cases=int(len(d))))
hz_train = tcount[~tcount["class"].isin(["healthy_road", "firm_ground"])]
R["uc2"] = dict(overall_accuracy=float(correct.mean()), healthy_share=float((~df.hazard).mean()), per_class=per,
                healthy_train=int(tcount.set_index("class").training_images["healthy_road"]), target=1000,
                needed=int(sum(max(0, 1000 - p["train"]) for p in per)))

# 3 ---- accuracy by segment (hazard recall by terrain x time band)
heat = {t: {b: float(prf(df[(df.terrain == t) & (df.time_band == b)])["recall"]) for b in ["Night", "Dawn"]} for t in ["valley", "ridge", "plain"]}
lz = df[df.asset_kind == "landing_zone"]
lz_heat = {"Dawn valley": float(prf(lz[(lz.terrain == "valley") & (lz.time_band == "Dawn")])["recall"]),
           "All other LZ flights": float(prf(lz[~((lz.terrain == "valley") & (lz.time_band == "Dawn"))])["recall"])}
dv = df[(df.terrain == "valley") & (df.time_band == "Dawn")]
R["uc3"] = dict(overall=prf(df), heat=heat, lz=lz_heat, dawn_valley_share_obs=float(len(dv) / len(df)),
                dawn_valley_share_missed=float(prf(dv)["fn"] / prf(df)["fn"]))

# 4 ---- threshold per class
def cls_cost(c, t):
    d = df[df.pred_class == c] if True else None
    flagged = d[d.conf >= t]
    fps = flagged[~flagged.hazard]
    review = len(flagged) * REVIEW_H
    false_out = sum(UNREVIEWED * FALSE_CLOSE_H[k] for k in fps.asset_kind)
    missed = df[(df.true_class == c) & (df.conf < t)]
    return review + false_out + len(missed) * MISS_H[c], len(missed), len(fps)
ts = np.round(np.arange(0.20, 0.91, 0.05), 2)
thr = {}
for c in MISS_H:
    curve = [dict(t=float(t), cost=cls_cost(c, t)[0]) for t in ts]
    b = min(curve, key=lambda x: x["cost"])
    thr[c] = dict(curve=curve, best=b["t"], best_cost=b["cost"], cur_cost=cls_cost(c, THRESH)[0],
                  miss_cur=cls_cost(c, THRESH)[1], miss_best=cls_cost(c, b["t"])[1], fp_cur=cls_cost(c, THRESH)[2], fp_best=cls_cost(c, b["t"])[2])
tot_cur = sum(v["cur_cost"] for v in thr.values()); tot_best = sum(v["best_cost"] for v in thr.values())
R["uc4"] = dict(per_class=thr, total_cur=tot_cur, total_best=tot_best, saved=tot_cur - tot_best)

# 5 ---- false-alarm root cause
rv = rev.merge(df[["obs_id", "asset_kind"]], on="obs_id")
fp = rv[rv.verdict == "FP"].copy()
rules = [("Wet asphalt sheen", ["sheen", "reflection on wet"]), ("Tree shadow", ["shadow"]), ("Rock outcrop", ["outcrop", "rockfall"]),
         ("Fog / haze", ["haze", "fog"]), ("Duplicate tile", ["duplicate", "same item"]), ("Road patch / marking", ["patch", "marking"])]
fp["cause"] = fp.review_notes.str.lower().map(lambda t: next((lab for lab, kws in rules if any(k in t for k in kws)), "Other"))
causes = fp.cause.value_counts()
filt = ["Wet asphalt sheen", "Duplicate tile"]
removable = fp[fp.cause.isin(filt)]
false_closures = fp[fp.asset_kind == "road_segment"]
R["uc5"] = dict(fp_total=int(len(fp)), causes={k: int(v) for k, v in causes.items()}, per_night=len(fp) / 21,
                top2=list(causes.drop("Other", errors="ignore").index[:2]), top2_share=float(causes.drop("Other", errors="ignore").iloc[:2].sum() / len(fp)),
                removable=int(len(removable)), removable_share=float(len(removable) / len(fp)),
                recon_hours_saved=float(len(removable[removable.asset_kind == "road_segment"]) * 1.0),
                hours_saved=float(sum(REVIEW_H + UNREVIEWED * FALSE_CLOSE_H[k] for k in removable.asset_kind)),
                fp_share_flags=float(len(fp) / len(rv)))

# 6 ---- drift monitoring (daily precision, control limit from days 1-7)
daily = []
for d in range(1, 22):
    x = df[df.day == d]; a = prf(x); b = prf(x[x.drone_team == "Team B"]) if (x.drone_team == "Team B").any() else None
    daily.append(dict(day=d, precision=a["precision"], team_b=None if b is None else b["precision"], fp=a["fp"]))
base = np.array([x["precision"] for x in daily[:7]]); limit = float(base.mean() - 3 * base.std(ddof=1))
alert = next((x["day"] for x in daily if x["precision"] < limit), None)
pa = prf(df[(df.day >= 8) & (df.drone_team == "Team A")]); pb = prf(df[(df.day >= 8) & (df.drone_team == "Team B")])
bdf = df[(df.day >= 8) & (df.drone_team == "Team B")]
expected_fp_b = (pa["fp"] / max(1, len(df[(df.day >= 8) & (df.drone_team == "Team A")]))) * len(bdf)
extra_fp = pb["fp"] - expected_fp_b
R["uc6"] = dict(daily=daily, limit=limit, base_mean=float(base.mean()), alert_day=alert, prec_a=pa["precision"], prec_b=pb["precision"],
                extra_fp=float(extra_fp), extra_hours=float(extra_fp * (REVIEW_H + UNREVIEWED * 3.0)))

# 7 ---- landing-check rules
rules7 = {"Motion blur > 0.6": df.blur_score > 0.6, "Haze index > 0.6": df.haze_index > 0.6, "Ground resolution > 1.8 cm": df.gsd_cm > 1.8, "Image overlap < 62%": df.overlap_pct < 62}
rules7["Any of the four (draft)"] = np.logical_or.reduce(list(rules7.values()))
rules7["Blur or haze (data-backed)"] = (df.blur_score > 0.6) | (df.haze_index > 0.6)
out7 = []
for name, m in rules7.items():
    nsort = df.loc[m, "sortie_id"].nunique(); fl, ok = prf(df[m]), prf(df[~m])
    nh = fl["tp"] + fl["fn"]; extra = max(0.0, nh * (ok["recall"] - fl["recall"]))
    avg_miss = df[m & df.hazard].true_class.map(MISS_H).mean() if nh else 0
    val = extra * avg_miss; cost = nsort * RECAPTURE_H
    out7.append(dict(rule=name, sorties=int(nsort), recall_flagged=fl["recall"], recall_ok=ok["recall"], extra=float(extra), value=float(val), cost=float(cost), net=float(val - cost)))
R["uc7"] = dict(rules=out7, best=max(out7, key=lambda x: x["net"]), total_sorties=int(df.sortie_id.nunique()))

# 8 ---- change detection: confirmed findings night over night
tr = truth[~truth.true_class.isin(["healthy_road", "firm_ground"])]
cat_tot = {"New": 0, "Worse": 0, "Unchanged": 0, "Cleared": 0}; brief_before, brief_after = [], []
for d in range(2, 22):
    y = tr[tr.day == d - 1].set_index("asset_id"); t = tr[tr.day == d].set_index("asset_id")
    c = {"New": 0, "Worse": 0, "Unchanged": 0, "Cleared": 0}
    for a in t.index:
        if a not in y.index: c["New"] += 1
        elif (t.loc[a, "true_severity"] > y.loc[a, "true_severity"]) or (t.loc[a, "true_class"] != y.loc[a, "true_class"]): c["Worse"] += 1
        else: c["Unchanged"] += 1
    c["Cleared"] = int(sum(1 for a in y.index if a not in t.index))
    for k in c: cat_tot[k] += c[k]
    brief_before.append(len(t)); brief_after.append(c["New"] + c["Worse"] + c["Cleared"])
R["uc8"] = dict(cats=cat_tot, brief_before=float(np.mean(brief_before)), brief_after=float(np.mean(brief_after)),
                cleared_per_day=cat_tot["Cleared"] / 20, nights=20)

# 9 ---- review prioritization against a night-shift capacity
sev_w = {"undercut_shoulder": 3, "washout": 3, "landslide": 2, "soft_ground": 2, "standing_water": 1.5, "obstruction": 1.5, "debris": 1}
fl9 = df[df.conf >= THRESH].copy()
fl9["risk"] = fl9.conf * fl9.importance * fl9.pred_class.map(sev_w)
fl9["critical"] = fl9.hazard & (fl9.importance == 3) & fl9.true_class.isin(["undercut_shoulder", "washout", "landslide", "soft_ground"])
def run(order_cols, asc):
    got = 0
    for d, g in fl9.groupby("day"):
        g = g.sort_values(order_cols, ascending=asc).head(NIGHT_CAPACITY); got += int(g.critical.sum())
    return got
crit = int(fl9.critical.sum())
fifo = run(["start_hour_sort", "obs_id"], [True, True]) if False else None
fl9["start_hour_sort"] = np.where(fl9.start_hour >= 12, fl9.start_hour - 24, fl9.start_hour)
fifo = run(["start_hour_sort", "obs_id"], [True, True]); rank = run(["risk"], [False])
curve_r, curve_f = [], []
grp = [g for _, g in fl9.groupby("day")]
for f in np.linspace(0, 1, 21):
    cr = sum(int(g.sort_values("risk", ascending=False).head(int(round(len(g) * f))).critical.sum()) for g in grp)
    cf = sum(int(g.sort_values(["start_hour_sort", "obs_id"]).head(int(round(len(g) * f))).critical.sum()) for g in grp)
    curve_r.append(cr / crit); curve_f.append(cf / crit)
R["uc9"] = dict(flags_per_night=float(len(fl9) / 21), capacity=NIGHT_CAPACITY, critical=crit, fifo=fifo, ranked=rank,
                fifo_share=fifo / crit, ranked_share=rank / crit, curve_r=curve_r, curve_f=curve_f, cap_frac=NIGHT_CAPACITY / (len(fl9) / 21))

# 10 ---- lineage completeness and an example finding
fields = [c for c in lin.columns if c.endswith("_present")]
comp = {c.replace("_present", ""): float(lin[c].mean()) for c in fields}
full = float(lin[fields].all(axis=1).mean())
ex = fl9[(fl9.true_class == "undercut_shoulder") & fl9.hazard & (fl9.route_class == "hospital_route") & (fl9.day >= 6)].sort_values("risk", ascending=False).iloc[0]
exr = rev.set_index("obs_id").loc[ex.obs_id]
R["uc10"] = dict(fields=comp, full=full, n=int(len(lin)), example=dict(
    finding="F" + ex.obs_id[1:], asset=ex.asset_id, route=ex.route_class, terrain=ex.terrain, day=int(ex.day), cls=ex.pred_class, conf=float(ex.confidence),
    model=ex.model_version, threshold=0.25, sortie=ex.sortie_id, band=ex.time_band, team=ex.drone_team, camera=ex.camera, blur=float(ex.blur_score),
    haze=float(ex.haze_index), drop=float(ex.lidar_edge_drop_m), reviewer=exr.reviewer_id, verdict=exr.verdict, risk=round(float(ex.risk), 2)))

R["meta"] = dict(segments=int((assets.asset_kind == "road_segment").sum()), lzs=int((assets.asset_kind == "landing_zone").sum()), days=21,
                 obs=int(len(obs)), sorties=int(len(sorties)), flags=int(len(fl9)), hazards=int(df.hazard.sum()),
                 rows={n: int(len(x)) for n, x in [("assets", assets), ("daily_truth", truth), ("sorties", sorties), ("observations", obs), ("flag_reviews", rev),
                                                   ("annotation_batch", ann), ("training_class_counts", tcount), ("finding_lineage_days1_5", lin)]})
def conv(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    raise TypeError(type(o))
json.dump(R, open(os.path.join(_B, "results.json"), "w"), indent=1, default=conv)
for k, v in R.items():
    print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if not isinstance(vv, (list, dict))})
