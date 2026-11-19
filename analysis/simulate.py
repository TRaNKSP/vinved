"""
Simulated data for the Problem 2 deck: a 21-day flood-relief mission.
A drone company surveys 400 km of roads (800 segments of 0.5 km) and 60 helicopter landing zones every night
for an army engineer battalion. All data is synthetic.  Run:  python simulate.py  -> writes data/*.csv
"""
import os
import numpy as np, pandas as pd

rng = np.random.default_rng(7)
_B = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_B, "..", "data") if os.path.isdir(os.path.join(_B, "..", "data")) else os.path.join(_B, "data")
os.makedirs(OUT, exist_ok=True)

DAYS = 21
ROAD_H = ["debris", "standing_water", "landslide", "washout", "undercut_shoulder"]
LZ_H = ["soft_ground", "obstruction"]
TRAIN = {"healthy_road": 12000, "debris": 2400, "standing_water": 1800, "landslide": 900, "washout": 700,
         "undercut_shoulder": 300, "firm_ground": 3000, "soft_ground": 250, "obstruction": 600}

# ---------------- road segments and landing zones ----------------
n_seg = 800
route = np.array(["main_supply_route"] * 120 + ["hospital_route"] * 60 + ["secondary"] * 260 + ["local"] * 360)
terrain = np.where(np.isin(route, ["main_supply_route", "hospital_route"]),
                   rng.choice(["valley", "ridge", "plain"], n_seg, p=[0.65, 0.20, 0.15]),
                   rng.choice(["valley", "ridge", "plain"], n_seg, p=[0.25, 0.28, 0.47]))
imp = {"main_supply_route": 3, "hospital_route": 3, "secondary": 2, "local": 1}
segs = pd.DataFrame({"asset_id": [f"RS{i:04d}" for i in range(1, n_seg + 1)], "asset_kind": "road_segment", "route_class": route,
                     "terrain": terrain, "importance": [imp[r] for r in route], "length_km": 0.5})
segs["lat"] = (35.2 + rng.normal(0, 0.15, n_seg)).round(5); segs["lon"] = (69.1 + rng.normal(0, 0.2, n_seg)).round(5)
lzs = pd.DataFrame({"asset_id": [f"LZ{i:02d}" for i in range(1, 61)], "asset_kind": "landing_zone", "route_class": "medevac_lz",
                    "terrain": rng.choice(["valley", "ridge", "plain"], 60, p=[0.5, 0.2, 0.3]), "importance": rng.choice([2, 3], 60, p=[0.4, 0.6]), "length_km": 0.0})
lzs["lat"] = (35.2 + rng.normal(0, 0.15, 60)).round(5); lzs["lon"] = (69.1 + rng.normal(0, 0.2, 60)).round(5)
assets = pd.concat([segs, lzs], ignore_index=True)

# ---------------- true hazard state, day by day ----------------
state = {}  # asset_id -> (class, severity) or None
def init_state(a):
    if a.asset_kind == "road_segment":
        p = 0.16 if a.terrain == "valley" else 0.10
        if rng.random() < p:
            return (rng.choice(ROAD_H, p=[0.30, 0.25, 0.12, 0.13, 0.20]), int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])))
    else:
        if rng.random() < 0.22: return ("soft_ground", int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])))
        if rng.random() < 0.05: return ("obstruction", 1)
    return None
for a in assets.itertuples(): state[a.asset_id] = init_state(a)
kinds = dict(zip(assets.asset_id, assets.asset_kind)); terr = dict(zip(assets.asset_id, assets.terrain))
truth = []
for d in range(1, DAYS + 1):
    if d > 1:
        for a, s in list(state.items()):
            if s is None:
                if kinds[a] == "road_segment" and rng.random() < 0.004:
                    state[a] = (rng.choice(ROAD_H, p=[0.35, 0.15, 0.15, 0.10, 0.25]), int(rng.choice([1, 2], p=[0.7, 0.3])))
                continue
            c, sev = s
            clear_p = {"debris": 0.16, "standing_water": 0.20, "landslide": 0.10, "washout": 0.05, "undercut_shoulder": 0.04, "soft_ground": 0.08, "obstruction": 0.20}[c]
            if rng.random() < clear_p: state[a] = None; continue
            if c == "undercut_shoulder" and rng.random() < 0.04: state[a] = ("washout", 3); continue
            if rng.random() < 0.05 and sev < 3: state[a] = (c, sev + 1)
    for a, s in state.items():
        truth.append((d, a, s[0] if s else ("healthy_road" if kinds[a] == "road_segment" else "firm_ground"), s[1] if s else 0))
truth = pd.DataFrame(truth, columns=["day", "asset_id", "true_class", "true_severity"])

# ---------------- sorties and model observations ----------------
q_class = {c: min(1.0, 0.15 + 0.85 * np.log(n) / np.log(2400)) for c, n in TRAIN.items()}
q_class["undercut_shoulder"] *= 0.62     # effect of the mislabeled training batch (use case 1)
obs, sorties = [], []
sid = 0
for d in range(1, DAYS + 1):
    t = truth[truth.day == d].merge(assets[["asset_id", "asset_kind", "terrain", "route_class"]], on="asset_id")
    band = np.where(t.terrain == "valley", rng.choice(["Dawn", "Night"], len(t), p=[0.6, 0.4]),
                    np.where(t.terrain == "ridge", rng.choice(["Dawn", "Night"], len(t), p=[0.3, 0.7]), rng.choice(["Dawn", "Night"], len(t), p=[0.5, 0.5])))
    t = t.assign(band=band).sort_values(["band", "terrain", "asset_id"], ascending=[False, True, True])  # Night first, Dawn last
    seq = 0
    for (b, ter), grp in t.groupby(["band", "terrain"], sort=False):
        for k in range(0, len(grp), 40):
            sid += 1; seq += 1
            team = "Team B" if (d >= 8 and rng.random() < 0.4) else "Team A"
            wind = float(np.clip(rng.gamma(2.2, 2.6 if ter == "ridge" else 2.0), 0, 18))
            blur = float(np.clip(0.08 + 0.045 * wind + rng.normal(0, 0.08), 0, 1))
            gsd = float(np.clip(rng.normal(1.45, 0.22), 0.9, 2.4))
            overlap = float(rng.uniform(58, 85))
            haze = float(np.clip(rng.beta(2, 5) + (0.25 if (ter == "valley" and b == "Dawn") else 0), 0, 1))
            start_h = 22.0 + seq * 0.33 if b == "Night" else 4.5 + rng.uniform(0, 0.8)
            sorties.append((f"S{sid:05d}", d, b, ter, team, "CAM-2" if team == "Team B" else "CAM-1", round(wind, 1), round(blur, 2), round(gsd, 2), round(overlap), round(haze, 2), round(start_h % 24, 2), seq))
            for r in grp.iloc[k:k + 40].itertuples():
                hazard = r.true_class not in ("healthy_road", "firm_ground")
                qpen = 1.0 - 1.4 * max(0, blur - 0.45) - 0.6 * max(0, gsd - 1.7) - 0.6 * max(0, haze - 0.55)
                if b == "Dawn": qpen -= 0.08
                if b == "Dawn" and ter == "valley" and r.true_class in ("soft_ground", "undercut_shoulder"): qpen -= 0.38
                if team == "Team B": qpen -= 0.04
                sheen = bool(r.asset_kind == "road_segment" and b == "Dawn" and d <= 12 and rng.random() < 0.22)
                shadow = bool(b == "Dawn" and rng.random() < (0.30 if ter == "valley" else 0.15))
                rocks = bool(ter == "ridge" and rng.random() < 0.10)
                dup = bool(rng.random() < 0.04)
                drop = round(float(rng.uniform(0.9, 2.2) if r.true_class in ("undercut_shoulder", "washout") else abs(rng.normal(0.1, 0.12))), 2) if r.asset_kind == "road_segment" else None
                if hazard:
                    q = float(np.clip(q_class[r.true_class] * qpen, 0.05, 1))
                    conf = rng.beta(0.8 + 5.5 * q, 2.3)
                    pool = ROAD_H if r.asset_kind == "road_segment" else LZ_H
                    pred = r.true_class if rng.random() < 0.9 else rng.choice([c for c in pool if c != r.true_class] or pool)
                else:
                    pfp = 0.012 * (4 if sheen else 1) * (3 if shadow else 1) * (3 if rocks else 1) * (2 if haze > 0.6 else 1) * (2.6 if team == "Team B" else 1)
                    if rng.random() < pfp:
                        conf = rng.beta(3.4, 2.2) if team == "Team B" else rng.beta(2.4, 2.6)
                        if r.asset_kind == "landing_zone": pred = "soft_ground" if shadow or rng.random() < 0.7 else "obstruction"
                        elif sheen: pred = "standing_water"
                        elif rocks: pred = rng.choice(["debris", "landslide"])
                        elif shadow: pred = rng.choice(["debris", "undercut_shoulder"], p=[0.7, 0.3])
                        else: pred = rng.choice(ROAD_H)
                    elif rng.random() < 0.07:
                        conf = rng.beta(1.6, 5.5); pred = rng.choice(ROAD_H if r.asset_kind == "road_segment" else LZ_H)
                    else:
                        conf, pred = None, ("healthy_road" if r.asset_kind == "road_segment" else "firm_ground")
                if conf is not None and conf < 0.2:
                    conf, pred = None, ("healthy_road" if r.asset_kind == "road_segment" else "firm_ground")
                obs.append((f"O{len(obs) + 1:06d}", f"S{sid:05d}", d, r.asset_id, r.true_class, r.true_severity, pred,
                            None if conf is None else round(float(conf), 3), "route-ai-v2.0" if d < 4 else "route-ai-v2.1", drop, sheen, shadow, rocks, dup))
sorties = pd.DataFrame(sorties, columns=["sortie_id", "day", "time_band", "terrain", "drone_team", "camera", "wind_mps", "blur_score", "gsd_cm", "overlap_pct", "haze_index", "start_hour", "night_sequence"])
obs = pd.DataFrame(obs, columns=["obs_id", "sortie_id", "day", "asset_id", "true_class", "true_severity", "pred_class", "confidence", "model_version",
                                 "lidar_edge_drop_m", "wet_sheen_flag", "shadow_flag", "rock_outcrop_flag", "duplicate_flag"])

# ---------------- verdicts on every flag (analyst at night, or engineer recon next day) ----------------
flags = obs[obs.confidence.notna() & (obs.confidence >= 0.5)].merge(sorties[["sortie_id", "haze_index", "blur_score"]], on="sortie_id")
notes_by = {"Wet asphalt sheen": ["Wet asphalt sheen, road dry underneath", "Reflection on wet surface, no water depth on LiDAR"],
            "Tree shadow": ["Tree shadow across road, no debris", "Shadow line read as edge drop"],
            "Rock outcrop": ["Natural rock outcrop beside road, not debris", "Old rockfall, already cleared, road open"],
            "Fog / haze": ["Haze obscured frame, nothing found on recheck", "Fog patch, image unreadable"],
            "Duplicate tile": ["Duplicate tile, hazard already logged", "Same item as previous frame"],
            "Road patch / marking": ["Fresh asphalt patch read as damage", "Painted road marking read as edge crack"],
            "Other": ["Nothing found on recheck", "Unclear, road appears normal"]}
v, nt, who = [], [], []
for r in flags.itertuples():
    hz = r.true_class not in ("healthy_road", "firm_ground")
    if r.duplicate_flag and rng.random() < 0.6: cause = "Duplicate tile"
    elif hz: cause = None
    elif r.wet_sheen_flag: cause = "Wet asphalt sheen"
    elif r.shadow_flag: cause = "Tree shadow"
    elif r.rock_outcrop_flag: cause = "Rock outcrop"
    elif r.haze_index > 0.6: cause = "Fog / haze"
    else: cause = rng.choice(["Road patch / marking", "Other"], p=[0.65, 0.35])
    v.append("TP" if cause is None else "FP"); nt.append("Hazard confirmed" if cause is None else rng.choice(notes_by[cause]))
    who.append(f"AN-{rng.integers(1, 7):02d}" if rng.random() < 0.6 else f"ENG-{rng.integers(1, 5):02d}")
reviews = pd.DataFrame({"obs_id": flags.obs_id, "reviewer_id": who, "verdict": v, "review_minutes": np.clip(rng.normal(5, 1.3, len(flags)), 2, 10).round(1), "review_notes": nt})

# ---------------- new labeling batch: 4,000 images x 3 annotators ----------------
batch = {"healthy_road": 1600, "undercut_shoulder": 300, "soft_ground": 250, "washout": 350, "debris": 600, "standing_water": 500, "landslide": 250, "obstruction": 150}
true_b = rng.permutation(sum([[c] * n for c, n in batch.items()], []))
acc = {"ANN-01": 0.96, "ANN-02": 0.95, "ANN-03": 0.94, "ANN-04": 0.93, "ANN-05": 0.97, "ANN-06": 0.86}
conf_to = {"undercut_shoulder": "healthy_road", "soft_ground": "healthy_road", "washout": "undercut_shoulder", "debris": "landslide",
           "standing_water": "healthy_road", "landslide": "debris", "obstruction": "debris", "healthy_road": "debris"}
ann = []
for i, t in enumerate(true_b):
    for a in rng.choice(list(acc), 3, replace=False):
        p = acc[a]
        if a == "ANN-06" and t == "undercut_shoulder": p = 0.22      # the guideline gap: undercut read as healthy road
        lab = t if rng.random() < p else (conf_to[t] if rng.random() < 0.75 else rng.choice([c for c in batch if c != t]))
        ann.append((f"NB{i + 1:05d}", a, lab))
ann = pd.DataFrame(ann, columns=["image_id", "annotator_id", "label"])
train_counts = pd.DataFrame({"class": list(TRAIN), "training_images": list(TRAIN.values())})

# ---------------- released findings and their lineage, days 1-5 (before TRACE) ----------------
rel = flags.merge(reviews, on="obs_id").query("verdict == 'TP' and day <= 5")
pres = {"source_image": 0.94, "lidar_scan": 0.71, "model_version": 0.58, "threshold_used": 0.33, "training_batch": 0.12, "reviewer_id": 0.82}
lin = pd.DataFrame({"finding_id": [f"F{x[1:]}" for x in rel.obs_id], "day": rel.day.values})
for k, p in pres.items(): lin[k + "_present"] = rng.random(len(lin)) < p

for name, df in [("assets", assets), ("daily_truth", truth), ("sorties", sorties), ("observations", obs), ("flag_reviews", reviews),
                 ("annotation_batch", ann), ("training_class_counts", train_counts), ("finding_lineage_days1_5", lin)]:
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    print(f"{name:26s} {len(df):>7,} rows")
