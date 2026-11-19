# VinVed One presentations

Three self-contained HTML slide decks for drone-company buyers:

| File | Deck | Slides |
|---|---|---|
| `sales-deck.html` | VinVed One for Drone Companies: From Flight Data to Business Performance | 33 |
| `problem2-data-use-cases.html` | Problem 2 Deep Dive: Making AI Findings Trustworthy, With Data | 18 |
| `ai-flight-school.html` | AI Flight School: Change Management, Training and AI Compliance | 27 |
| `index.html` | Landing page that links to both decks | — |

Each deck is a single file with no external dependencies (fonts, icons and charts are all built in), so it works offline and on any static host.

## Publish on GitHub Pages

1. Create a repository and upload everything in this folder to the root: the three deck files, `index.html`, this README, and the `data/` and `analysis/` folders.
2. Go to **Settings → Pages**, set **Source** to *Deploy from a branch*, choose `main` and `/ (root)`, and save.
3. After a minute the site is live at `https://<your-user>.github.io/<repo>/`.

> GitHub Pages sites are public to anyone with the link. To restrict who can view them, you need GitHub Enterprise Cloud with private Pages, or a host with access control.

## Presenting

| Action | Keys |
|---|---|
| Next / previous slide | → ← · Space · Page Up/Down · swipe on touch screens |
| First / last slide | Home / End |
| Slide overview | O (Esc closes) |
| Speaker notes | N |
| Full screen | F |
| Save as PDF | PDF button, or the browser's Print → Save as PDF |

Every slide has its own link (for example `sales-deck.html#12`), so you can send someone straight to a slide.

## Notes on the content

Figures marked "typical baseline", "illustrative" or "industry-cited" are planning assumptions. Replace them with each client's own data before presenting.

## Problem 2 data and analysis

The Problem 2 deck follows a simulated 21-day flood-relief mission. A drone company surveys 400 km of roads (800 half-km segments) and 60 helicopter landing zones every night for an army engineer battalion, and its AI drafts the 06:00 route map. Every number in the deck can be reproduced.

| Path | What it is |
|---|---|
| `data/assets.csv` | 800 road segments and 60 landing zones: route (main supply, hospital, secondary, local), terrain, importance |
| `data/daily_truth.csv` | The confirmed hazard (or none) on every asset, every night |
| `data/sorties.csv` | 512 drone sorties: night, time band, terrain, team, camera, wind, blur, haze, resolution, overlap |
| `data/observations.csv` | What the AI said for each asset each night: class, confidence, LiDAR edge drop, image flags |
| `data/flag_reviews.csv` | Verdicts (TP/FP) and notes from analysts and engineers for every flag at confidence ≥ 0.50 |
| `data/annotation_batch.csv` | The day-2 labeling batch: 4,000 images, each labeled by three of six annotators |
| `data/training_class_counts.csv` | Training images per class |
| `data/finding_lineage_days1_5.csv` | Six lineage fields, present or missing, for findings released on days 1–5 |
| `analysis/simulate.py` | Rebuilds the data (`python analysis/simulate.py`, seed 7) |
| `analysis/analysis.py` | Runs all ten use cases and writes `analysis/results.json` (`python analysis/analysis.py`; needs pandas and numpy) |

Costs are in mission hours (convoy, crew and analyst time). The assumptions are at the top of `analysis.py`; change them to match a client.
