# AccessMap: Sidewalk Accessibility Modeling & Retrieval

**Live demo:** https://accessmap-13a.streamlit.app

## Team — Group 13A

Zheng Zhang, Zaina Nadeem, Laxmi Pesara, Gavino Vargas, Ashton Moraes, Amen Bush  

---

## Part 1 — The Problem & Why It Matters

Wheelchair users can't tell if a sidewalk is passable before they get
there. Too narrow, curb too high — they find out on the spot. A wrong
route isn't an inconvenience, it can end a trip.

There's no simple way to check ahead. Google Maps tells you the road
exists. It doesn't tell you if a wheelchair can use it.

So the question we set out to answer: **can data predict which
sidewalks are accessible?**

Why it's worth doing:
- Real users with a real pain point (wheelchair users, caregivers, seniors)
- Cities collect sidewalk data but rarely turn it into anything usable
- If the model works, it plugs into any routing app as a scoring layer

### The Data

Two datasets, two angles on the same question:

| Dataset | What it is | Size |
|---------|-----------|------|
| PMR | Amsterdam sidewalk measurements (width, curb height, crossings) | 72,274 segments |
| Housing | US addresses with crowd-sourced "sidewalk ok?" labels from Google Maps images | 10,104 rows |

PMR measures the physical world. Housing measures what people see.
They don't overlap geographically, so we run the same pipeline on
each and compare results. We never merge rows.

### What We Built

Three notebooks and one app:

1. **01 — PMR cleaning + EDA**
2. **02 — Housing cleaning + EDA + bias audit**
3. **03 — Modeling (Random Forest) + embeddings (ModernBERT) + search (FAISS)**
4. **app.py — Streamlit app** that loads the trained models so anyone
   can use them without touching a notebook

Pipeline outputs: 2 models, 3 encoders/scalers, 1 embedding file,
2 FAISS indexes. 8 files, all verified at export. The app loads these
directly — no retraining.

---

## Part 2 — What We Tried, Where We Failed, How We Iterated

We didn't get it right the first time. Here's the actual path.

### V1: Baseline models — two problems show up

- **PMR**: 84% accuracy vs. 81% baseline. Looked fine on the surface.
  But precision on "not accessible" was only 0.55 — the model
  over-flagged good sidewalks as bad.
- **Housing**: 67% accuracy vs. 85% baseline. Worse than guessing.
  One feature (`state`) was 89% of feature importance. The model was
  learning geography, not house condition. A bias problem.

### V2: Fixing the Housing bias

Three options on the table:
1. Drop `state` and retrain
2. Keep `state`, relabel the model as "predicts by region"
3. Swap algorithms

We picked option 1 because it's diagnostic, not just a fix. Dropping
`state` tells us WHY the model is biased: if accuracy holds, the
signal was there and `state` was drowning it out. If accuracy drops,
the signal never existed.

**The pit we fell into:** accuracy came back at 85.0% vs. 85.3%
baseline — a tie. But recall on bad sidewalks fell to 0.01. The model
just says "ok" to everything now. It looked like the fix worked
(accuracy went "up" from 67% to 85%), but it actually revealed the
model has nothing left to learn from.

We went back to the raw CSV hunting for more features. House age,
repair records, anything. None exist. The dataset has house type, two
confidence columns, and location. That's it.

### V3: Fixing the PMR precision

**First try — move the threshold (failed).** We raised the
"not accessible" decision cutoff from 0.5 to 0.65. Nothing changed.
Precision stayed at 0.55, same as before.

**Why it failed:** we checked the probability distribution. Only 14
of 14,455 test predictions sat anywhere near the cutoff. The model's
probabilities were almost all pinned at 0 or 1. It wasn't unsure — it
was confidently wrong. A threshold can't fix confident mistakes.

**Second try — fix the features (worked).** Our label rule is built
from width + curb height. But the model only ever saw width. It was
guessing the curb half blind. We added `curb_height_max` and a
missing-value flag (74% of rows have no curb measurement — the
missingness itself is information).

### Retrieval (FAISS)

- **Housing**: works well. Query an address, get similar addresses
  ranked by distance.
- **PMR**: groups segments with identical accessibility profiles, but
  many exact ties — most segments share the same width/curb values
  (standardized construction + 74% missing curb data, all filled with
  the same median). It's a group finder, not a ranker. We tried adding
  more attribute features; ties stayed. It's a data limit, not a bug.

### V4: Deployment (Streamlit app)

We turned the models into a user-facing app with 3 tabs:

1. **Sidewalk Checker (PMR)** — enter width, curb height, crossing,
   length. Get accessible / not accessible with a confidence score.
2. **Address Search (Housing)** — pick an address, FAISS returns
   similar addresses with their sidewalk labels and distances.
3. **About & Limits** — what the app does, what it refuses to do,
   and why.

**Deployment decisions we made:**
- The app does NOT predict sidewalk condition for US addresses. We
  tested that model — it has no signal (see V2). It reports the crowd
  label instead of pretending to predict. Honesty is a feature.
- Search only covers the 6,425 embedded addresses. New-address input
  would need a live ModernBERT download — too slow for a demo.
- Missing curb height is a checkbox. Unchecked = the same median fill
  + missing flag the model was trained with. App inputs match
  training inputs exactly.
- Models load once with `@st.cache_resource`, not on every click.

**Testing with realistic inputs (all passed):**

| Test | Input | Output |
|------|-------|--------|
| Narrow sidewalk | width 0.5, curb 0.02 | Not accessible (71%) |
| Narrow + high curb | width 0.5, curb 0.10 | Not accessible (99%) |
| Curb not measured | width 0.5, missing | Not accessible (98%) |
| Address search | Beaverton, OR query | 5 Beaverton results, distance 0.154 → 0.491 |

**Bug we hit:** "Failed to fetch dynamically imported module" on the
search tab. Not a code bug — stale Streamlit files in the browser
cache. Fix: hard refresh (Ctrl+Shift+R).

---

## Part 3 — Results & What We Learned

### Final results

| Model | Accuracy | Baseline | What it really means |
|-------|----------|----------|----------------------|
| PMR | 100% | 80.8% | Model reconstructs our own label rule. Proves the pipeline is consistent — nothing more. |
| Housing | 85.0% | 85.3% | A null finding. Without `state`, no signal is left. |

Neither number should be read at face value:
- **PMR's 100% is expected, not impressive.** The model sees both
  inputs of the rule we wrote, so it recovers the rule. We report it
  as "the model recovers our rule-based label," never "we achieved 100%."
- **Housing's 85% is the majority-class rate.** The honest conclusion:
  this dataset can't predict sidewalk condition from house-level
  features alone. `state` wasn't just bias — it was the only real
  signal, even if that signal was geography.

### What we learned (including from the failures)

1. **Check the probability distribution before tuning a threshold.**
   If the model isn't unsure, a threshold has nothing to fix. We
   wasted a step by not checking first.
2. **A dominant feature isn't always removable.** Dropping `state`
   confirmed the bias diagnosis AND removed the only signal. Both
   facts go in the report.
3. **A null result is a result.** "This dataset can't do it, and
   here's why" is a legitimate finding for a bias-focused project.
   We'd rather report that than hide it behind a prettier number.
4. **Perfect accuracy should make you suspicious.** Ours came from
   the model seeing its own label rule. Always ask where the label
   came from before celebrating.

### Limitations

- PMR's label is our own rule (width ≥ 0.9m, curb ≤ 0.06m), not
  verified ground truth. Only 50 real labels exist — too few to train on.
- Housing's labels are crowd judgments from photos, not measurements.
- Neither model transfers to the other dataset or region.
- PMR retrieval can't rank within tied groups without location data.

### If we had more time

- Train PMR on only the features OUTSIDE the label rule (crossing,
  length, width_fill). That answers a real question: can you guess
  accessibility without measuring width or curb? The one experiment
  where the model could show actual skill.
- Get real accessibility labels for a sample of PMR segments and
  validate our rule against them.
- Add coordinates to PMR retrieval for location-aware search.

---

## How to Run

**Pipeline (run once):**
```
1. Run notebooks in order: 01 → 02 → 03
2. Notebook 03 Step 6 downloads ModernBERT from Hugging Face (needs internet)
3. Outputs land in data/processed/ and data/models/
```

**App (after the pipeline):**
```
pip install streamlit
streamlit run app.py
```
Opens at http://localhost:8501. If a tab shows a
"Failed to fetch module" error, hard refresh (Ctrl+Shift+R).

Requirements: pandas, geopandas, scikit-learn, sentence-transformers, faiss, streamlit
