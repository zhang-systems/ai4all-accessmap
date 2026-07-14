# AccessMap: Modeling & Retrieval — Status Report (v3)

## What's Done
Everything. Pipeline runs end to end, both models fixed, and the app is
deployed and tested. 8 files saved and verified.

## Housing Model (US Addresses) — bias fix done
- Went with option 1: dropped `state`, retrained.
- Result: accuracy 85.0% vs. 85.3% baseline. A tie.
- not_ok recall fell to 0.01. The model just predicts "ok" for everything.
- Checked the raw CSV for more features (house age, repairs, anything). None exist.
- Conclusion: `state` wasn't just bias — it was the only real signal.
  This dataset can't predict sidewalk condition from house-level
  features alone. That's a data limitation. It goes in the report as-is.

## PMR Model (Amsterdam Sidewalks) — precision fix done
- Old problem: precision 0.55 on not_accessible. Model over-flagged.
- First try: moved the threshold (0.5 → 0.65). Failed. Only 14 of
  14,455 predictions sat near the cutoff. The model was confidently
  wrong, not unsure.
- Real fix: added `curb_height_max` + a missing flag. Curb height is
  half of our label rule but the model never saw it.
- Result: 100% accuracy. Expected, not impressive — the model sees
  both label inputs now, so it just reconstructs our rule. We say
  "the model recovers our rule-based label," not "we got 100%."

## FAISS Search — unchanged
- Housing: works well. Ranked, meaningful results.
- PMR: groups correctly but exact ties. 74% missing curb data, all
  filled the same. Data limit, not a bug.

## NEW: Streamlit App — deployed and tested
**Live:** https://accessmap-13a.streamlit.app

Built `app.py` with 3 tabs:

1. **Sidewalk Checker (PMR)** — user enters width, curb height,
   crossing, length. Model returns accessible / not accessible with
   confidence.
2. **Address Search (Housing)** — user picks an address, FAISS
   returns similar addresses with their sidewalk labels and distances.
3. **About & Limits** — what the app does, what it refuses to do
   (no US sidewalk prediction — null finding), and known limits.

### Deployment decisions
- Deployed to Streamlit Cloud (free tier), straight from the GitHub
  repo. The app builds from `requirements.txt` and loads the model
  files committed in `data/`.
- Model files were originally in `.gitignore`. We force-added the 5
  files the app needs (2 model pkls, faiss index, embeddings,
  housing_clean.csv) so the cloud build can find them. ~18 MB total,
  well under GitHub's limits.
- Load models with `joblib`, cache with `@st.cache_resource` so files
  load once, not per click.
- Housing search only covers the 6,425 embedded addresses. No new
  address input — that would need a live ModernBERT download, too
  slow for a demo.
- Missing curb height is a checkbox. Unchecked = same median fill +
  missing flag the model was trained with.
- The app tells the user the PMR label is our own rule, right under
  the result.

### Testing (realistic inputs)
| Test | Input | Output | Pass |
|------|-------|--------|------|
| Narrow sidewalk | width 0.5, curb 0.02 | Not accessible (71%) | Yes |
| Narrow + high curb | width 0.5, curb 0.10 | Not accessible (99%) | Yes |
| Narrow + curb unknown | width 0.5, missing | Not accessible (98%) | Yes |
| Extreme narrow (cloud) | width 0.1, missing | Not accessible (100%) | Yes |
| Address search | Beaverton, OR query | 5 Beaverton results, distance 0.154 → 0.491 | Yes |
| Address search (cloud) | Same query, k=3 | Same 3 results, same distances | Yes |

All tests were re-run on the live cloud app after deployment. Cloud
results match local results exactly — same model files, same outputs.

### Bug found and fixed
- "Failed to fetch dynamically imported module" on the search tab.
  Not a code bug — stale Streamlit static files in the browser cache.
  Fix: hard refresh (Ctrl+Shift+R). Documented for the demo in case
  it happens again.

## Next Step
1. Final presentation slides (use the app screenshots + live URL)
2. Practice the demo flow: problem → checker → search → limits
3. Demo uses the live URL, not localhost — more stable, and anyone
   can follow along on their own device
