# AccessMap: Modeling & Retrieval — Status Report (v2)

## What's Done
Notebook 3 runs end to end. Both models retrained after the bias fix. 8 files saved and verified (was 9 — dropped state_encoder, see below).

## Housing Model (US Addresses) — bias fix done
- We went with option 1: dropped `state`, retrained.
- Result: accuracy 85.0% vs. 85.3% baseline. Basically a tie.
- not_ok recall fell to 0.01. The model just predicts "ok" for everything now.
- We checked the raw CSV for more features (house age, repairs, anything). None exist.
- Conclusion: `state` wasn't just bias — it was the only feature with real signal. This dataset can't predict sidewalk condition from house-level features alone. That's a data limitation, and it goes in the report as-is.

## PMR Model (Amsterdam Sidewalks) — precision fix done
- Old problem: precision 0.55 on not_accessible. Model over-flagged.
- First try: moved the decision threshold (0.5 → 0.65). Didn't work. Only 14 of 14,455 predictions sat near the cutoff. The model was confidently wrong, not unsure.
- Real fix: added `curb_height_max` + a missing flag as features. Curb height is half of our label rule, but the model never saw it.
- Result: 100% accuracy. Expected, not impressive — the model now sees both label inputs, so it just reconstructs our own rule. In the report we say "the model recovers our rule-based label," not "we got 100%."

## FAISS Search — unchanged
- Housing: works well. Similar addresses come back ranked and meaningful.
- PMR: still groups correctly but with exact ties. 74% of rows have no curb height, all get the same fill value. Data limit, not a bug.

## What This Means for the Report
- Housing's result is a null finding. That's fine. Documenting "why this dataset can't do it" is a legit bias finding.
- PMR's 100% shows the pipeline is consistent, nothing more.
- Both limits are data problems, not code problems. Say that directly.

## Possible Bonus (if time allows)
Train PMR on only the features OUTSIDE the label rule (crossing, length, width_fill). That answers a real question: can we guess accessibility without measuring width or curb height? This would be the one experiment where the model could actually show skill.

## Next Step
1. Bias write-up (both datasets)
2. README (document what we tried and why)
3. Bonus experiment above, only if time allows
