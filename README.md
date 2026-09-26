# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline: **normalize → pair-key blocking → LightGBM pairwise matcher → one-S1-per-record assignment with an F0.5-tuned threshold.**

## Data layout

```
dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
```

## Run end-to-end (from this folder)

```bash
pip install -r requirements.txt
python src/normalize.py dataset work          # 1. normalize all 6 source files -> work/*.parquet
python src/run_block.py work train 8          # 2. candidate generation (train)
python src/run_block.py work test 8           # 3. candidate generation (test)
python src/train.py work dataset 8            # 4. train matcher, tune threshold on out-of-fold F0.5
cd src && python predict.py ../work ../output && cd ..   # 5. write output/*.tsv
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Outputs: `output/matching_results.tsv` (leaderboard file) and `output/candidate_pairs.tsv` (blocking set the model scores).

## Method (short)

1. **Normalization** (`normalize.py`) — any script transliterated to ASCII (Unidecode), lowercase, `&`→and,
   digit-for-letter typo repair (`p1atinum`→`platinum`), repeated-letter collapse (`investtmentt`→`investment`),
   legal-suffix-free "core" name, squashed name for website-style names, street/state abbreviation
   canonicalization. Nothing is country-specific in code paths; country is treated as an open string label.
2. **Blocking** (`blocking.py`) — single tokens are heavily reused in this data (e.g. 202 S1 records named
   "Apex Inc"), so the index is built on **pairs of each record's rarest tokens** (name×name, name×address,
   address×address), scoped by country label. Pair keys shared by >60 S1 records are dropped. Each S2/S3
   record keeps its top-8 S1 candidates by summed IDF of shared pair keys.
3. **Matcher** (`features.py`, `train.py`) — 34 features (RapidFuzz token-set / sort / partial / ratio /
   Jaro-Winkler on name, core name, squashed name and address; house-number overlap; blocking score, rank,
   gap to best; name frequency; source flag). LightGBM trained on hard negatives from blocking, 2-fold
   split **grouped by S1 entity**.
4. **Decision** (`predict.py`) — each S2/S3 record matches at most one S1 entity in the ground truth, so
   each record is assigned to its single best-scoring S1 candidate if its probability clears a threshold
   chosen by sweeping out-of-fold macro F0.5.

No external data, APIs or pretrained models are used.
