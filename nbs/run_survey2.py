"""Resume from saved gene_expr.pkl — run bootstrap survey, save as CSV."""
import pickle, pathlib
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

DATA_DIR = pathlib.Path("gtex_data")

BRAIN_REGIONS = {
    "Hypothalamus": "Brain - Hypothalamus", "Amygdala": "Brain - Amygdala",
    "Hippocampus": "Brain - Hippocampus", "Ant. Cing. Ctx": "Brain - Anterior cingulate cortex (BA24)",
    "Frontal Cortex": "Brain - Frontal Cortex (BA9)", "Cortex": "Brain - Cortex",
    "Caudate": "Brain - Caudate (basal ganglia)", "Putamen": "Brain - Putamen (basal ganglia)",
    "Nucleus Accumbens": "Brain - Nucleus accumbens (basal ganglia)",
    "Cerebellum": "Brain - Cerebellum", "Cerebellar Hemi.": "Brain - Cerebellar Hemisphere",
    "Substantia Nigra": "Brain - Substantia nigra", "Spinal Cord": "Brain - Spinal cord (cervical c-1)",
}

print("Loading gene_expr.pkl...", flush=True)
with open(DATA_DIR / "gene_expr.pkl", "rb") as f:
    gene_expr = pickle.load(f)
print(f"Loaded {len(gene_expr)} genes.", flush=True)

pheno = pd.read_csv(DATA_DIR / "GTEx_SubjectPhenotypes.txt", sep="\t")
sex_map = pheno.set_index("SUBJID")["SEX"].map({1: "Male", 2: "Female"})
attr = pd.read_csv(DATA_DIR / "GTEx_SampleAttributes.txt", sep="\t", usecols=["SAMPID", "SMTSD"])
attr["SUBJID"] = attr["SAMPID"].str.extract(r"(GTEX-[^-]+)")
attr["SEX"] = attr["SUBJID"].map(sex_map)
region_rev = {v: k for k, v in BRAIN_REGIONS.items()}
brain_attr = attr[attr["SMTSD"].isin(set(BRAIN_REGIONS.values())) & attr["SEX"].notna()].copy()
brain_attr["region"] = brain_attr["SMTSD"].map(region_rev)
sample_meta = brain_attr.set_index("SAMPID")[["region", "SEX"]].rename(columns={"SEX": "sex"})
sample_meta.to_csv(DATA_DIR / "sample_meta.csv")
print(f"sample_meta saved ({len(sample_meta)} samples)", flush=True)

RNG = np.random.default_rng(42)
N_BOOT = 2000

def _bmd(f, m):
    f, m = np.asarray(f), np.asarray(m)
    return (RNG.choice(f, (N_BOOT, len(f)), replace=True).mean(1)
          - RNG.choice(m, (N_BOOT, len(m)), replace=True).mean(1))

def _bimod(x):
    n = len(x); g = skew(x); k = kurtosis(x, fisher=True)
    return (g**2 + 1) / (k + 3*(n-1)**2/((n-2)*(n-3)))

rows = []
genes = list(gene_expr.keys())
for gi, gene in enumerate(genes):
    gd = gene_expr[gene]
    for region in BRAIN_REGIONS:
        mask = sample_meta["region"] == region
        sids = sample_meta.index[mask]
        f_vals = [gd[s] for s in sids if s in gd and sample_meta.loc[s,"sex"]=="Female"]
        m_vals = [gd[s] for s in sids if s in gd and sample_meta.loc[s,"sex"]=="Male"]
        if len(f_vals) < 5 or len(m_vals) < 5:
            continue
        boots = _bmd(f_vals, m_vals)
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(gene=gene, region=region, n_f=len(f_vals), n_m=len(m_vals),
                         mean_md=boots.mean(), ci_lo=ci_lo, ci_hi=ci_hi,
                         ci_width=ci_hi-ci_lo, bimod=_bimod(boots),
                         skewness=float(skew(boots)), crosses_zero=int(ci_lo<0<ci_hi)))
    print(f"  {gi+1}/{len(genes)}: {gene}    ", end="\r", flush=True)

survey = pd.DataFrame(rows)
survey.to_csv(DATA_DIR / "survey.csv", index=False)
print(f"\nSurvey done: {len(survey)} cells → gtex_data/survey.csv", flush=True)

gs = survey.groupby("gene").agg(
    mean_effect=("mean_md","mean"), max_ci_width=("ci_width","max"),
    max_bimod=("bimod","max"), n_crosses=("crosses_zero","sum"),
).sort_values("mean_effect").round(3)

print("\n── Strongly negative F−M (M>>F) ──")
print(gs[gs["mean_effect"] < -1].to_string())
print("\n── Strongly positive F−M (F>>M) ──")
print(gs[gs["mean_effect"] > 0.5].to_string())
print("\n── Wide CI, near-zero mean ──")
print(gs[(gs["mean_effect"].abs()<0.3)&(gs["max_ci_width"]>0.5)].sort_values("max_ci_width",ascending=False).to_string())
print("\n── CI crosses zero in ≥8 regions ──")
print(gs[gs["n_crosses"]>=8].sort_values("mean_effect").to_string())
print("\n── Highest bimodality ──")
print(gs.sort_values("max_bimod",ascending=False).head(25).to_string())
