"""GTEx survey script — runs outside Jupyter, saves results to survey.parquet."""
import os, gzip, pathlib, urllib.request, pickle
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

DATA_DIR = pathlib.Path("gtex_data")
DATA_DIR.mkdir(exist_ok=True)

TPM_GZ     = DATA_DIR / "GTEx_gene_tpm.gct.gz"
ATTR_FILE  = DATA_DIR / "GTEx_SampleAttributes.txt"
PHENO_FILE = DATA_DIR / "GTEx_SubjectPhenotypes.txt"

GTEX_BASE      = "https://storage.googleapis.com/adult-gtex"
GTEX_TPM_URL   = (f"{GTEX_BASE}/bulk-gex/v8/rna-seq/"
                   "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz")
GTEX_ATTR_URL  = (f"{GTEX_BASE}/annotations/v8/metadata-files/"
                   "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt")
GTEX_PHENO_URL = (f"{GTEX_BASE}/annotations/v8/metadata-files/"
                   "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt")

BRAIN_REGIONS = {
    "Hypothalamus":      "Brain - Hypothalamus",
    "Amygdala":          "Brain - Amygdala",
    "Hippocampus":       "Brain - Hippocampus",
    "Ant. Cing. Ctx":    "Brain - Anterior cingulate cortex (BA24)",
    "Frontal Cortex":    "Brain - Frontal Cortex (BA9)",
    "Cortex":            "Brain - Cortex",
    "Caudate":           "Brain - Caudate (basal ganglia)",
    "Putamen":           "Brain - Putamen (basal ganglia)",
    "Nucleus Accumbens": "Brain - Nucleus accumbens (basal ganglia)",
    "Cerebellum":        "Brain - Cerebellum",
    "Cerebellar Hemi.":  "Brain - Cerebellar Hemisphere",
    "Substantia Nigra":  "Brain - Substantia nigra",
    "Spinal Cord":       "Brain - Spinal cord (cervical c-1)",
}

CANDIDATE_GENES = [
    "XIST", "RPS4Y1", "DDX3Y", "KDM5D", "USP9Y", "EIF1AY", "KDM6A", "KDM5C", "MECP2",
    "ESR1", "ESR2", "AR", "PGR", "CYP19A1", "NR3C1", "NR3C2", "FKBP5",
    "DRD1", "DRD2", "DRD3", "DRD4", "DRD5", "SLC6A3", "TH", "DDC", "COMT", "MAOB",
    "HTR1A", "HTR1B", "HTR2A", "HTR2B", "HTR2C", "HTR3A", "SLC6A4", "TPH2", "MAOA",
    "GABRA1", "GABRA2", "GABRA3", "GABRA4", "GABRA5",
    "GABRB1", "GABRB2", "GABRB3", "GABRG2", "GAD1", "GAD2",
    "GRIA1", "GRIA2", "GRIA3", "GRIN1", "GRIN2A", "GRIN2B", "GRM1", "GRM5",
    "CHRM1", "CHRM2", "CHRNA4", "CHRNA7",
    "ADRA1A", "ADRA2A", "SLC6A2",
    "OPRM1", "OPRD1", "OPRK1", "PENK", "PDYN",
    "CNR1",
    "NPY", "VIP", "SST", "CRH", "CRHR1", "CRHR2", "NTS", "CARTPT", "POMC", "AGRP",
    "OXTR", "AVPR1A", "AVPR1B", "KISS1", "KISS1R",
    "BDNF", "NTRK2", "NGF", "NTF3",
    "NRXN1", "NRXN2", "NRXN3", "NLGN1", "NLGN3", "SHANK2", "SHANK3",
    "SYP", "SYN1", "SNAP25",
    "MBP", "PLP1", "GFAP", "AIF1", "S100B", "MOG",
    "IL6", "TNF", "IL1B", "TGFB1",
    "CLOCK", "ARNTL", "PER1", "PER2", "CRY1", "CRY2",
    "APP", "MAPT", "SNCA", "LRRK2", "APOE", "CLU",
    "FMR1", "PTEN",
    "ACTB", "GAPDH", "RPL19", "UBB", "HPRT1", "B2M",
]

# ── Download ──────────────────────────────────────────────────────────────────
def _download(url, dest, label):
    if dest.exists():
        print(f"cached: {dest}", flush=True)
        return
    print(f"downloading {label}...", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"done ({dest.stat().st_size/1e6:.0f} MB)", flush=True)

_download(GTEX_PHENO_URL, PHENO_FILE, "subject phenotypes")
_download(GTEX_ATTR_URL,  ATTR_FILE,  "sample attributes")
_download(GTEX_TPM_URL,   TPM_GZ,     "gene TPM (~2 GB)")

# ── Metadata ──────────────────────────────────────────────────────────────────
pheno = pd.read_csv(PHENO_FILE, sep="\t")
sex_map = pheno.set_index("SUBJID")["SEX"].map({1: "Male", 2: "Female"})

attr = pd.read_csv(ATTR_FILE, sep="\t", usecols=["SAMPID", "SMTSD"])
attr["SUBJID"] = attr["SAMPID"].str.extract(r"(GTEX-[^-]+)")
attr["SEX"]    = attr["SUBJID"].map(sex_map)

target_tissues = set(BRAIN_REGIONS.values())
brain_attr = attr[attr["SMTSD"].isin(target_tissues) & attr["SEX"].notna()].copy()
region_rev = {v: k for k, v in BRAIN_REGIONS.items()}
brain_attr["region"] = brain_attr["SMTSD"].map(region_rev)
sample_meta = brain_attr.set_index("SAMPID")[["region", "SEX"]].rename(columns={"SEX": "sex"})

print(f"Brain samples: {len(sample_meta)}", flush=True)
counts = sample_meta.groupby(["region", "sex"]).size().unstack(fill_value=0)
print(counts.to_string(), flush=True)

# ── Stream TPM ────────────────────────────────────────────────────────────────
target_genes = set(CANDIDATE_GENES)
brain_samples = set(sample_meta.index)
gene_expr = {}

print("Streaming TPM file...", flush=True)
with gzip.open(TPM_GZ, "rt") as fh:
    fh.readline(); fh.readline()
    header = fh.readline().rstrip().split("\t")
    all_samples = header[2:]
    brain_col_idx  = [i for i, s in enumerate(all_samples) if s in brain_samples]
    brain_col_names = [all_samples[i] for i in brain_col_idx]
    found = 0
    for line in fh:
        parts = line.rstrip().split("\t")
        gene = parts[1]
        if gene not in target_genes:
            continue
        tpm = np.array([float(parts[2 + i]) for i in brain_col_idx])
        gene_expr[gene] = dict(zip(brain_col_names, np.log2(tpm + 1)))
        found += 1
        print(f"  {found}/{len(target_genes)}: {gene}      ", end="\r", flush=True)
        if found == len(target_genes):
            break

print(f"\nFound {len(gene_expr)} genes.", flush=True)
missing = target_genes - set(gene_expr)
if missing:
    print(f"Not in GTEx: {sorted(missing)}", flush=True)

# save extracted expression for reuse in notebook
with open(DATA_DIR / "gene_expr.pkl", "wb") as f:
    pickle.dump(gene_expr, f)
sample_meta.to_parquet(DATA_DIR / "sample_meta.parquet")
print("Saved gene_expr.pkl and sample_meta.parquet", flush=True)

# ── Bootstrap survey ──────────────────────────────────────────────────────────
RNG    = np.random.default_rng(42)
N_BOOT = 2000

def _bootstrap_md(f, m):
    f, m = np.asarray(f), np.asarray(m)
    return (RNG.choice(f, (N_BOOT, len(f)), replace=True).mean(1)
          - RNG.choice(m, (N_BOOT, len(m)), replace=True).mean(1))

def _bimod(x):
    n = len(x); g = skew(x); k = kurtosis(x, fisher=True)
    return (g**2 + 1) / (k + 3*(n-1)**2/((n-2)*(n-3)))

rows = []
boot_cache = {}
for gene in CANDIDATE_GENES:
    if gene not in gene_expr:
        continue
    gd = gene_expr[gene]
    for region in BRAIN_REGIONS:
        mask  = sample_meta["region"] == region
        sids  = sample_meta.index[mask]
        f_vals = [gd[s] for s in sids if s in gd and sample_meta.loc[s,"sex"]=="Female"]
        m_vals = [gd[s] for s in sids if s in gd and sample_meta.loc[s,"sex"]=="Male"]
        if len(f_vals) < 5 or len(m_vals) < 5:
            continue
        boots = _bootstrap_md(f_vals, m_vals)
        boot_cache[(gene, region)] = boots
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(
            gene=gene, region=region,
            n_f=len(f_vals), n_m=len(m_vals),
            mean_md=boots.mean(), ci_lo=ci_lo, ci_hi=ci_hi,
            ci_width=ci_hi-ci_lo,
            bimod=_bimod(boots),
            skewness=float(skew(boots)),
            crosses_zero=int(ci_lo<0<ci_hi),
        ))

survey = pd.DataFrame(rows)
survey.to_parquet(DATA_DIR / "survey.parquet")
print(f"Survey done: {len(survey)} cells. Saved to gtex_data/survey.parquet", flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────
gs = (survey.groupby("gene").agg(
    mean_effect=("mean_md","mean"),
    max_ci_width=("ci_width","max"),
    max_bimod=("bimod","max"),
    n_crosses=("crosses_zero","sum"),
).sort_values("mean_effect").round(3))

print("\n── Strongly negative F−M (M>>F) ──")
print(gs[gs["mean_effect"] < -1].to_string())
print("\n── Strongly positive F−M (F>>M) ──")
print(gs[gs["mean_effect"] > 0.5].to_string())
print("\n── Wide CI, near-zero mean ──")
print(gs[(gs["mean_effect"].abs()<0.3) & (gs["max_ci_width"]>0.5)]
      .sort_values("max_ci_width",ascending=False).to_string())
print("\n── CI crosses zero in ≥8 regions ──")
print(gs[gs["n_crosses"]>=8].sort_values("mean_effect").to_string())
print("\n── Highest bimodality ──")
print(gs.sort_values("max_bimod",ascending=False).head(20).to_string())
