"""
Danki Impact Scoring
=========================================================
Synthetic investment dataset generator.

Produces 2,000 realistic investment project records across:
  - 6 asset classes
  - 50+ countries (EU + global)
  - €1,500 → €50,000,000 ticket sizes
  - 5 regulatory frameworks (EU Taxonomy, SFDR, CSRD, TCFD, PRIIPs/MiFID II)
  - 8 impact dimensions (Climate, Water, Gender, Social Mobility,
    Territory, Governance/Corruption, Pollution, Innovation)
  - 18 SFDR mandatory PAI indicators
  - Composite danki score (0–100) with ML calibration layer

Ground-truth scoring logic:
  - Weighted composite index (regulator-friendly, fully explainable)
  - Country/sector risk multipliers calibrated to public indices
    (Transparency International CPI, ND-GAIN Climate Vulnerability,
     UNDP HDI, World Bank Doing Business successor)
  - Investment size non-linearity (log-scale normalisation)

Author : Nambona YANGUERE / Danki Studio
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────
# 1. REFERENCE TABLES
# ─────────────────────────────────────────────────────────────

COUNTRIES = {
    # country: (region, cpi_score 0-100, climate_vuln 0-1, hdi 0-1, eu_member)
    "France":         ("Western Europe",  71, 0.28, 0.903, True),
    "Germany":        ("Western Europe",  78, 0.25, 0.942, True),
    "Netherlands":    ("Western Europe",  79, 0.30, 0.941, True),
    "Spain":          ("Southern Europe", 60, 0.38, 0.905, True),
    "Italy":          ("Southern Europe", 56, 0.35, 0.895, True),
    "Poland":         ("Eastern Europe",  54, 0.32, 0.880, True),
    "Romania":        ("Eastern Europe",  46, 0.40, 0.821, True),
    "Portugal":       ("Southern Europe", 62, 0.36, 0.866, True),
    "Sweden":         ("Northern Europe", 85, 0.20, 0.952, True),
    "Denmark":        ("Northern Europe", 90, 0.18, 0.948, True),
    "Belgium":        ("Western Europe",  73, 0.27, 0.937, True),
    "Austria":        ("Western Europe",  74, 0.22, 0.916, True),
    "Czech Republic": ("Eastern Europe",  57, 0.31, 0.900, True),
    "Hungary":        ("Eastern Europe",  42, 0.42, 0.854, True),
    "Greece":         ("Southern Europe", 49, 0.45, 0.887, True),
    "United Kingdom": ("Western Europe",  71, 0.26, 0.929, False),
    "Switzerland":    ("Western Europe",  82, 0.19, 0.962, False),
    "Norway":         ("Northern Europe", 84, 0.17, 0.966, False),
    "United States":  ("North America",   69, 0.33, 0.926, False),
    "Canada":         ("North America",   76, 0.28, 0.936, False),
    "Brazil":         ("Latin America",   36, 0.55, 0.760, False),
    "Mexico":         ("Latin America",   31, 0.60, 0.758, False),
    "Colombia":       ("Latin America",   39, 0.58, 0.752, False),
    "Morocco":        ("MENA",            38, 0.62, 0.683, False),
    "Tunisia":        ("MENA",            40, 0.65, 0.731, False),
    "Egypt":          ("MENA",            35, 0.68, 0.728, False),
    "Senegal":        ("Sub-Saharan Africa", 43, 0.72, 0.511, False),
    "Kenya":          ("Sub-Saharan Africa", 36, 0.70, 0.601, False),
    "Nigeria":        ("Sub-Saharan Africa", 25, 0.75, 0.535, False),
    "South Africa":   ("Sub-Saharan Africa", 41, 0.65, 0.713, False),
    "India":          ("South Asia",      39, 0.58, 0.633, False),
    "Bangladesh":     ("South Asia",      24, 0.78, 0.661, False),
    "Vietnam":        ("Southeast Asia",  41, 0.62, 0.703, False),
    "Indonesia":      ("Southeast Asia",  34, 0.66, 0.705, False),
    "Philippines":    ("Southeast Asia",  33, 0.70, 0.699, False),
    "Japan":          ("East Asia",       73, 0.35, 0.920, False),
    "South Korea":    ("East Asia",       63, 0.30, 0.929, False),
    "Australia":      ("Oceania",         75, 0.32, 0.951, False),
    "New Zealand":    ("Oceania",         85, 0.25, 0.937, False),
    "Chile":          ("Latin America",   66, 0.50, 0.860, False),
    "Argentina":      ("Latin America",   37, 0.52, 0.842, False),
    "Turkey":         ("MENA",            34, 0.50, 0.838, False),
    "Ukraine":        ("Eastern Europe",  33, 0.45, 0.773, False),
    "Kazakhstan":     ("Central Asia",    36, 0.48, 0.811, False),
    "Ghana":          ("Sub-Saharan Africa", 43, 0.68, 0.632, False),
    "Ethiopia":       ("Sub-Saharan Africa", 37, 0.80, 0.492, False),
    "Pakistan":       ("South Asia",      29, 0.72, 0.544, False),
    "Sri Lanka":      ("South Asia",      34, 0.60, 0.782, False),
    "Peru":           ("Latin America",   40, 0.55, 0.762, False),
    "Bolivia":        ("Latin America",   31, 0.58, 0.698, False),
}

SECTORS = {
    # sector: (taxonomy_eligible, ghg_intensity_factor, gender_gap_factor, social_factor)
    "Renewable Energy":          (True,  0.05, 0.55, 0.65),
    "Energy Efficiency":         (True,  0.10, 0.52, 0.68),
    "Sustainable Agriculture":   (True,  0.35, 0.60, 0.80),
    "Water & Sanitation":        (True,  0.08, 0.62, 0.85),
    "Clean Transportation":      (True,  0.15, 0.48, 0.70),
    "Green Building / Real Estate": (True, 0.20, 0.58, 0.62),
    "Circular Economy":          (True,  0.18, 0.55, 0.72),
    "Biodiversity / Nature":     (True,  0.05, 0.63, 0.75),
    "Healthcare":                (False, 0.22, 0.72, 0.90),
    "Education & Skills":        (False, 0.12, 0.75, 0.95),
    "Financial Inclusion":       (False, 0.10, 0.65, 0.88),
    "Digital Infrastructure":    (False, 0.25, 0.42, 0.60),
    "Affordable Housing":        (False, 0.28, 0.60, 0.88),
    "Food & Nutrition":          (False, 0.42, 0.62, 0.78),
    "Manufacturing (conventional)": (False, 0.65, 0.50, 0.55),
    "Extractive Industry":       (False, 0.85, 0.38, 0.42),
    "Private Equity (diversified)": (False, 0.40, 0.52, 0.60),
    "SME Finance":               (False, 0.30, 0.60, 0.72),
    "Microfinance":              (False, 0.15, 0.68, 0.88),
    "Social Infrastructure":     (True,  0.10, 0.70, 0.92),
}

ASSET_CLASSES = [
    "Private Equity / Venture",
    "Project Finance",
    "Real Estate",
    "SME Debt",
    "Green Bond",
    "Microfinance / Social Bond",
]

SFDR_ARTICLES = ["Article 6", "Article 8", "Article 9"]


# ─────────────────────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def clamp(x, lo=0.0, hi=1.0):
    return float(np.clip(x, lo, hi))

def noisy(base, sigma=0.08):
    """Add Gaussian noise to a base probability [0-1]."""
    return clamp(base + RNG.normal(0, sigma))

def noisy100(base, sigma=3.0):
    """Add Gaussian noise to a 0-100 score."""
    return clamp(base + RNG.normal(0, sigma), 0.0, 100.0)


def investment_size_factor(amount_eur):
    """
    Log-scale normalisation: maps €1,500 → 0.0, €50,000,000 → 1.0
    Reflects that larger projects have more regulatory scrutiny,
    more data availability, and typically more structured governance.
    """
    lo, hi = np.log(1_500), np.log(50_000_000)
    return clamp((np.log(amount_eur) - lo) / (hi - lo))


# ─────────────────────────────────────────────────────────────
# 3. COMPOSITE SCORING ENGINE
# ─────────────────────────────────────────────────────────────

DIMENSION_WEIGHTS = {
    "gender":     0.20,
    "social":     0.15,
    "governance": 0.15,
    "climate":    0.18,
    "pollution":  0.10,
    "water":      0.08,
    "territory":  0.08,
    "innovation": 0.06,
}

SOCIAL_VETO_THRESHOLD = 30.0

assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1"


def score_climate(ghg_factor, taxonomy_eligible, climate_vuln, size_factor):
    """
    Climate & Environment dimension (0–100).
    Higher score = stronger climate contribution / avoided emissions.
    """
    base = (1 - ghg_factor) * 70          # low GHG intensity → high base
    if taxonomy_eligible:
        base += 20                          # taxonomy alignment bonus
    base -= climate_vuln * 15              # operating in vulnerable region
    base += size_factor * 5               # scale of impact
    return clamp(base / 100) * 100


def score_water(sector_key, ghg_factor, country_vuln):
    """Water & Resource use dimension (0–100)."""
    water_intensive = ghg_factor > 0.4
    base = 70 - (40 if water_intensive else 0) - country_vuln * 20
    return clamp(base / 100) * 100


def score_gender(gender_factor, cpi, hdi):
    """
    Gender & Social equity dimension (0–100).
    Spans full range: worst-case countries/sectors score ~15,
    best-case ~100. Corruption actively penalises gender rights.
    """
    raw = gender_factor * 55 + (cpi / 100) * 25 + hdi * 15 - (1 - cpi / 100) * 20
    return clamp(raw * (100 / 75), 0.0, 100.0)


def score_social(social_factor, hdi, size_factor):
    """
    Social mobility dimension (0–100).
    Structural deprivation penalty compounds on low-HDI + low-social-factor
    contexts — extractive projects in deprived regions score very low.
    """
    raw = social_factor * 50 + hdi * 25 + size_factor * 15 - (1 - hdi) * (1 - social_factor) * 25
    return clamp(raw, 0.0, 100.0)


def score_territory(hdi, region, size_factor, eu_member):
    """Territory & Local wealth dimension (0–100)."""
    # Projects in lower HDI regions can have stronger territory impact
    territory_leverage = (1 - hdi) * 40    # higher in less developed areas
    governance_bonus = 20 if eu_member else 10
    base = 30 + territory_leverage + governance_bonus + size_factor * 10
    return clamp(base / 100) * 100


def score_governance(cpi, eu_member, asset_class):
    """Governance & Corruption dimension (0–100)."""
    base = (cpi / 100) * 65
    if eu_member:
        base += 20
    if asset_class in ("Green Bond", "Project Finance"):
        base += 10     # more structured governance required
    return clamp(base / 100) * 100


def score_pollution(ghg_factor, taxonomy_eligible):
    """Pollution & Health dimension (0–100)."""
    base = (1 - ghg_factor) * 75
    if taxonomy_eligible:
        base += 20
    return clamp(base / 100) * 100


def score_innovation(sector_key, size_factor, hdi):
    """Innovation & Resilience dimension (0–100)."""
    innovation_sectors = {
        "Renewable Energy", "Digital Infrastructure", "Clean Transportation",
        "Energy Efficiency", "Circular Economy", "Education & Skills",
        "Financial Inclusion", "Water & Sanitation",
    }
    base = 40
    if sector_key in innovation_sectors:
        base += 30
    base += size_factor * 20 + hdi * 10
    return clamp(base / 100) * 100


def compute_danki_score(dims: dict) -> float:
    """Weighted composite danki score (0–100)."""
    return sum(DIMENSION_WEIGHTS[k] * dims[k] for k in DIMENSION_WEIGHTS)


# ─────────────────────────────────────────────────────────────
# 4. REGULATORY FRAMEWORK LOGIC
# ─────────────────────────────────────────────────────────────

def taxonomy_alignment(sector_key, ghg_factor, climate_vuln, danki_score):
    """
    Returns (eligible, aligned, dnsh_pass, substantial_contribution_score).
    EU Taxonomy: Substantial Contribution to ≥1 of 6 env. objectives + DNSH.
    """
    eligible = SECTORS[sector_key][0]
    if not eligible:
        return False, False, False, 0.0

    # Substantial contribution proxy
    sc_score = clamp((1 - ghg_factor) * 0.6 + (danki_score / 100) * 0.4)

    # DNSH: Do No Significant Harm — fails if GHG too high OR climate too vulnerable
    dnsh_pass = (ghg_factor < 0.50) and (climate_vuln < 0.70)

    aligned = dnsh_pass and (sc_score > 0.40)
    return eligible, aligned, dnsh_pass, round(sc_score * 100, 1)


def sfdr_classification(danki_score, taxonomy_aligned, pai_score, governance_score):
    """
    Infer SFDR article classification.
    Art. 9: highest bar — sustainable investment objective + Taxonomy alignment
    Art. 8: E/S characteristics promoted + PAI consideration
    Art. 6: no specific sustainability claims
    """
    if taxonomy_aligned and danki_score >= 72 and pai_score >= 70:
        return "Article 9"
    elif danki_score >= 50 and pai_score >= 50:
        return "Article 8"
    else:
        return "Article 6"


def tcfd_risk(climate_vuln, ghg_factor, sector_key):
    """
    TCFD physical + transition risk flags.
    Returns (physical_risk: low/medium/high, transition_risk: low/medium/high).
    """
    physical = "High" if climate_vuln > 0.65 else ("Medium" if climate_vuln > 0.40 else "Low")
    high_transition = {
        "Extractive Industry", "Manufacturing (conventional)",
        "Food & Nutrition", "Private Equity (diversified)"
    }
    low_transition = {
        "Renewable Energy", "Energy Efficiency", "Clean Transportation",
        "Circular Economy", "Biodiversity / Nature"
    }
    if sector_key in high_transition:
        transition = "High"
    elif sector_key in low_transition:
        transition = "Low"
    else:
        transition = "Medium"
    return physical, transition


def pai_score(ghg_factor, gender_factor, cpi, climate_vuln, governance_score):
    """
    Aggregate PAI score (0–100) covering the 18 mandatory SFDR indicators.
    Higher = fewer adverse impacts.
    """
    # Key PAI sub-scores (simplified mapping)
    ghg_pai       = (1 - ghg_factor) * 100          # PAI 1: GHG emissions
    energy_pai    = (1 - ghg_factor * 0.8) * 100    # PAI 2: carbon footprint
    biodiv_pai    = (1 - climate_vuln * 0.5) * 100  # PAI 7: biodiversity
    water_pai     = (1 - ghg_factor * 0.4) * 100    # PAI 8: water
    waste_pai     = (1 - ghg_factor * 0.6) * 100    # PAI 9: hazardous waste
    gender_pai    = gender_factor * 100              # PAI 12: gender pay gap
    board_div_pai = gender_factor * 90               # PAI 13: board diversity
    corrupt_pai   = (cpi / 100) * 100               # PAI 16: corruption
    tax_pai       = (cpi / 100) * 85                # PAI 17: tax haven

    scores = [ghg_pai, energy_pai, biodiv_pai, water_pai, waste_pai,
              gender_pai, board_div_pai, corrupt_pai, tax_pai, governance_score]
    return round(np.mean(scores), 1)


def mifid_suitability(danki_score, sfdr_art, tcfd_physical, tcfd_transition):
    """
    PRIIPs / MiFID II sustainability suitability flag.
    Returns suitability_score (0–10) and investor_profile.
    """
    base = danki_score / 10
    if sfdr_art == "Article 9":
        base = min(base + 1.5, 10)
    elif sfdr_art == "Article 8":
        base = min(base + 0.5, 10)
    if tcfd_physical == "High" or tcfd_transition == "High":
        base = max(base - 1.0, 0)

    if base >= 7.5:
        profile = "Sustainability-focused (MiFID Art. 9 preference)"
    elif base >= 5.0:
        profile = "ESG-integrated (MiFID Art. 8 preference)"
    else:
        profile = "Conventional (Article 6 compatible)"
    return round(base, 1), profile


def csrd_materiality(size_factor, sector_key, eu_member, investment_eur):
    """
    CSRD / ESRS double materiality flag.
    Post-Omnibus 2025: applies to companies >1,000 employees and >€450M turnover.
    We proxy by investment size + sector + EU membership.
    """
    # Large investments in EU in regulated sectors likely touch CSRD-scope companies
    likely_in_scope = (
        investment_eur > 5_000_000 and
        eu_member and
        sector_key not in ("Microfinance", "SME Finance", "Financial Inclusion")
    )
    impact_material  = sector_key in {
        "Renewable Energy", "Extractive Industry", "Manufacturing (conventional)",
        "Food & Nutrition", "Clean Transportation", "Green Building / Real Estate"
    }
    financial_material = eu_member and size_factor > 0.6
    return likely_in_scope, impact_material, financial_material


# ─────────────────────────────────────────────────────────────
# 5. DATASET GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_danki_data(n: int = 2000) -> pd.DataFrame:
    country_list = list(COUNTRIES.keys())
    sector_list  = list(SECTORS.keys())

    rows = []
    for _ in range(n):
        # ── Sample metadata ──────────────────────────────────
        country   = RNG.choice(country_list)
        sector    = RNG.choice(sector_list)
        asset_cls = RNG.choice(ASSET_CLASSES)

        region, cpi, climate_vuln, hdi, eu_member = COUNTRIES[country]
        tax_elig, ghg_base, gender_base, social_base = SECTORS[sector]

        # Investment size: log-uniform between €1,500 and €50M
        investment_eur = float(np.exp(
            RNG.uniform(np.log(1_500), np.log(50_000_000))
        ))
        size_factor = investment_size_factor(investment_eur)

        # Add sector/country noise to base rates
        ghg_factor    = clamp(noisy(ghg_base,    sigma=0.08))
        gender_factor = clamp(noisy(gender_base, sigma=0.07))
        social_factor = clamp(noisy(social_base, sigma=0.07))
        vuln          = clamp(noisy(climate_vuln, sigma=0.05))
        cpi_n         = cpi / 100  # normalised 0–1

        # ── Impact dimension scores ───────────────────────────
        dims = {
            "climate":    noisy100(score_climate(ghg_factor, tax_elig, vuln, size_factor)),
            "water":      noisy100(score_water(sector, ghg_factor, vuln)),
            "gender":     noisy100(score_gender(gender_factor, cpi, hdi)),
            "social":     noisy100(score_social(social_factor, hdi, size_factor)),
            "territory":  noisy100(score_territory(hdi, region, size_factor, eu_member)),
            "governance": noisy100(score_governance(cpi, eu_member, asset_cls)),
            "pollution":  noisy100(score_pollution(ghg_factor, tax_elig)),
            "innovation": noisy100(score_innovation(sector, size_factor, hdi)),
        }
        # Clamp all dimensions to [0, 100]
        dims = {k: clamp(v, 0, 100) for k, v in dims.items()}

        danki = round(compute_danki_score(dims), 2)

        # ── Regulatory framework outputs ──────────────────────
        tax_elig_f, tax_aligned, dnsh_pass, sc_score = taxonomy_alignment(
            sector, ghg_factor, vuln, danki
        )
        p_score    = pai_score(ghg_factor, gender_factor, cpi, vuln, dims["governance"])
        sfdr_art   = sfdr_classification(danki, tax_aligned, p_score, dims["governance"])
        phys_risk, trans_risk = tcfd_risk(vuln, ghg_factor, sector)
        mifid_s, mifid_prof  = mifid_suitability(danki, sfdr_art, phys_risk, trans_risk)
        csrd_scope, imp_mat, fin_mat = csrd_materiality(
            size_factor, sector, eu_member, investment_eur
        )

        # ── danki band ────────────────────────────────────────
        if danki >= 75:
            band = "Dark Green"
        elif danki >= 58:
            band = "Green"
        elif danki >= 40:
            band = "Amber"
        else:
            band = "Red"

        # Social veto: cap at Amber if gender or social below threshold
        social_veto = (
            dims["gender"] < SOCIAL_VETO_THRESHOLD or
            dims["social"] < SOCIAL_VETO_THRESHOLD
        )
        if social_veto and band in ("Dark Green", "Green"):
            band = "Amber"

        rows.append({
            # Metadata
            "country":           country,
            "region":            region,
            "eu_member":         eu_member,
            "sector":            sector,
            "asset_class":       asset_cls,
            "investment_eur":    round(investment_eur, 2),
            "size_factor":       round(size_factor, 4),

            # Country context
            "cpi_score":         cpi,
            "climate_vuln":      round(vuln, 3),
            "hdi":               round(hdi, 3),

            # Sector context
            "ghg_intensity":     round(ghg_factor, 3),
            "gender_factor":     round(gender_factor, 3),
            "social_factor":     round(social_factor, 3),

            # Impact dimensions (0–100)
            "dim_climate":       round(dims["climate"],    1),
            "dim_water":         round(dims["water"],      1),
            "dim_gender":        round(dims["gender"],     1),
            "dim_social":        round(dims["social"],     1),
            "dim_territory":     round(dims["territory"],  1),
            "dim_governance":    round(dims["governance"], 1),
            "dim_pollution":     round(dims["pollution"],  1),
            "dim_innovation":    round(dims["innovation"], 1),

            # Composite score
            "danki_score":       danki,
            "danki_band":        band,

            # EU Taxonomy
            "taxonomy_eligible": tax_elig_f,
            "taxonomy_aligned":  tax_aligned,
            "dnsh_pass":         dnsh_pass,
            "sc_score":          sc_score,

            # SFDR
            "sfdr_article":      sfdr_art,
            "pai_score":         p_score,

            # TCFD
            "tcfd_physical":     phys_risk,
            "tcfd_transition":   trans_risk,

            # MiFID II / PRIIPs
            "mifid_suitability": mifid_s,
            "mifid_profile":     mifid_prof,

            # CSRD / ESRS
            "csrd_in_scope":     csrd_scope,
            "impact_material":   imp_mat,
            "financial_material": fin_mat,
        })

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating danki synthetic dataset (n=2,000)...")
    df = generate_danki_data(2000)

    out = Path(__file__).parent.parent / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    # parquet skipped - no pyarrow in this env
    df.to_csv(out / "investment_impacts.csv", index=False)

    print(f"Saved to {out}/investment_impacts.parquet")
    print(f"\nDataset shape : {df.shape}")
    print(f"danki range   : {df.danki_score.min():.1f} – {df.danki_score.max():.1f}")
    print(f"Mean score    : {df.danki_score.mean():.1f}")
    print(f"\nSFDR breakdown:\n{df.sfdr_article.value_counts()}")
    print(f"\ndanki bands:\n{df.danki_band.value_counts()}")
    print(f"\nTaxonomy aligned: {df.taxonomy_aligned.sum()} / {len(df)}")
    print(f"\nCountry sample:\n{df.groupby('country')['danki_score'].mean().sort_values().head(10)}")
