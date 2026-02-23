"""
IDRIS Scoring Engine
====================
Standalone module — accepts a single project dict and returns
the full scoring breakdown. Used by both the app and the report.

All logic is explicit and traceable — no black box.
Every sub-score maps to a published standard or index.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────
# DIMENSION WEIGHTS  (must sum to 1.0)
# ─────────────────────────────────────────────────────────────
# ── IDRIS Impact Philosophy ──────────────────────────────────
# Gender + Social + Governance = 50% of total score.
# Rationale: social determinants are UPSTREAM of all other outcomes.
# A project with strong climate metrics but weak gender equity or
# social mobility is extractive, not transformative. ESG scores it
# green. IDRIS does not.
#
# Social Veto Rule (enforced in score_project()):
#   if dim_gender < 30 OR dim_social < 30 → band capped at Amber
#   regardless of composite score. Certain social failures are
#   disqualifying, not merely penalising.
# ─────────────────────────────────────────────────────────────
WEIGHTS = {
    "gender":     0.20,   # upstream driver of all outcomes (World Bank, UNDP)
    "social":     0.15,   # determines whether value stays local or extracts
    "governance": 0.15,   # multiplier: destroys all other impact if it fails
    "climate":    0.18,   # material but downstream of social foundations
    "pollution":  0.10,   # partially captured by climate dimension
    "water":      0.08,   # remains material, less primary
    "territory":  0.08,   # partially covered by social mobility
    "innovation": 0.06,   # relevant but tertiary
}

# Social veto thresholds
SOCIAL_VETO_THRESHOLD = 30.0   # if gender OR social below this → cap at Amber

# ─────────────────────────────────────────────────────────────
# REFERENCE TABLES (embedded so the module is self-contained)
# ─────────────────────────────────────────────────────────────

# Country: (cpi 0-100, climate_vuln 0-1, hdi 0-1, eu_member)
COUNTRY_DATA: dict[str, tuple] = {
    "France": (71, 0.28, 0.903, True),
    "Germany": (78, 0.25, 0.942, True),
    "Netherlands": (79, 0.30, 0.941, True),
    "Spain": (60, 0.38, 0.905, True),
    "Italy": (56, 0.35, 0.895, True),
    "Poland": (54, 0.32, 0.880, True),
    "Romania": (46, 0.40, 0.821, True),
    "Portugal": (62, 0.36, 0.866, True),
    "Sweden": (85, 0.20, 0.952, True),
    "Denmark": (90, 0.18, 0.948, True),
    "Belgium": (73, 0.27, 0.937, True),
    "Austria": (74, 0.22, 0.916, True),
    "Czech Republic": (57, 0.31, 0.900, True),
    "Hungary": (42, 0.42, 0.854, True),
    "Greece": (49, 0.45, 0.887, True),
    "United Kingdom": (71, 0.26, 0.929, False),
    "Switzerland": (82, 0.19, 0.962, False),
    "Norway": (84, 0.17, 0.966, False),
    "United States": (69, 0.33, 0.926, False),
    "Canada": (76, 0.28, 0.936, False),
    "Brazil": (36, 0.55, 0.760, False),
    "Mexico": (31, 0.60, 0.758, False),
    "Morocco": (38, 0.62, 0.683, False),
    "Senegal": (43, 0.72, 0.511, False),
    "Kenya": (36, 0.70, 0.601, False),
    "Nigeria": (25, 0.75, 0.535, False),
    "South Africa": (41, 0.65, 0.713, False),
    "India": (39, 0.58, 0.633, False),
    "Vietnam": (41, 0.62, 0.703, False),
    "Indonesia": (34, 0.66, 0.705, False),
    "Japan": (73, 0.35, 0.920, False),
    "Australia": (75, 0.32, 0.951, False),
    "Chile": (66, 0.50, 0.860, False),
    "Turkey": (34, 0.50, 0.838, False),
    "Other / Unknown": (45, 0.50, 0.700, False),
}

# Sector: (taxonomy_eligible, ghg_intensity 0-1)
SECTOR_DATA: dict[str, tuple] = {
    "Renewable Energy": (True, 0.05),
    "Energy Efficiency": (True, 0.10),
    "Sustainable Agriculture": (True, 0.35),
    "Water & Sanitation": (True, 0.08),
    "Clean Transportation": (True, 0.15),
    "Green Building / Real Estate": (True, 0.20),
    "Circular Economy": (True, 0.18),
    "Biodiversity / Nature": (True, 0.05),
    "Healthcare": (False, 0.22),
    "Education & Skills": (False, 0.12),
    "Financial Inclusion": (False, 0.10),
    "Digital Infrastructure": (False, 0.25),
    "Affordable Housing": (False, 0.28),
    "Food & Nutrition": (False, 0.42),
    "Manufacturing (conventional)": (False, 0.65),
    "Extractive Industry": (False, 0.85),
    "Private Equity (diversified)": (False, 0.40),
    "SME Finance": (False, 0.30),
    "Microfinance": (False, 0.15),
    "Social Infrastructure": (True, 0.10),
    "Other": (False, 0.40),
}


# ─────────────────────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class IDRISResult:
    # Composite
    idris_score: float
    idris_band:  str

    # Dimensions (0-100 each)
    dimensions: dict[str, float]

    # Regulatory
    taxonomy_eligible:  bool
    taxonomy_aligned:   bool
    dnsh_pass:          bool
    sc_score:           float        # Substantial Contribution %
    sfdr_article:       str
    pai_score:          float
    tcfd_physical:      str
    tcfd_transition:    str
    mifid_suitability:  float
    mifid_profile:      str
    csrd_in_scope:      bool
    impact_material:    bool
    financial_material: bool

    # Warnings
    social_veto: bool = False
    warnings: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# MAIN SCORING FUNCTION
# ─────────────────────────────────────────────────────────────

def score_project(
    country: str,
    sector: str,
    asset_class: str,
    investment_eur: float,
    # Optional overrides (if analyst has real data)
    ghg_override:    float | None = None,
    gender_override: float | None = None,
    governance_override: float | None = None,
) -> IDRISResult:
    """
    Score a single investment project.
    Returns a fully populated IDRISResult.

    Parameters
    ----------
    country          : country name (see COUNTRY_DATA keys)
    sector           : sector name (see SECTOR_DATA keys)
    asset_class      : asset class string
    investment_eur   : investment size in EUR
    ghg_override     : analyst-provided GHG intensity (0-1), overrides sector default
    gender_override  : analyst-provided gender equality score (0-1)
    governance_override: analyst-provided governance score (0-100)
    """

    # ── Resolve reference data ──────────────────────────────
    c_data  = COUNTRY_DATA.get(country, COUNTRY_DATA["Other / Unknown"])
    s_data  = SECTOR_DATA.get(sector,  SECTOR_DATA["Other"])
    cpi, climate_vuln, hdi, eu_member = c_data
    tax_elig, ghg_base = s_data

    ghg       = ghg_override    if ghg_override    is not None else ghg_base
    gender    = gender_override if gender_override is not None else _default_gender(sector)
    size_f    = _size_factor(investment_eur)

    # ── Dimension scores ────────────────────────────────────
    dims = {
        "climate":    _score_climate(ghg, tax_elig, climate_vuln, size_f),
        "water":      _score_water(ghg, climate_vuln),
        "gender":     _score_gender(gender, cpi, hdi),
        "social":     _score_social(sector, hdi, size_f),
        "territory":  _score_territory(hdi, size_f, eu_member),
        "governance": governance_override if governance_override is not None
                      else _score_governance(cpi, eu_member, asset_class),
        "pollution":  _score_pollution(ghg, tax_elig),
        "innovation": _score_innovation(sector, size_f, hdi),
    }

    idris = round(sum(WEIGHTS[k] * dims[k] for k in WEIGHTS), 2)

    # ── Social veto rule ────────────────────────────────────────
    # Gender or social mobility below threshold → cap at Amber.
    # This enforces that no project can be Dark Green or Green
    # if its social foundation is failing, regardless of composite score.
    social_veto_triggered = (
        dims["gender"] < SOCIAL_VETO_THRESHOLD or
        dims["social"] < SOCIAL_VETO_THRESHOLD
    )

    if idris >= 75:   band = "Dark Green"
    elif idris >= 58: band = "Green"
    elif idris >= 40: band = "Amber"
    else:             band = "Red"

    if social_veto_triggered and band in ("Dark Green", "Green"):
        band = "Amber"  # veto applied

    # ── Regulatory ──────────────────────────────────────────
    tax_elig_f, tax_aligned, dnsh, sc = _taxonomy(sector, ghg, climate_vuln, idris)
    p_score  = _pai(ghg, gender, cpi, climate_vuln, dims["governance"])
    sfdr     = _sfdr(idris, tax_aligned, p_score, dims["governance"])
    phys, tr = _tcfd(climate_vuln, ghg, sector)
    ms, mp   = _mifid(idris, sfdr, phys, tr)
    cs, im, fm = _csrd(size_f, sector, eu_member, investment_eur)

    # ── Warnings & strengths ────────────────────────────────
    warnings, strengths = _flags(dims, ghg, climate_vuln, cpi, eu_member,
                                  tax_aligned, sfdr, phys, tr)

    return IDRISResult(
        idris_score=idris, idris_band=band,
        dimensions=dims,
        social_veto=social_veto_triggered,
        taxonomy_eligible=tax_elig_f, taxonomy_aligned=tax_aligned,
        dnsh_pass=dnsh, sc_score=sc,
        sfdr_article=sfdr, pai_score=p_score,
        tcfd_physical=phys, tcfd_transition=tr,
        mifid_suitability=ms, mifid_profile=mp,
        csrd_in_scope=cs, impact_material=im, financial_material=fm,
        warnings=warnings, strengths=strengths,
    )


# ─────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────

def _clamp(x, lo=0.0, hi=100.0):
    return float(np.clip(x, lo, hi))

def _size_factor(eur: float) -> float:
    lo, hi = np.log(1_500), np.log(50_000_000)
    return float(np.clip((np.log(max(eur, 1_500)) - lo) / (hi - lo), 0, 1))

def _default_gender(sector: str) -> float:
    high = {"Healthcare", "Education & Skills", "Social Infrastructure",
            "Financial Inclusion", "Microfinance", "Water & Sanitation"}
    low  = {"Extractive Industry", "Digital Infrastructure",
            "Manufacturing (conventional)", "Clean Transportation"}
    if sector in high:  return 0.72
    if sector in low:   return 0.44
    return 0.58

def _score_climate(ghg, tax_elig, vuln, size_f):
    base = (1 - ghg) * 70 + (20 if tax_elig else 0) - vuln * 15 + size_f * 5
    return _clamp(base)

def _score_water(ghg, vuln):
    base = 70 - (40 if ghg > 0.4 else ghg * 30) - vuln * 20
    return _clamp(base)

def _score_gender(gender, cpi, hdi):
    # Spans full 0-100 range:
    #   base      = gender_factor*55   (core equality signal)
    #   gov_bonus = (cpi/100)*25       (institutional support)
    #   dev_bonus = hdi*15             (development context)
    #   gov_penalty = -(1-cpi/100)*20  (corruption suppresses gender rights)
    # Rescaled: raw max ~75 → *100/75
    raw = gender * 55 + (cpi / 100) * 25 + hdi * 15 - (1 - cpi / 100) * 20
    return _clamp(raw * (100 / 75))

def _score_social(sector, hdi, size_f):
    # Spans full 0-100 range:
    #   base    = social_factor * 50  (local value creation signal)
    #   dev     = hdi * 25            (development context)
    #   scale   = size_f * 15         (scale of economic contribution)
    #   penalty = -(1-hdi)*(1-sf)*25  (structural deprivation amplifier)
    high_social = {"Healthcare", "Education & Skills", "Microfinance",
                   "Financial Inclusion", "Social Infrastructure",
                   "Affordable Housing", "Water & Sanitation"}
    low_social  = {"Extractive Industry", "Manufacturing (conventional)",
                   "Private Equity (diversified)"}
    sf = 0.82 if sector in high_social else (0.38 if sector in low_social else 0.60)
    bonus = 15 if sector in high_social else 0
    raw = sf * 50 + hdi * 25 + size_f * 15 + bonus - (1 - hdi) * (1 - sf) * 25
    return _clamp(raw)

def _score_territory(hdi, size_f, eu_member):
    leverage = (1 - hdi) * 40
    gov_bonus = 20 if eu_member else 10
    return _clamp(30 + leverage + gov_bonus + size_f * 10)

def _score_governance(cpi, eu_member, asset_class):
    base = (cpi / 100) * 65 + (20 if eu_member else 0)
    if asset_class in ("Green Bond", "Project Finance"):
        base += 10
    return _clamp(base)

def _score_pollution(ghg, tax_elig):
    return _clamp((1 - ghg) * 75 + (20 if tax_elig else 0))

def _score_innovation(sector, size_f, hdi):
    innov = {"Renewable Energy", "Digital Infrastructure", "Clean Transportation",
             "Energy Efficiency", "Circular Economy", "Education & Skills",
             "Financial Inclusion", "Water & Sanitation"}
    return _clamp(40 + (30 if sector in innov else 0) + size_f * 20 + hdi * 10)

def _taxonomy(sector, ghg, vuln, idris):
    elig = SECTOR_DATA.get(sector, SECTOR_DATA["Other"])[0]
    if not elig:
        return False, False, False, 0.0
    sc   = float(np.clip((1 - ghg) * 0.6 + (idris / 100) * 0.4, 0, 1))
    dnsh = (ghg < 0.50) and (vuln < 0.70)
    aligned = dnsh and (sc > 0.40)
    return True, aligned, dnsh, round(sc * 100, 1)

def _pai(ghg, gender, cpi, vuln, gov):
    scores = [
        (1 - ghg) * 100,
        (1 - ghg * 0.8) * 100,
        (1 - vuln * 0.5) * 100,
        (1 - ghg * 0.4) * 100,
        (1 - ghg * 0.6) * 100,
        gender * 100,
        gender * 90,
        (cpi / 100) * 100,
        (cpi / 100) * 85,
        gov,
    ]
    return round(float(np.mean(scores)), 1)

def _sfdr(idris, tax_aligned, pai, gov):
    if tax_aligned and idris >= 72 and pai >= 70:
        return "Article 9"
    elif idris >= 50 and pai >= 50:
        return "Article 8"
    return "Article 6"

def _tcfd(vuln, ghg, sector):
    phys = "High" if vuln > 0.65 else ("Medium" if vuln > 0.40 else "Low")
    high_t = {"Extractive Industry", "Manufacturing (conventional)",
              "Food & Nutrition", "Private Equity (diversified)"}
    low_t  = {"Renewable Energy", "Energy Efficiency", "Clean Transportation",
              "Circular Economy", "Biodiversity / Nature"}
    trans  = "High" if sector in high_t else ("Low" if sector in low_t else "Medium")
    return phys, trans

def _mifid(idris, sfdr, phys, trans):
    base = idris / 10
    if sfdr == "Article 9":   base = min(base + 1.5, 10)
    elif sfdr == "Article 8": base = min(base + 0.5, 10)
    if phys == "High" or trans == "High": base = max(base - 1.0, 0)
    base = round(base, 1)
    if base >= 7.5:   prof = "Sustainability-focused (MiFID Art. 9 preference)"
    elif base >= 5.0: prof = "ESG-integrated (MiFID Art. 8 preference)"
    else:             prof = "Conventional (Article 6 compatible)"
    return base, prof

def _csrd(size_f, sector, eu_member, eur):
    in_scope = (eur > 5_000_000 and eu_member and
                sector not in {"Microfinance", "SME Finance", "Financial Inclusion"})
    imp_mat  = sector in {"Renewable Energy", "Extractive Industry",
                          "Manufacturing (conventional)", "Food & Nutrition",
                          "Clean Transportation", "Green Building / Real Estate"}
    fin_mat  = eu_member and size_f > 0.6
    return in_scope, imp_mat, fin_mat

def _flags(dims, ghg, vuln, cpi, eu_member, tax_aligned, sfdr, phys, trans):
    warnings, strengths = [], []

    # Social veto warnings — always listed first when triggered
    if dims["gender"] < SOCIAL_VETO_THRESHOLD:
        warnings.append(
            f"SOCIAL VETO TRIGGERED: Gender equity score ({dims['gender']:.0f}/100) "
            f"is below the {SOCIAL_VETO_THRESHOLD:.0f}-point floor. Band capped at Amber. "
            "Fundamental gender equity failure is disqualifying under IDRIS methodology."
        )
    if dims["social"] < SOCIAL_VETO_THRESHOLD:
        warnings.append(
            f"SOCIAL VETO TRIGGERED: Social mobility score ({dims['social']:.0f}/100) "
            f"is below the {SOCIAL_VETO_THRESHOLD:.0f}-point floor. Band capped at Amber. "
            "Insufficient local value creation and employment opportunity is disqualifying."
        )

    if dims["governance"] < 40:
        warnings.append("Governance score below threshold — high corruption risk in country of operation.")
    if dims["climate"] < 45:
        warnings.append("Climate score below sector average — review GHG intensity and taxonomy eligibility.")
    if dims["gender"] < 40:
        warnings.append("Gender equity score is low — PAI 12 (gender pay gap) likely adverse.")
    if dims["pollution"] < 40:
        warnings.append("Pollution score indicates significant environmental harm — DNSH risk.")
    if phys == "High":
        warnings.append("TCFD: High physical climate risk — asset may be stranded under 2°C/1.5°C scenarios.")
    if trans == "High":
        warnings.append("TCFD: High transition risk — sector faces significant regulatory/market disruption.")
    if not eu_member:
        warnings.append("Non-EU jurisdiction — additional due diligence required for SFDR/Taxonomy reporting.")
    if ghg > 0.6:
        warnings.append("GHG intensity is high — likely to fail DNSH climate mitigation criterion.")

    if tax_aligned:
        strengths.append("EU Taxonomy aligned — eligible for Article 9 fund eligibility.")
    if sfdr == "Article 9":
        strengths.append("Qualifies as Article 9 sustainable investment under SFDR.")
    elif sfdr == "Article 8":
        strengths.append("Promotes E/S characteristics — classifiable as Article 8 under SFDR.")
    if dims["climate"] >= 75:
        strengths.append("Strong climate contribution — significant GHG avoidance or clean energy generation.")
    if dims["social"] >= 75:
        strengths.append("High social impact — supports employment, skills and local economic development.")
    if dims["governance"] >= 75:
        strengths.append("Strong governance framework — low corruption exposure, robust transparency.")

    return warnings, strengths
