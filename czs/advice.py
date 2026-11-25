"""Rule-based, advisory recommendations from zone statistics (not a substitute for scouting)."""
from .config import ALERT_DISEASED_PCT

DISCLAIMER = ("Advisory only: derived from spectral indices. Confirm by field scouting or with your "
              "agronomist before applying any treatment.")


def advise(stats, area_ha=None):
    h, s, d = stats["zone_pct"]
    m = stats.get("zone_means", {})
    out = []

    def ha(p):
        return f" (~{area_ha * p / 100:.1f} ha)" if area_ha else ""

    if d >= ALERT_DISEASED_PCT:
        out.append(("High", "Urgent scouting of diseased zones",
                    f"{d:.0f}% of the field{ha(d)} is classed Diseased. Walk the red zones within 48 hours, "
                    "sample leaves, and consult your agronomist. Clean equipment when moving from red to green areas."))
    elif d >= 5:
        out.append(("Medium", "Scout emerging problem patches",
                    f"{d:.0f}% of the field{ha(d)} is Diseased. Inspect these patches this week before they spread."))
    if s >= 25:
        out.append(("Medium", "Investigate stressed zones",
                    f"{s:.0f}% of the field{ha(s)} shows stress. Check irrigation uniformity, nutrient status and pests."))
    try:
        ndre, ndvi = m["NDRE"], m["NDVI"]
        if None not in (ndre[0], ndre[1], ndvi[0], ndvi[1]) and (ndre[0] - ndre[1]) > 0.08 \
                and (ndvi[0] - ndvi[1]) < 0.15:
            out.append(("Medium", "Early chlorophyll / nitrogen signal",
                        "Red-edge (NDRE) is falling faster than NDVI in stressed zones, an early sign of "
                        "nutrient or chlorophyll decline. Consider tissue or soil sampling."))
        ndwi = m["NDWI"]
        if None not in (ndwi[0], ndwi[1]) and (ndwi[1] - ndwi[0]) > 0.08:
            out.append(("Medium", "Possible water deficit",
                        "Canopy water index (NDWI) is lower in stressed zones. Check soil moisture and "
                        "irrigation coverage."))
    except (KeyError, TypeError):
        pass
    if h >= 80 and not out:
        out.append(("Low", "Field in good condition", "Most of the field is healthy. Continue routine monitoring."))
    if not out:
        out.append(("Low", "Routine monitoring", "No major issues detected. Re-scan in 7–10 days."))
    return out
