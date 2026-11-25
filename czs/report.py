"""Self-contained HTML report (print to PDF from the browser)."""
import base64
import html
import io

import numpy as np
from PIL import Image

from . import CLASS_COLORS, CLASS_NAMES


def _png_b64(cls, scale=3):
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in CLASS_COLORS], np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb[cls]).resize((cls.shape[1] * scale, cls.shape[0] * scale), Image.NEAREST).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_report(scan, official_cls, ai_cls, review, advice_items, area_ha, disclaimer):
    e = html.escape
    pct = [float((official_cls == i).mean() * 100) for i in range(3)]
    rows = "".join(f"<tr><td style='color:{CLASS_COLORS[i]};font-weight:bold'>{CLASS_NAMES[i]}</td>"
                   f"<td>{pct[i]:.1f}%</td><td>{(area_ha * pct[i] / 100):.2f} ha</td></tr>" if area_ha else
                   f"<tr><td>{CLASS_NAMES[i]}</td><td>{pct[i]:.1f}%</td><td>–</td></tr>" for i in range(3))
    adv = "".join(f"<li><b>[{e(p)}] {e(t)}</b> – {e(x)}</li>" for p, t, x in advice_items)
    rev = "<p><i>Not yet validated by an agronomist.</i></p>"
    if review:
        m = review.get("metrics", {})
        rev = (f"<p><b>Validated by:</b> {e(review['agronomist_name'])} on {e(review['created'][:10])}<br>"
               f"<b>Status:</b> {e(review['status'])}<br><b>Agreement with K-Means:</b> accuracy "
               f"{m.get('accuracy', 0) * 100:.1f}%, κ {m.get('kappa', 0):.2f}, boundary F1 {m.get('bf1', 0):.2f}</p>"
               f"<p><b>Notes:</b> {e(review.get('notes') or '–')}<br><b>Recommendation:</b> "
               f"{e(review.get('recommendation') or '–')}</p>")
    imgs = f"<figure><img src='data:image/png;base64,{_png_b64(official_cls)}'><figcaption>" \
           f"{'Agronomist-validated zones' if review else 'K-Means zones'}</figcaption></figure>"
    if review:
        imgs += f"<figure><img src='data:image/png;base64,{_png_b64(ai_cls)}'><figcaption>K-Means zones</figcaption></figure>"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Crop stress report</title>
<style>body{{font-family:Arial,sans-serif;max-width:900px;margin:30px auto;color:#222}}
h1{{color:#2e7d32}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 14px}}
figure{{display:inline-block;margin:8px;text-align:center}}img{{image-rendering:pixelated;width:300px}}
.note{{color:#777;font-size:12px}}</style></head><body>
<h1>Crop Stress Zoning Report</h1>
<p><b>Field:</b> {e(scan['field_name'])} ({e(scan.get('crop') or '–')}) · <b>Farmer:</b> {e(scan['farmer_name'])}<br>
<b>Scan #{scan['id']}</b> · {e(scan['created'][:16].replace('T', ' '))} UTC</p>
{imgs}
<h2>Zone summary</h2><table><tr><th>Zone</th><th>Share</th><th>Area</th></tr>{rows}</table>
<h2>Agronomist review</h2>{rev}
<h2>Recommendations</h2><ul>{adv}</ul>
<p class='note'>{e(disclaimer)}</p></body></html>"""


def build_photo_report(details, rgb, overlay_img, advice_items, disclaimer, title="Crop photo analysis", meta="", diagnosis=None):
    e = html.escape
    from PIL import Image as _I

    def b64(a):
        buf = io.BytesIO(); _I.fromarray(a).save(buf, "JPEG", quality=88); return base64.b64encode(buf.getvalue()).decode()
    z = details["zone_pct"]
    adv = "".join(f"<li><b>[{e(p)}] {e(t)}</b> – {e(x)}</li>" for p, t, x in advice_items)
    diag = f"<h2>Agronomist diagnosis</h2><p>{e(diagnosis)}</p>" if diagnosis else ""
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{e(title)}</title>
<style>body{{font-family:Arial,sans-serif;max-width:900px;margin:30px auto;color:#222}}h1{{color:#2e7d32}}
td,th{{border:1px solid #ccc;padding:6px 14px}}table{{border-collapse:collapse}}img{{width:420px;margin:4px}}
.big{{font-size:42px;font-weight:bold;color:#2e7d32}}.note{{color:#777;font-size:12px}}</style></head><body>
<h1>{e(title)}</h1><p>{e(meta)}</p>
<p><span class='big'>{details['health_score']:.0f}/100</span> &nbsp; {e(details['grade'])}</p>
<img src='data:image/jpeg;base64,{b64(rgb)}'><img src='data:image/jpeg;base64,{b64(overlay_img)}'>
<h2>Details</h2><table><tr><th>Healthy</th><th>Yellowing</th><th>Browning / diseased</th><th>Canopy cover</th><th>Lesions</th></tr>
<tr><td>{z[0]:.1f}%</td><td>{z[1]:.1f}%</td><td>{z[2]:.1f}%</td><td>{details['canopy_pct']:.1f}%</td><td>{details['lesion_count']}</td></tr></table>
{diag}<h2>Recommendations</h2><ul>{adv}</ul><p class='note'>{e(disclaimer)}</p></body></html>"""
