"""Build an interactive HTML viewer for OCR baseline results."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


FIELDS = ("seller", "address", "timestamp", "total_cost")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an OCR comparison HTML report.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("reports/metrics/ocr_baseline/development/results_detailed.csv"),
    )
    parser.add_argument(
        "--images-dir", type=Path, default=Path("data/raw/mc_ocr2021/train_images")
    )
    parser.add_argument("--output", type=Path, help="Default: report.html beside results")
    return parser.parse_args()


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def esc(value: object) -> str:
    text = "" if value is None else str(value)
    return html.escape(text if text.lower() != "nan" else "")


def build_card(row: dict[str, str], image_src: str, index: int) -> str:
    field_rows = []
    has_error = False
    for field in FIELDS:
        if field in ("seller", "address"):
            score = float(row.get(f"{field}_similarity") or 0)
            matched = score >= 0.8
            metric = f"{score * 100:.1f}% giống"
        else:
            matched = as_bool(row.get(f"{field}_match", ""))
            metric = "Khớp" if matched else "Không khớp"
        has_error |= not matched
        state = "ok" if matched else "bad"
        field_rows.append(
            f'''<section class="field {state}" data-field="{field}" data-ok="{str(matched).lower()}">
              <div class="field-title"><strong>{field}</strong><span>{metric}</span></div>
              <div class="pair"><label>GT</label><div>{esc(row.get(f"{field}_gt")) or '<em>Trống</em>'}</div></div>
              <div class="pair"><label>Pred</label><div>{esc(row.get(f"{field}_pred")) or '<em>Trống</em>'}</div></div>
            </section>'''
        )
    return f'''<article class="card" data-error="{str(has_error).lower()}" data-search="{esc(' '.join(row.values())).lower()}">
      <div class="image-wrap">
        <div class="number">#{index}</div>
        <img src="{esc(image_src)}" loading="lazy" alt="{esc(row['img_id'])}">
        <div class="img-id">{esc(row['img_id'])}</div>
      </div>
      <div class="fields">{''.join(field_rows)}</div>
    </article>'''


def main() -> None:
    args = parse_args()
    results = args.results.resolve()
    output = (args.output or args.results.with_name("report.html")).resolve()
    images_dir = args.images_dir.resolve()
    if not results.is_file():
        raise FileNotFoundError(f"Results not found: {results}")

    with results.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    cards = []
    missing = 0
    for index, row in enumerate(rows, 1):
        image = images_dir / row["img_id"]
        if not image.is_file():
            missing += 1
        image_src = image.as_uri()
        cards.append(build_card(row, image_src, index))

    metrics_path = results.with_name("metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    macro = metrics.get("macro", {})
    summary = (
        f"{len(rows)} ảnh · Coverage {macro.get('prediction_coverage', 0) * 100:.1f}% · "
        f"Exact match {macro.get('normalized_exact_match', 0) * 100:.1f}%"
    )
    document = f'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR Baseline Report</title>
<style>
:root{{--bg:#f4f5f7;--panel:#fff;--text:#18202a;--muted:#68717d;--ok:#137333;--okbg:#e6f4ea;--bad:#b3261e;--badbg:#fce8e6;--line:#dde1e6}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:17px/1.55 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:5;background:#17212b;color:white;padding:16px 24px;box-shadow:0 2px 8px #0003}}
h1{{font-size:26px;margin:0 0 4px}} .summary{{color:#cbd5df}} .controls{{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}}
input,select,button{{font:inherit;border:1px solid #73808d;border-radius:7px;padding:9px 12px;background:white;min-width:190px}}
#count{{align-self:center;color:#cbd5df}} main{{max-width:1440px;margin:auto;padding:20px}}
.navigation{{display:flex;justify-content:center;align-items:center;gap:16px;margin:0 0 16px}} .navigation button{{min-width:150px;cursor:pointer;font-weight:700}} .navigation button:disabled{{opacity:.4;cursor:not-allowed}} #position{{min-width:130px;text-align:center;font-weight:700}}
.card{{display:none;grid-template-columns:minmax(300px,42%) 1fr;gap:22px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:18px;box-shadow:0 2px 7px #0000000d}}
.card.active{{display:grid}}
.image-wrap{{position:relative;text-align:center}} img{{max-width:100%;max-height:580px;object-fit:contain;background:#eee;border-radius:6px}}
.number{{position:absolute;top:8px;left:8px;background:#17212bdd;color:white;padding:4px 8px;border-radius:5px}} .img-id{{font:700 17px/1.4 monospace;margin-top:9px;word-break:break-all}}
.fields{{display:grid;grid-template-columns:1fr 1fr;gap:10px}} .field{{border:1px solid var(--line);border-left:5px solid;border-radius:7px;padding:10px;min-width:0}}
.field.ok{{border-left-color:var(--ok);background:var(--okbg)}} .field.bad{{border-left-color:var(--bad);background:var(--badbg)}}
.field-title{{display:flex;justify-content:space-between;text-transform:uppercase;margin-bottom:8px;font-size:19px}} .field-title span{{font-size:15px;color:var(--muted)}}
.pair{{display:grid;grid-template-columns:42px 1fr;margin-top:6px}} .pair label{{font-weight:700;color:var(--muted)}} .pair div{{white-space:pre-wrap;word-break:break-word}} em{{color:var(--muted)}}
.warning{{background:#fff3cd;color:#664d03;padding:8px 24px}}
@media(max-width:850px){{.card.active{{grid-template-columns:1fr}}.fields{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>OCR Baseline – Development</h1><div class="summary">{esc(summary)}</div>
<div class="controls"><input id="search" placeholder="Tìm ID hoặc nội dung…"><select id="filter"><option value="all">Tất cả kết quả</option><option value="error">Có ít nhất một lỗi</option><option value="seller">Seller sai</option><option value="address">Address sai</option><option value="timestamp">Timestamp sai</option><option value="total_cost">Total cost sai</option></select><span id="count"></span></div></header>
{f'<div class="warning">Không tìm thấy {missing} ảnh. Kiểm tra lại --images-dir.</div>' if missing else ''}
<main><nav class="navigation"><button id="previous" type="button">← Hóa đơn trước</button><span id="position"></span><button id="next" type="button">Hóa đơn sau →</button></nav>{''.join(cards)}</main>
<script>
const cards=[...document.querySelectorAll('.card')], search=document.querySelector('#search'), filter=document.querySelector('#filter'), count=document.querySelector('#count'), position=document.querySelector('#position'), previous=document.querySelector('#previous'), next=document.querySelector('#next');
let visible=[], current=0;
function show(){{cards.forEach(c=>c.classList.remove('active'));if(visible.length){{current=Math.max(0,Math.min(current,visible.length-1));visible[current].classList.add('active')}}position.textContent=visible.length?`${{current+1}} / ${{visible.length}}`:'Không có kết quả';count.textContent=`${{visible.length}}/${{cards.length}} hóa đơn`;previous.disabled=!visible.length||current===0;next.disabled=!visible.length||current===visible.length-1}}
function update(){{const q=search.value.trim().toLowerCase(), f=filter.value;visible=cards.filter(c=>{{let ok=!q||c.dataset.search.includes(q);if(f==='error')ok=ok&&c.dataset.error==='true';else if(f!=='all'){{const x=c.querySelector(`[data-field="${{f}}"]`);ok=ok&&x&&x.dataset.ok==='false'}}return ok}});current=0;show()}}
previous.addEventListener('click',()=>{{if(current>0){{current--;show();scrollTo(0,0)}}}});next.addEventListener('click',()=>{{if(current<visible.length-1){{current++;show();scrollTo(0,0)}}}});
document.addEventListener('keydown',e=>{{if(e.target.matches('input,select'))return;if(e.key==='ArrowLeft')previous.click();if(e.key==='ArrowRight')next.click()}});
search.addEventListener('input',update);filter.addEventListener('change',update);update();
</script></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"Report created: {output}")


if __name__ == "__main__":
    main()
