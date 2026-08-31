"""Build a one-image-at-a-time before/after HTML comparison viewer."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare matching images from two folders.")
    parser.add_argument("--before-dir", type=Path, required=True)
    parser.add_argument("--after-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/figures/image_comparison.html"))
    parser.add_argument("--before-label", default="Before")
    parser.add_argument("--after-label", default="After")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before_dir = args.before_dir.resolve()
    after_dir = args.after_dir.resolve()
    if not before_dir.is_dir() or not after_dir.is_dir():
        raise FileNotFoundError("Both --before-dir and --after-dir must exist.")

    before = {
        path.name: path for path in before_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    }
    after = {
        path.name: path for path in after_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    }
    names = sorted(before.keys() & after.keys())
    if not names:
        raise ValueError("No matching image filenames were found in the two folders.")
    records = [
        {"name": name, "before": before[name].as_uri(), "after": after[name].as_uri()}
        for name in names
    ]
    data = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    before_label = html.escape(args.before_label)
    after_label = html.escape(args.after_label)

    document = f'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Before / After Image Comparison</title>
<style>
:root{{--bg:#eef1f4;--panel:#fff;--dark:#17212b;--muted:#65717d}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);font:17px/1.45 system-ui,sans-serif;color:#17212b}}
header{{position:sticky;top:0;z-index:3;background:var(--dark);color:#fff;padding:14px 22px;box-shadow:0 2px 8px #0004}}
.top,.nav{{display:flex;align-items:center;justify-content:center;gap:14px;flex-wrap:wrap}} h1{{font-size:23px;margin:0 auto 0 0}}
input,button{{font:inherit;border:1px solid #7d8995;border-radius:7px;padding:8px 12px}} input{{min-width:270px}}
button{{cursor:pointer;background:#fff;font-weight:700;min-width:145px}} button:disabled{{opacity:.4;cursor:not-allowed}}
.nav{{margin-top:11px}} #position{{min-width:130px;text-align:center}} #filename{{font:700 16px/1.4 monospace;color:#dce5ee;min-width:330px;text-align:center}}
main{{padding:18px;max-width:1800px;margin:auto}} .comparison{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.pane{{background:var(--panel);border-radius:10px;box-shadow:0 2px 8px #0001;padding:12px;min-width:0}}
.pane h2{{font-size:20px;margin:0 0 10px;text-align:center}} .image-box{{height:calc(100vh - 190px);min-height:500px;overflow:auto;background:#dfe3e7;border-radius:7px;text-align:center}}
img{{max-width:100%;height:auto;display:inline-block;vertical-align:top}} .hint{{text-align:center;color:var(--muted);font-size:14px;margin-top:8px}}
@media(max-width:850px){{.comparison{{grid-template-columns:1fr}}.image-box{{height:auto;min-height:0}}}}
</style></head><body>
<header><div class="top"><h1>So sánh ảnh Before / After</h1><input id="search" placeholder="Tìm tên ảnh…"></div>
<div class="nav"><button id="previous">← Ảnh trước</button><span id="position"></span><span id="filename"></span><button id="next">Ảnh sau →</button></div></header>
<main><div class="comparison">
<section class="pane"><h2>{before_label}</h2><div class="image-box"><img id="before" alt="Before"></div></section>
<section class="pane"><h2>{after_label}</h2><div class="image-box"><img id="after" alt="After"></div></section>
</div><div class="hint">Dùng nút hoặc phím ← → để chuyển ảnh. Cuộn trong từng khung để xem ảnh dài.</div></main>
<script>
const all={data};let items=all,current=0;
const before=document.querySelector('#before'),after=document.querySelector('#after'),filename=document.querySelector('#filename'),position=document.querySelector('#position'),previous=document.querySelector('#previous'),next=document.querySelector('#next'),search=document.querySelector('#search');
function show(){{if(!items.length){{before.removeAttribute('src');after.removeAttribute('src');filename.textContent='Không có kết quả';position.textContent='0 / 0';previous.disabled=next.disabled=true;return}}current=Math.max(0,Math.min(current,items.length-1));const item=items[current];before.src=item.before;after.src=item.after;filename.textContent=item.name;position.textContent=`${{current+1}} / ${{items.length}}`;previous.disabled=current===0;next.disabled=current===items.length-1;document.querySelectorAll('.image-box').forEach(x=>x.scrollTop=0)}}
previous.onclick=()=>{{if(current>0){{current--;show()}}}};next.onclick=()=>{{if(current<items.length-1){{current++;show()}}}};
search.oninput=()=>{{const q=search.value.trim().toLowerCase();items=all.filter(x=>x.name.toLowerCase().includes(q));current=0;show()}};
document.addEventListener('keydown',event=>{{if(event.target===search)return;if(event.key==='ArrowLeft')previous.click();if(event.key==='ArrowRight')next.click()}});show();
</script></body></html>'''
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"Matched images: {len(records)}")
    print(f"Report created: {output}")


if __name__ == "__main__":
    main()
