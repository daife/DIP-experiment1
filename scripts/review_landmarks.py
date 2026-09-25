"""Local browser editor for human correction of a 28-point preannotation run."""

from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "datasets/annotations/review_sets/landmark320/annotations.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets/annotations/corrected/landmark28_review320.jsonl"
SCHEMA = ROOT / "datasets/annotations/landmark28_schema.json"
REVIEW_IMAGE_DIR = ROOT / "datasets/annotations/review_sets/landmark320/images"
RAW_IMAGE_DIR = ROOT / "datasets/raw/anime256"


def resolve_image_path(relative_path: str) -> Path:
    """Resolve a review image while keeping requests inside known image roots."""
    candidate = (ROOT / "datasets" / relative_path).resolve()
    allowed_roots = (REVIEW_IMAGE_DIR.resolve(), RAW_IMAGE_DIR.resolve())
    if not any(candidate.is_relative_to(root) for root in allowed_roots):
        raise ValueError("Image path outside review image roots")
    return candidate

HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>28 点人工复核</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f2eb;color:#242321;font:16px system-ui,"Microsoft YaHei",sans-serif}
header{padding:14px 24px;background:#232b31;color:white;display:flex;justify-content:space-between;align-items:center}
h1{font-size:21px;margin:0}main{display:grid;grid-template-columns:minmax(620px,1fr) 360px;gap:20px;padding:20px;max-width:1280px;margin:auto}
canvas{width:min(100%,800px);height:auto;background:#dedbd4;display:block;margin:auto;cursor:crosshair;touch-action:none;border:1px solid #aaa}
aside{background:white;padding:20px;border:1px solid #ded8ce;align-self:start}button,input,textarea{font:inherit}
button{padding:9px 13px;margin:3px;cursor:pointer;background:#e7ecec;border:1px solid #abb9bb;border-radius:4px}
button.primary{background:#245d6f;color:white;border-color:#245d6f}button:disabled{opacity:.5;cursor:default}
textarea{width:100%;padding:8px;margin:5px 0 12px;border:1px solid #aaa}
textarea{min-height:55px}p{line-height:1.5}small{color:#625f5b}.warn{color:#ad3b2e;font-weight:bold}.ok{color:#24724d}
.row{display:flex;flex-wrap:wrap;gap:4px}.tag{display:inline-block;padding:3px 8px;border-radius:4px;background:#eee}
@media(max-width:1000px){main{grid-template-columns:1fr}aside{max-width:800px;margin:auto;width:100%}}
</style></head><body><header><h1>28 点人工复核</h1><span id="progress"></span></header>
<main><canvas id="board" width="800" height="800"></canvas><aside>
<div class="row"><button id="prev">← 上一张</button><button id="next">下一张 →</button><button id="jump">跳到未完成</button></div>
<p id="identity"></p><p id="selected"></p>
<p>逐点检查：可见点拖到实际位置；遮挡但能依据脸部结构合理推定的点，先拖到推定位置，再按 H。无法可靠推定或出画的点按 U。其余点保存时记为可见。</p>
<div class="row"><button id="visible">V：可见 1</button><button id="hidden">H：遮挡，位置可推定</button><button id="uncertain">U：位置不可靠</button></div>
<p><button id="allHidden">整图不可标（备注原因）</button></p>
<label>备注（可留空；整图不可标时必填）<textarea id="note"></textarea></label>
<p><button class="primary" id="save">检查完后保存并下一张（S）</button></p><p id="message"></p>
<hr><p><small>黄圈＝自动点；绿色＝可见；灰色＝遮挡且位置可推定；红色＝位置不可靠。模型置信度不等于可见性。灰色点可用于后续训练对照，NME 仍只统计可见点。</small></p>
</aside></main><script>
let records=[],index=0,selected=0,image=new Image(),dirty=false,dragging=false,loaded=false;
const q=id=>document.getElementById(id),canvas=q('board'),ctx=canvas.getContext('2d');
const colors={null:'#d89b22','0':'#737b82','1':'#158454'};
const L={left:48,top:48,size:704};
function pointScreen(p){return [L.left+p.x*L.size/records[index].width,L.top+p.y*L.size/records[index].height]}
function draw(){if(!loaded)return;ctx.fillStyle='#e5e0d7';ctx.fillRect(0,0,800,800);ctx.drawImage(image,L.left,L.top,L.size,L.size);
  const pts=records[index].annotations[0].landmarks;
  pts.forEach((p,i)=>{const [x,y]=pointScreen(p);ctx.beginPath();ctx.arc(x,y,i===selected?10:7,0,Math.PI*2);ctx.fillStyle=p.visibility===0&&p.coordinate_quality!=='reliable_estimate'?'#b5443b':colors[String(p.visibility)];ctx.fill();ctx.lineWidth=i===selected?3:1.5;ctx.strokeStyle='white';ctx.stroke();ctx.font='bold 17px system-ui';ctx.fillStyle='white';ctx.strokeStyle='#172025';ctx.lineWidth=3;ctx.strokeText(String(i),x+9,y-8);ctx.fillText(String(i),x+9,y-8)});
  const p=pts[selected];q('selected').textContent=`选中 #${selected} · (${p.x.toFixed(1)}, ${p.y.toFixed(1)}) · ${p.visibility===null?'未单独标记（保存时为 1）':'visibility='+p.visibility} · ${p.visibility===0?(p.coordinate_quality==='reliable_estimate'?'位置可推定':'位置不可靠'):''} · confidence=${p.confidence?.toFixed(3)??'—'}`;
}
function setMessage(s,bad=false){q('message').textContent=s;q('message').className=bad?'warn':'ok'}
function show(n){if(dirty&&!confirm('本图改动尚未保存，确定离开？'))return;index=Math.max(0,Math.min(records.length-1,n));selected=0;dirty=false;loaded=false;
  const r=records[index];q('identity').innerHTML=`<b>${index+1}/${records.length}</b> · ${r.split} · <code>${r.image_id}</code>${r.review?'<br><span class="ok">已保存复核</span>':''}`;
  q('note').value=r.review?.note??'';q('prev').disabled=index===0;q('next').disabled=index===records.length-1;
  image=new Image();image.onload=()=>{loaded=true;draw()};image.onerror=()=>setMessage('图片无法读取',true);image.src=`/image/${index}`;
  q('progress').textContent=`已保存 ${records.filter(x=>x.review).length}/${records.length}`;setMessage('');
}
function mark(v,quality=null){const p=records[index].annotations[0].landmarks[selected];p.visibility=v;if(v===0)p.coordinate_quality=quality;else delete p.coordinate_quality;dirty=true;draw()}
function nearest(e){const b=canvas.getBoundingClientRect(),x=(e.clientX-b.left)*800/b.width,y=(e.clientY-b.top)*800/b.height;
  let best=-1,dist=1e9;records[index].annotations[0].landmarks.forEach((p,i)=>{const [px,py]=pointScreen(p),d=(x-px)**2+(y-py)**2;if(d<dist){dist=d;best=i}});return {best,dist,x,y}}
canvas.onpointerdown=e=>{if(!loaded)return;const n=nearest(e);if(n.dist<45**2){selected=n.best;dragging=true;canvas.setPointerCapture(e.pointerId);draw()}};
canvas.onpointermove=e=>{if(!dragging)return;const n=nearest(e),p=records[index].annotations[0].landmarks[selected];p.x=Math.round((n.x-L.left)*records[index].width/L.size*10)/10;p.y=Math.round((n.y-L.top)*records[index].height/L.size*10)/10;dirty=true;draw()};
canvas.onpointerup=()=>dragging=false;canvas.onpointercancel=()=>dragging=false;
q('prev').onclick=()=>show(index-1);q('next').onclick=()=>show(index+1);q('visible').onclick=()=>mark(1);q('hidden').onclick=()=>mark(0,'reliable_estimate');q('uncertain').onclick=()=>mark(0,'uncertain');
q('jump').onclick=()=>{const n=records.findIndex((r,i)=>i>index&&!r.review);show(n<0?records.findIndex(r=>!r.review):n)};
q('allHidden').onclick=()=>{if(!confirm('仅用于无可靠人脸或整图无法定位的样本。确认把 28 点全部标为 0 且位置不可靠？请在备注中写原因。'))return;records[index].annotations[0].landmarks.forEach(p=>{p.visibility=0;p.coordinate_quality='uncertain'});dirty=true;draw()};
async function save(){const r=records[index];
 if(r.annotations[0].landmarks.every(p=>p.visibility===0)&&!q('note').value.trim())return setMessage('整图不可标时请填写原因。',true);
 const response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index,landmarks:r.annotations[0].landmarks,note:q('note').value.trim()})});
 const body=await response.json();if(!response.ok)return setMessage(body.error??'保存失败',true);
 r.review=body.review;r.annotations[0].landmarks=body.landmarks;dirty=false;
 if(index<records.length-1){show(index+1);setMessage('上一张已保存。')}else{draw();setMessage('本批最后一张已保存。')}
}
q('save').onclick=save;document.onkeydown=e=>{if(['INPUT','TEXTAREA'].includes(document.activeElement.tagName))return;
 if(e.key.toLowerCase()==='v')mark(1);if(e.key.toLowerCase()==='h')mark(0,'reliable_estimate');if(e.key.toLowerCase()==='u')mark(0,'uncertain');if(e.key.toLowerCase()==='s'){e.preventDefault();save()}if(e.key==='ArrowRight')show(index+1);if(e.key==='ArrowLeft')show(index-1)};
fetch('/api/records').then(r=>r.json()).then(data=>{records=data;show(0)}).catch(e=>setMessage(String(e),true));
</script></body></html>"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    source = args.annotations.resolve()
    output = args.output.resolve()
    if source == output or not source.is_file():
        raise ValueError("Input must exist and output must be a separate corrected file")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    original = read_jsonl(source)
    if not original or any(len(r["annotations"][0]["landmarks"]) != 28 for r in original):
        raise ValueError("Input must contain 28-point annotations")
    ids = [r["image_id"] for r in original]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image_id in input")
    saved = {r["image_id"]: r for r in read_jsonl(output)} if output.exists() else {}
    if set(saved) - set(ids):
        raise ValueError("Output contains IDs absent from input")
    records = [copy.deepcopy(saved.get(r["image_id"], r)) for r in original]

    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, mime: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json_response(self, obj: object, status: int = 200) -> None:
            self.send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                return self.send(HTML.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/records":
                return self.json_response(records)
            if path.startswith("/image/"):
                try:
                    i = int(path.removeprefix("/image/"))
                    r = original[i]
                    image_path = resolve_image_path(r["image_path"])
                    return self.send(image_path.read_bytes(), "image/jpeg")
                except (ValueError, IndexError, FileNotFoundError):
                    return self.json_response({"error": "image not found"}, 404)
            self.json_response({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if self.path != "/api/save":
                return self.json_response({"error": "not found"}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 200_000:
                    raise ValueError("Invalid body length")
                data = json.loads(self.rfile.read(length))
                i = data["index"]
                points = data["landmarks"]
                if type(i) is not int or not 0 <= i < len(records):
                    raise ValueError("Invalid index")
                if len(points) != schema["count"]:
                    raise ValueError("Expected 28 points")
                for p in points:
                    if p["visibility"] is not None and (type(p["visibility"]) is not int or p["visibility"] not in (0, 1)):
                        raise ValueError("Invalid visibility")
                    if not all(isinstance(p[k], (int, float)) for k in ("x", "y")):
                        raise ValueError("Invalid landmark")
                    if p.get("coordinate_quality") not in (None, "reliable_estimate", "uncertain"):
                        raise ValueError("Invalid coordinate quality")
                if all(p["visibility"] == 0 for p in points) and not str(data.get("note", "")).strip():
                    raise ValueError("An unlabelable image requires a reason in note")
                record = copy.deepcopy(records[i])
                defaulted = sum(p["visibility"] is None for p in points)
                adjusted = []
                for j, (target, point, source_point) in enumerate(zip(record["annotations"][0]["landmarks"], points, original[i]["annotations"][0]["landmarks"], strict=True)):
                    target["x"], target["y"], target["visibility"] = point["x"], point["y"], 1 if point["visibility"] is None else point["visibility"]
                    if target["visibility"] == 0:
                        target["coordinate_quality"] = point.get("coordinate_quality") or "uncertain"
                    else:
                        target.pop("coordinate_quality", None)
                    if (target["x"], target["y"]) != (source_point["x"], source_point["y"]):
                        adjusted.append(j)
                record["landmark_schema_id"] = schema["schema_id"]
                record["review"] = {"reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "method": "local_ui_explicit_save", "note": str(data.get("note", ""))[:2000],
                                    "status": "human_reviewed", "default_visible_count": defaulted,
                                    "adjusted_point_indices": adjusted,
                                    "occluded_point_indices": [j for j, p in enumerate(record["annotations"][0]["landmarks"]) if p["visibility"] == 0],
                                    "reliable_occluded_point_indices": [j for j, p in enumerate(record["annotations"][0]["landmarks"]) if p["visibility"] == 0 and p["coordinate_quality"] == "reliable_estimate"]}
                updated = records.copy()
                updated[i] = record
                output.parent.mkdir(parents=True, exist_ok=True)
                temp = output.with_suffix(output.suffix + ".tmp")
                with temp.open("w", encoding="utf-8", newline="\n") as handle:
                    for item in updated:
                        if "review" in item:
                            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                os.replace(temp, output)
                records[i] = record
                self.json_response({"review": record["review"], "landmarks": record["annotations"][0]["landmarks"]})
            except (ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
                self.json_response({"error": str(exc)}, 400)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Review {len(records)} images at {url}")
    print(f"Corrections: {output.relative_to(ROOT)}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
