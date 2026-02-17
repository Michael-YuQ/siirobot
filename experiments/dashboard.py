"""
AT-PC 实验实时监控面板 v2

本地部署，从网盘服务器拉取数据，带 SVG 曲线图。
用法: python -m experiments.dashboard
浏览器打开 http://localhost:8050
"""
import json
import time
import threading
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import requests

BASE_URL = "http://111.170.6.103:10005"
REMOTE_ROOT = "atpc_experiments"
METHODS = ["dr", "paired", "atpc"]
GENERATORS = ["G1", "G2", "G3", "G4"]
SEEDS = [42, 123, 456]
ROUND3_START = "2026-02-11 15:41"

cache = {"last_update": None, "experiments": {}, "error": None, "loading": True}


def api_list(path):
    try:
        r = requests.get(f"{BASE_URL}/list", params={"path": path}, timeout=10)
        if r.status_code == 200:
            d = r.json()
            return d.get("items", []) if isinstance(d, dict) else d
    except:
        pass
    return []


def api_download(path):
    try:
        r = requests.get(f"{BASE_URL}/download", params={"path": path}, timeout=20)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


def fetch_one(exp_id):
    items = api_list(f"{REMOTE_ROOT}/{exp_id}")
    if not items:
        return {"exp_id": exp_id, "status": "no_data"}

    # FINAL 文件
    finals = [f for f in items if "_FINAL_" in f.get("name", "")]
    r3_finals = [f for f in finals if f.get("modified", "") >= ROUND3_START]

    # 最新 stats
    stats_files = []
    for f in items:
        n = f.get("name", "")
        if n.startswith("training_stats_") and n.endswith(".json"):
            try:
                num = int(n.replace("training_stats_", "").replace(".json", ""))
                stats_files.append((num, n, f.get("modified", "")))
            except:
                pass

    result = {
        "exp_id": exp_id,
        "dir_modified": max((f.get("modified", "") for f in items), default=""),
        "is_r3": any(f.get("modified", "") >= ROUND3_START for f in items),
        "is_complete": len(r3_finals) > 0,
    }

    if not stats_files:
        result["status"] = "no_stats"
        return result

    stats_files.sort(reverse=True)
    latest_name = stats_files[0][1]
    result["latest_stats_modified"] = stats_files[0][2]

    data = api_download(f"{REMOTE_ROOT}/{exp_id}/{latest_name}")
    if not data:
        result["status"] = "download_failed"
        return result

    result["total_records"] = len(data)

    # 提取所有有 solver_reward 的记录（包括 DR 的每 10/50 iter 评估）
    sr_records = [d for d in data if "solver_reward" in d]

    # 如果没有 solver_reward，尝试用 v_loss 趋势代替（DR 旧代码）
    if not sr_records:
        # DR 没有 solver_reward，用 step_reward 或 v_loss
        step_records = [d for d in data if d.get("step_reward", 0) != 0]
        if step_records:
            sr_records = step_records
            for r in sr_records:
                r["solver_reward"] = r["step_reward"]

    # 构建趋势数据（用于曲线图）
    trend = []
    if sr_records:
        result["latest_iter"] = sr_records[-1].get("iter", 0)
        result["latest_sr"] = sr_records[-1].get("solver_reward", 0)
        result["latest_source"] = sr_records[-1].get("source", "DR")
        result["latest_ar"] = sr_records[-1].get("accept_rate", 0)
        result["latest_regret"] = sr_records[-1].get("regret", 0)

        early = sr_records[:max(len(sr_records) // 5, 1)]
        late = sr_records[-max(len(sr_records) // 5, 1):]
        result["early_sr"] = sum(d["solver_reward"] for d in early) / len(early)
        result["late_sr"] = sum(d["solver_reward"] for d in late) / len(late)
        result["sr_improvement"] = result["late_sr"] - result["early_sr"]

        for rec in sr_records:
            trend.append({
                "iter": rec.get("iter", 0),
                "sr": rec.get("solver_reward", 0),
            })
    else:
        # 完全没有 reward 数据，用 iter 数推断进度
        result["latest_iter"] = data[-1].get("iter", 0) if data else 0
        result["latest_sr"] = 0
        result["latest_source"] = "DR"
        result["latest_ar"] = 0
        result["early_sr"] = 0
        result["late_sr"] = 0
        result["sr_improvement"] = 0

    result["trend"] = trend
    result["has_step_reward"] = any("step_reward" in d for d in data[:50])
    result["reward_fixed"] = any(d.get("reward", 0) != 0 for d in data[-50:])
    result["status"] = "complete" if result["is_complete"] else "running"
    return result


def refresh_data():
    try:
        experiments = {}
        all_ids = []
        for method in METHODS:
            for gen in GENERATORS:
                for seed in SEEDS:
                    all_ids.append(f"{method}-{gen}-seed{seed}")
        for i, exp_id in enumerate(all_ids):
            if (i + 1) % 6 == 0 or i == 0:
                print(f"  [{i+1}/{len(all_ids)}] {exp_id}...")
            experiments[exp_id] = fetch_one(exp_id)
        cache["experiments"] = experiments
        cache["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cache["error"] = None
        cache["loading"] = False
    except Exception as e:
        cache["error"] = str(e)
        cache["loading"] = False


def background_refresh(interval=120):
    while True:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Refreshing...")
        refresh_data()
        n = len(cache["experiments"])
        ok = sum(1 for e in cache["experiments"].values() if e.get("is_complete"))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Done: {ok}/{n} complete")
        time.sleep(interval)


def make_svg_chart(trend, width=400, height=100):
    """生成 SVG 折线图"""
    if not trend or len(trend) < 2:
        return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle" fill="#484f58" font-size="12">无数据</text></svg>'

    max_iter = max(t["iter"] for t in trend) or 1
    max_sr = max(t["sr"] for t in trend) or 0.001
    pad = 5

    points = []
    for t in trend:
        x = pad + (t["iter"] / max_iter) * (width - 2 * pad)
        y = height - pad - (t["sr"] / max_sr) * (height - 2 * pad)
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)
    last = trend[-1]
    last_x = pad + (last["iter"] / max_iter) * (width - 2 * pad)
    last_y = height - pad - (last["sr"] / max_sr) * (height - 2 * pad)

    return f'''<svg width="{width}" height="{height}" style="background:#0d1117;border:1px solid #21262d;border-radius:4px">
  <polyline points="{polyline}" fill="none" stroke="#58a6ff" stroke-width="1.5" opacity="0.8"/>
  <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3" fill="#58a6ff"/>
  <text x="{width-pad}" y="12" text-anchor="end" fill="#484f58" font-size="9">max={max_sr:.4f}</text>
  <text x="{pad}" y="{height-2}" fill="#484f58" font-size="9">0</text>
  <text x="{width-pad}" y="{height-2}" text-anchor="end" fill="#484f58" font-size="9">{max_iter}</text>
</svg>'''


def generate_html():
    exps = cache["experiments"]
    last_update = cache["last_update"] or "加载中..."
    error = cache["error"]
    loading = cache["loading"]

    total = 36
    r3_complete = sum(1 for e in exps.values() if e.get("is_complete"))
    r3_with_data = sum(1 for e in exps.values() if e.get("is_r3"))

    # 找当前正在跑的（有第三轮数据但没 FINAL，按时间排序）
    running_list = []
    for eid, e in exps.items():
        if e.get("is_r3") and not e.get("is_complete") and e.get("latest_iter", 0) > 0:
            running_list.append((eid, e))
    running_list.sort(key=lambda x: x[1].get("dir_modified", ""), reverse=True)

    # 聚合表
    agg = {}
    for method in METHODS:
        agg[method] = {}
        for gen in GENERATORS:
            vals = []
            for seed in SEEDS:
                eid = f"{method}-{gen}-seed{seed}"
                if eid in exps and exps[eid].get("late_sr", 0) > 0:
                    vals.append(exps[eid]["late_sr"])
            agg[method][gen] = sum(vals) / len(vals) if vals else None

    html = f"""<!DOCTYPE html>
<html lang="zh"><head>
<meta charset="UTF-8"><meta http-equiv="refresh" content="35">
<title>AT-PC Monitor</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'SF Mono',Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:16px}}
h1{{color:#58a6ff;font-size:1.3em;margin-bottom:4px}}
.meta{{color:#8b949e;font-size:.8em;margin-bottom:16px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-bottom:20px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px}}
.card h3{{color:#58a6ff;font-size:.95em;margin-bottom:8px}}
.row{{display:flex;justify-content:space-between;padding:3px 0;font-size:.85em}}
.row .l{{color:#8b949e}}.row .v{{font-weight:bold}}
.g{{color:#3fb950}}.w{{color:#d29922}}.b{{color:#f85149}}.bl{{color:#58a6ff}}
.bar{{background:#21262d;border-radius:3px;height:6px;margin-top:6px}}
.bar-fill{{height:100%;border-radius:3px}}
table{{width:100%;border-collapse:collapse;font-size:.82em;margin-bottom:20px}}
th,td{{padding:6px 8px;border-bottom:1px solid #21262d;text-align:left}}
th{{color:#8b949e;font-weight:normal;text-transform:uppercase;font-size:.75em}}
.tag{{display:inline-block;padding:1px 6px;border-radius:10px;font-size:.7em}}
.t-ok{{background:#1b3a2d;color:#3fb950}}.t-run{{background:#2d1f00;color:#d29922}}.t-no{{background:#3d1418;color:#f85149}}
.agg td{{text-align:center}}.agg .m{{text-align:left;font-weight:bold}}
.section{{color:#58a6ff;font-size:1em;margin:16px 0 8px;border-bottom:1px solid #21262d;padding-bottom:4px}}
.chart-row{{display:flex;align-items:center;gap:12px}}
.chart-info{{min-width:180px;font-size:.82em}}
.chart-info .name{{color:#58a6ff;font-weight:bold;margin-bottom:4px}}
</style></head><body>
<h1>🐕 AT-PC 实验监控</h1>
<div class="meta">数据源: {BASE_URL} | 刷新: {last_update} | 自动30s{"" if not loading else " | ⏳ 加载中..."}{f' | <span class="b">错误: {error}</span>' if error else ""}</div>

<div class="cards">
<div class="card"><h3>📊 第三轮进度</h3>
<div class="row"><span class="l">完成</span><span class="v g">{r3_complete}/36</span></div>
<div class="row"><span class="l">有数据</span><span class="v">{r3_with_data}/36</span></div>
<div class="row"><span class="l">缺失</span><span class="v {"b" if 36-r3_with_data>0 else ""}">{36-r3_with_data}</span></div>
<div class="bar"><div class="bar-fill" style="width:{r3_complete/36*100:.0f}%;background:#3fb950"></div></div>
</div>
"""

    # 当前运行卡片
    if running_list:
        cur_eid, cur = running_list[0]
        pct = cur.get("latest_iter", 0) / 2000 * 100
        html += f"""<div class="card"><h3>🏃 正在运行</h3>
<div class="row"><span class="l">实验</span><span class="v bl">{cur_eid}</span></div>
<div class="row"><span class="l">进度</span><span class="v">{cur.get('latest_iter',0)}/2000 ({pct:.0f}%)</span></div>
<div class="row"><span class="l">solver_reward</span><span class="v">{cur.get('latest_sr',0):.4f}</span></div>
<div class="row"><span class="l">来源</span><span class="v">{cur.get('latest_source','?')}</span></div>
<div class="row"><span class="l">AR</span><span class="v">{cur.get('latest_ar',0):.1%}</span></div>
<div class="bar"><div class="bar-fill" style="width:{pct:.0f}%;background:#58a6ff"></div></div>
</div>
"""

    # 代码检测
    has_sr = sum(1 for e in exps.values() if e.get("has_step_reward"))
    rew_ok = sum(1 for e in exps.values() if e.get("reward_fixed"))
    html += f"""<div class="card"><h3>🔧 代码检测</h3>
<div class="row"><span class="l">step_reward</span><span class="v {"g" if has_sr else "b"}">{has_sr}/{len(exps)}</span></div>
<div class="row"><span class="l">reward修复</span><span class="v {"g" if rew_ok else "b"}">{rew_ok}/{len(exps)}</span></div>
</div></div>
"""

    # 聚合表
    html += '<div class="section">📈 Solver Reward 聚合 (末期均值)</div><table class="agg"><tr><th>方法</th>'
    for g in GENERATORS:
        html += f"<th>{g}</th>"
    html += "<th>均值</th></tr>"
    for method in METHODS:
        vals = []
        html += f'<tr><td class="m">{method.upper()}</td>'
        for gen in GENERATORS:
            v = agg[method][gen]
            if v is not None:
                vals.append(v)
                c = "g" if v > 0.02 else ("w" if v > 0.01 else "b")
                html += f'<td class="{c}">{v:.4f}</td>'
            else:
                html += '<td style="color:#484f58">-</td>'
        avg = sum(vals) / len(vals) if vals else 0
        html += f'<td style="font-weight:bold">{avg:.4f}</td></tr>'
    html += "</table>"

    # 曲线图区域 — 按方法分组
    html += '<div class="section">📉 训练曲线 (solver_reward)</div>'

    for method in METHODS:
        html += f'<div style="margin-bottom:16px"><div style="color:#8b949e;font-size:.85em;margin-bottom:6px">{method.upper()}</div>'
        html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:8px">'
        for gen in GENERATORS:
            for seed in SEEDS:
                eid = f"{method}-{gen}-seed{seed}"
                e = exps.get(eid, {})
                trend = e.get("trend", [])
                sr = e.get("latest_sr", 0)
                it = e.get("latest_iter", 0)
                src = e.get("latest_source", "-")
                ar = e.get("latest_ar", 0)
                complete = e.get("is_complete", False)

                status_color = "#3fb950" if complete else ("#d29922" if it > 0 else "#484f58")
                chart = make_svg_chart(trend, width=260, height=70)

                html += f'''<div class="chart-row" style="background:#161b22;border:1px solid #21262d;border-radius:4px;padding:8px">
<div class="chart-info">
<div class="name" style="color:{status_color}">{eid}</div>
<div style="color:#8b949e">iter {it}/2000 | sr={sr:.4f}</div>
<div style="color:#8b949e">src={src} | AR={ar:.0%}</div>
</div>{chart}</div>'''
        html += '</div></div>'

    # 详细表格
    html += '<div class="section">📋 全部实验</div><table>'
    html += '<tr><th>实验</th><th>状态</th><th>进度</th><th>solver_reward</th><th>改善</th><th>AR</th><th>来源</th><th>最后更新</th></tr>'

    for method in METHODS:
        for gen in GENERATORS:
            for seed in SEEDS:
                eid = f"{method}-{gen}-seed{seed}"
                e = exps.get(eid, {})
                if e.get("is_complete"):
                    tag = '<span class="tag t-ok">完成</span>'
                elif e.get("is_r3"):
                    tag = '<span class="tag t-run">运行中</span>'
                else:
                    tag = '<span class="tag t-no">缺失</span>'

                it = e.get("latest_iter", 0)
                sr = e.get("latest_sr", 0)
                imp = e.get("sr_improvement", 0)
                ar = e.get("latest_ar", 0)
                src = e.get("latest_source", "-")
                mod = e.get("dir_modified", "-")
                sc = "g" if sr > 0.02 else ("w" if sr > 0.01 else "")

                html += f'<tr><td>{eid}</td><td>{tag}</td><td>{it}/2000</td>'
                html += f'<td class="{sc}">{sr:.4f}</td><td>{imp:+.4f}</td>'
                html += f'<td>{ar:.0%}</td><td>{src}</td>'
                html += f'<td style="color:#8b949e;font-size:.75em">{mod}</td></tr>'

    html += "</table></body></html>"
    return html


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = generate_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/api/refresh":
            refresh_data()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


def main():
    port = 8050
    print(f"AT-PC 实验监控面板 v2")
    print(f"数据源: {BASE_URL}")
    print(f"打开: http://localhost:{port}")
    print(f"后台加载中...\n")

    t = threading.Thread(target=background_refresh, args=(120,), daemon=True)
    t.start()

    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
