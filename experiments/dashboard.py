"""
AT-PC 实验实时监控面板 v3 — 支持轮数切换

本地部署，从网盘服务器拉取数据，带 SVG 曲线图。
用法: python -m experiments.dashboard
浏览器打开 http://localhost:8050

轮数切换: 点击顶部按钮，首次点击某轮会触发加载（约30s）
"""
import json
import re
import time
import threading
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import requests

BASE_URL = "http://111.170.6.103:10005"
REMOTE_ROOT = "atpc_experiments"
METHODS = ["dr", "paired", "atpc"]
GENERATORS = ["G1", "G2", "G3", "G4"]
SEEDS = [42, 123, 456]
ALL_IDS = [f"{m}-{g}-seed{s}" for m in METHODS for g in GENERATORS for s in SEEDS]

cache = {
    "last_update": None,
    "raw_files": {},       # exp_id -> [file list]
    "round_list": [],      # [(round_num, label, start_ts, end_ts)]
    "rounds": {},          # round_num -> {exp_id -> parsed_data}
    "loading_round": None, # 正在加载的轮
    "error": None,
    "ready": False,
}
cache_lock = threading.Lock()


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


def detect_rounds(all_files):
    """从 FINAL 文件时间戳自动检测轮数"""
    all_finals = []
    for exp_id, files in all_files.items():
        for f in files:
            name = f.get("name", "") if isinstance(f, dict) else str(f)
            m = re.search(r'_FINAL_(\d{8}_\d{6})\.tar\.gz', name)
            if m:
                all_finals.append(m.group(1))
    if not all_finals:
        return []
    all_finals.sort()

    # 聚类: 间隔 > 4 小时 = 不同轮
    rounds = []
    group = [all_finals[0]]
    for ts in all_finals[1:]:
        prev = group[-1]
        # 粗略判断: 不同天 或 同天但小时差 > 4
        diff_day = ts[:8] != prev[:8]
        diff_hour = abs(int(ts[9:11]) - int(prev[9:11])) > 4 if not diff_day else True
        if diff_day or diff_hour:
            rounds.append(group)
            group = [ts]
        else:
            group.append(ts)
    rounds.append(group)

    result = []
    for i, grp in enumerate(rounds):
        date_str = f"{grp[0][4:6]}-{grp[0][6:8]}"
        n_finals = len(grp)
        label = f"第{i+1}轮 ({date_str}, {n_finals}个)"
        result.append((i + 1, label, grp[0], grp[-1]))
    return result


def load_round_data(round_num):
    """加载某一轮的数据 (下载 stats 文件)"""
    round_list = cache["round_list"]
    raw_files = cache["raw_files"]

    round_info = None
    for ri in round_list:
        if ri[0] == round_num:
            round_info = ri
            break
    if not round_info:
        return {}

    _, label, round_start, round_end = round_info

    # 确定该轮的 stats 文件编号范围
    # 每轮约 40 个 stats (2000 iter / 50 iter per upload)
    # 但编号是累积的
    expected_base = (round_num - 1) * 40

    exps = {}
    for i, exp_id in enumerate(ALL_IDS):
        files = raw_files.get(exp_id, [])
        names = [f.get("name", "") if isinstance(f, dict) else str(f) for f in files]

        # 该轮有 FINAL?
        has_final = any(
            round_start <= m.group(1) <= round_end
            for n in names
            for m in [re.search(r'_FINAL_(\d{8}_\d{6})\.tar\.gz', n)]
            if m
        )

        # 该轮有 checkpoint?
        checkpoints = []
        for n in names:
            m = re.search(r'_iter(\d+)_(\d{8}_\d{6})\.tar\.gz', n)
            if m and round_start <= m.group(2) <= round_end:
                checkpoints.append(int(m.group(1)))

        # 找 stats 文件
        stats_nums = []
        for n in names:
            m = re.match(r'training_stats_(\d+)\.json', n)
            if m:
                stats_nums.append(int(m.group(1)))
        stats_nums.sort()

        # 选该轮的 stats: 编号在 [expected_base, expected_base+44) 范围
        round_stats = [n for n in stats_nums if expected_base <= n < expected_base + 45]

        stats_file = None
        if round_stats:
            stats_file = f"training_stats_{max(round_stats)}.json"
        elif "training_stats.json" in names and not has_final:
            stats_file = "training_stats.json"
        elif round_num == 1 and any(n < 45 for n in stats_nums):
            stats_file = f"training_stats_{max(n for n in stats_nums if n < 45)}.json"

        result = {
            "exp_id": exp_id,
            "is_complete": has_final,
            "has_data": has_final or len(checkpoints) > 0,
            "latest_iter": 0, "latest_sr": 0, "latest_source": "-",
            "latest_ar": 0, "latest_regret": 0,
            "early_sr": 0, "late_sr": 0, "sr_improvement": 0,
            "trend": [], "has_step_reward": False, "reward_fixed": False,
        }

        if stats_file:
            data = api_download(f"{REMOTE_ROOT}/{exp_id}/{stats_file}")
            if data:
                result["stats_file"] = stats_file
                result["total_records"] = len(data)

                sr_records = [d for d in data if d.get("solver_reward", 0) > 0]
                if sr_records:
                    result["latest_iter"] = sr_records[-1].get("iter", 0)
                    result["latest_sr"] = sr_records[-1].get("solver_reward", 0)
                    result["latest_source"] = sr_records[-1].get("source", "DR")
                    result["latest_ar"] = sr_records[-1].get("accept_rate", 0)
                    result["latest_regret"] = sr_records[-1].get("regret", 0)
                    n5 = max(len(sr_records) // 5, 1)
                    result["early_sr"] = sum(d["solver_reward"] for d in sr_records[:n5]) / n5
                    result["late_sr"] = sum(d["solver_reward"] for d in sr_records[-n5:]) / n5
                    result["sr_improvement"] = result["late_sr"] - result["early_sr"]
                    result["trend"] = [{"iter": r.get("iter", 0), "sr": r.get("solver_reward", 0)} for r in sr_records]
                else:
                    result["latest_iter"] = data[-1].get("iter", 0) if data else 0

                result["has_step_reward"] = any("step_reward" in d for d in data[:50])
                result["reward_fixed"] = any(d.get("reward", 0) != 0 for d in data[-50:])

        result["status"] = "complete" if has_final else ("running" if result["latest_iter"] > 0 else "no_data")
        exps[exp_id] = result

    return exps


def refresh_file_lists():
    """只刷新文件列表和轮数检测 (快速)"""
    raw_files = {}
    for i, exp_id in enumerate(ALL_IDS):
        if (i + 1) % 12 == 0 or i == 0:
            print(f"  [{i+1}/{len(ALL_IDS)}] listing {exp_id}...")
        raw_files[exp_id] = api_list(f"{REMOTE_ROOT}/{exp_id}")

    with cache_lock:
        cache["raw_files"] = raw_files
        cache["round_list"] = detect_rounds(raw_files)
        cache["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cache["error"] = None

    return cache["round_list"]


def ensure_round_loaded(round_num):
    """确保某轮数据已加载"""
    with cache_lock:
        if round_num in cache["rounds"]:
            return True
        if cache["loading_round"] == round_num:
            return False  # 正在加载

    with cache_lock:
        cache["loading_round"] = round_num

    print(f"  加载第{round_num}轮数据...")
    data = load_round_data(round_num)

    with cache_lock:
        cache["rounds"][round_num] = data
        cache["loading_round"] = None
        cache["ready"] = True

    n_ok = sum(1 for e in data.values() if e.get("is_complete"))
    print(f"  第{round_num}轮: {n_ok}/36 完成")
    return True


def background_refresh(interval=120):
    """后台定时刷新"""
    while True:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Refreshing file lists...")
        round_list = refresh_file_lists()
        print(f"  检测到 {len(round_list)} 轮")
        for rn, label, s, e in round_list:
            print(f"    {label}: {s} ~ {e}")

        # 自动加载最新一轮
        if round_list:
            latest = round_list[-1][0]
            ensure_round_loaded(latest)

        time.sleep(interval)


def make_svg_chart(trend, width=400, height=100):
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
    lx = pad + (last["iter"] / max_iter) * (width - 2 * pad)
    ly = height - pad - (last["sr"] / max_sr) * (height - 2 * pad)
    return f'''<svg width="{width}" height="{height}" style="background:#0d1117;border:1px solid #21262d;border-radius:4px">
  <polyline points="{polyline}" fill="none" stroke="#58a6ff" stroke-width="1.5" opacity="0.8"/>
  <circle cx="{lx:.1f}" cy="{ly:.1f}" r="3" fill="#58a6ff"/>
  <text x="{width-pad}" y="12" text-anchor="end" fill="#484f58" font-size="9">max={max_sr:.4f}</text>
  <text x="{pad}" y="{height-2}" fill="#484f58" font-size="9">0</text>
  <text x="{width-pad}" y="{height-2}" text-anchor="end" fill="#484f58" font-size="9">{max_iter}</text>
</svg>'''


def generate_html(selected_round=None):
    with cache_lock:
        round_list = list(cache.get("round_list", []))
        rounds_data = cache.get("rounds", {})
        last_update = cache["last_update"] or "加载中..."
        error = cache["error"]
        loading_round = cache["loading_round"]

    if not round_list:
        return """<!DOCTYPE html><html><head><meta charset="UTF-8"><meta http-equiv="refresh" content="5">
<title>AT-PC Monitor</title><style>body{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:40px;text-align:center}
</style></head><body><h2 style="color:#58a6ff">🐕 AT-PC 实验监控</h2><p>⏳ 正在从服务器加载数据...</p></body></html>"""

    # 默认最新轮
    if selected_round is None or selected_round not in [r[0] for r in round_list]:
        selected_round = round_list[-1][0]

    # 确保该轮数据已加载
    if selected_round not in rounds_data:
        # 触发后台加载
        threading.Thread(target=ensure_round_loaded, args=(selected_round,), daemon=True).start()
        round_label = ""
        for rn, label, _, _ in round_list:
            if rn == selected_round:
                round_label = label
        round_buttons = ""
        for rn, label, _, _ in round_list:
            active = "active" if rn == selected_round else ""
            round_buttons += f'<a href="/?round={rn}" class="rbtn {active}">{label}</a> '
        return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta http-equiv="refresh" content="3">
<title>AT-PC Monitor - {round_label}</title><style>
body{{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:20px}}
.round-bar{{display:flex;gap:8px;margin:16px 0;flex-wrap:wrap}}
.rbtn{{padding:6px 14px;border-radius:16px;font-size:.82em;text-decoration:none;color:#8b949e;background:#161b22;border:1px solid #30363d}}
.rbtn.active{{color:#fff;background:#1f6feb;border-color:#1f6feb}}
</style></head><body>
<h2 style="color:#58a6ff">🐕 AT-PC 实验监控</h2>
<div class="round-bar">{round_buttons}</div>
<p>⏳ 正在加载{round_label}数据 (约30秒)...</p></body></html>"""

    exps = rounds_data.get(selected_round, {})
    round_label = ""
    for rn, label, _, _ in round_list:
        if rn == selected_round:
            round_label = label

    total = 36
    n_complete = sum(1 for e in exps.values() if e.get("is_complete"))
    n_with_data = sum(1 for e in exps.values() if e.get("has_data"))

    running_list = [(eid, e) for eid, e in exps.items()
                    if not e.get("is_complete") and e.get("latest_iter", 0) > 0]
    running_list.sort(key=lambda x: x[1].get("latest_iter", 0), reverse=True)

    agg = {}
    for method in METHODS:
        agg[method] = {}
        for gen in GENERATORS:
            vals = [exps[f"{method}-{gen}-seed{s}"]["late_sr"]
                    for s in SEEDS
                    if f"{method}-{gen}-seed{s}" in exps and exps[f"{method}-{gen}-seed{s}"].get("late_sr", 0) > 0]
            agg[method][gen] = sum(vals) / len(vals) if vals else None

    round_buttons = ""
    for rn, label, _, _ in round_list:
        active = "active" if rn == selected_round else ""
        loaded = " ✓" if rn in rounds_data else ""
        round_buttons += f'<a href="/?round={rn}" class="rbtn {active}">{label}{loaded}</a> '

    html = f"""<!DOCTYPE html><html lang="zh"><head>
<meta charset="UTF-8"><meta http-equiv="refresh" content="35">
<title>AT-PC Monitor - {round_label}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'SF Mono',Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:16px}}
h1{{color:#58a6ff;font-size:1.3em;margin-bottom:4px}}
.meta{{color:#8b949e;font-size:.8em;margin-bottom:12px}}
.round-bar{{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}}
.rbtn{{display:inline-block;padding:6px 14px;border-radius:16px;font-size:.82em;
  text-decoration:none;color:#8b949e;background:#161b22;border:1px solid #30363d;cursor:pointer;transition:all .2s}}
.rbtn:hover{{color:#c9d1d9;border-color:#58a6ff}}
.rbtn.active{{color:#fff;background:#1f6feb;border-color:#1f6feb}}
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
<div class="meta">数据源: {BASE_URL} | 刷新: {last_update} | 自动35s{f' | <span class="b">错误: {error}</span>' if error else ""}</div>
<div class="round-bar">{round_buttons}</div>

<div class="cards">
<div class="card"><h3>📊 {round_label}</h3>
<div class="row"><span class="l">完成</span><span class="v g">{n_complete}/36</span></div>
<div class="row"><span class="l">有数据</span><span class="v">{n_with_data}/36</span></div>
<div class="row"><span class="l">缺失</span><span class="v {"b" if 36-n_with_data>0 else ""}">{36-n_with_data}</span></div>
<div class="bar"><div class="bar-fill" style="width:{n_complete/36*100:.0f}%;background:#3fb950"></div></div>
</div>"""

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
</div>"""

    has_sr = sum(1 for e in exps.values() if e.get("has_step_reward"))
    rew_ok = sum(1 for e in exps.values() if e.get("reward_fixed"))
    html += f"""<div class="card"><h3>🔧 代码检测</h3>
<div class="row"><span class="l">step_reward字段</span><span class="v {"g" if has_sr else "b"}">{has_sr}/36</span></div>
<div class="row"><span class="l">reward非零</span><span class="v {"g" if rew_ok else "b"}">{rew_ok}/36</span></div>
</div></div>"""

    # 聚合表
    html += f'<div class="section">📈 Solver Reward 聚合 (末期均值)</div><table class="agg"><tr><th>方法</th>'
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

    # 曲线图
    html += f'<div class="section">📉 训练曲线 (solver_reward)</div>'
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
                sc = "#3fb950" if complete else ("#d29922" if it > 0 else "#484f58")
                chart = make_svg_chart(trend, width=260, height=70)
                html += f'''<div class="chart-row" style="background:#161b22;border:1px solid #21262d;border-radius:4px;padding:8px">
<div class="chart-info"><div class="name" style="color:{sc}">{eid}</div>
<div style="color:#8b949e">iter {it}/2000 | sr={sr:.4f}</div>
<div style="color:#8b949e">src={src} | AR={ar:.0%}</div>
</div>{chart}</div>'''
        html += '</div></div>'

    # 详细表格
    html += f'<div class="section">📋 全部实验</div><table>'
    html += '<tr><th>实验</th><th>状态</th><th>进度</th><th>solver_reward</th><th>改善</th><th>AR</th><th>来源</th></tr>'
    for method in METHODS:
        for gen in GENERATORS:
            for seed in SEEDS:
                eid = f"{method}-{gen}-seed{seed}"
                e = exps.get(eid, {})
                if e.get("is_complete"):
                    tag = '<span class="tag t-ok">完成</span>'
                elif e.get("latest_iter", 0) > 0:
                    tag = '<span class="tag t-run">运行中</span>'
                else:
                    tag = '<span class="tag t-no">无数据</span>'
                it = e.get("latest_iter", 0)
                sr = e.get("latest_sr", 0)
                imp = e.get("sr_improvement", 0)
                ar = e.get("latest_ar", 0)
                src = e.get("latest_source", "-")
                sc = "g" if sr > 0.02 else ("w" if sr > 0.01 else "")
                html += f'<tr><td>{eid}</td><td>{tag}</td><td>{it}/2000</td>'
                html += f'<td class="{sc}">{sr:.4f}</td><td>{imp:+.4f}</td>'
                html += f'<td>{ar:.0%}</td><td>{src}</td></tr>'
    html += "</table></body></html>"
    return html


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            qs = parse_qs(parsed.query)
            selected_round = None
            if "round" in qs:
                try:
                    selected_round = int(qs["round"][0])
                except:
                    pass
            html = generate_html(selected_round)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        elif parsed.path == "/api/refresh":
            threading.Thread(target=refresh_file_lists, daemon=True).start()
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
    print(f"AT-PC 实验监控面板 v3 (支持轮数切换)")
    print(f"数据源: {BASE_URL}")
    print(f"打开: http://localhost:{port}")
    print(f"后台加载中...\n")

    t = threading.Thread(target=background_refresh, args=(120,), daemon=True)
    t.start()

    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
