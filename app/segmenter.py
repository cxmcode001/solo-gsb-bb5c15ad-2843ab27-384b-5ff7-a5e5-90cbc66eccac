"""字幕断句核心:全局动态规划选择分段,使加权总代价最低。

代价项(均可加权):
  - 镜头切点偏差: 分段边界偏离最近镜头切点的距离(归一化), 以及分段跨越切点的惩罚
  - 行长不均:     两行长度差(归一化)
  - 不自然停顿:   断点处词间间隔过小(归一化)

硬约束:
  - 每段最多 2 行, 每行不超过 max_chars_per_line 个字符
  - 阅读速度(字符/秒)不超过 max_cps
  - 每段时长在 [min_duration, max_duration] 内
  - 不可拆短语(phrase)作为原子单元, 不得被分段或换行拆开
  - 边界约束(boundaries): 指定词间隙必须断开(must_break)或禁止断开(no_break),
    在动态规划中作为硬条件生效; 不可拆短语内部的词间隙不允许设置
  - 锁定段(lock): 固定词范围与内部换行位置, 重算时只优化其余区间;
    锁定段的两个段边界转为 must_break、内部词间隙转为 no_break,
    锁内唯一合法分段边(锁定边)按锁定换行取行长不均代价。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    max_chars_per_line: int = 42
    max_cps: float = 20.0
    min_duration: float = 0.8
    max_duration: float = 7.0
    pause_threshold: float = 0.3      # 词间间隔小于该值视为不自然断点
    shot_dev_cap: float = 0.5         # 镜头切点偏差归一化上限(秒)
    crossing_penalty: float = 1.0     # 分段跨越一个镜头切点的惩罚(归一化)
    w_shot: float = 1.0
    w_imbalance: float = 1.0
    w_pause: float = 1.0


@dataclass
class Unit:
    """原子单元: 单个词或一个不可拆短语。"""
    word_indices: list[int]
    text: str
    start: float
    end: float


def build_units(words: list[dict], phrases: list[list[int]]) -> list[Unit]:
    """把词序列与不可拆短语(词下标区间)合并为原子单元序列。"""
    n = len(words)
    owner: dict[int, list[int]] = {}
    for ph in phrases:
        if not ph:
            continue
        if ph != list(range(ph[0], ph[-1] + 1)):
            raise ValueError(f"短语下标不连续: {ph}")
        for i in ph:
            if i < 0 or i >= n:
                raise ValueError(f"短语下标越界: {i}")
            if i in owner:
                raise ValueError(f"短语重叠于词 {i}")
        for i in ph:
            owner[i] = ph

    units: list[Unit] = []
    i = 0
    while i < n:
        if i in owner:
            ph = owner[i]
            units.append(Unit(
                word_indices=ph,
                text=" ".join(words[k]["text"] for k in ph),
                start=words[ph[0]]["start"],
                end=words[ph[-1]]["end"],
            ))
            i = ph[-1] + 1
        else:
            units.append(Unit(
                word_indices=[i],
                text=words[i]["text"],
                start=words[i]["start"],
                end=words[i]["end"],
            ))
            i += 1
    return units


def validate_boundaries(boundaries: list[dict], n_words: int,
                        phrases: list[list[int]], units: list[Unit]):
    """校验词间隙边界约束, 返回 (must_break, no_break) 两个单元边界集合。

    边界下标 k 表示词 k-1 与词 k 之间的间隙(1 <= k <= n_words-1);
    单元边界 u 位于单元 u-1 与单元 u 之间。不可拆短语内部的间隙不得设置。
    """
    mb: set[int] = set()
    nb: set[int] = set()
    if not boundaries:
        return mb, nb
    inner: set[int] = set()  # 短语内部的词间隙
    for ph in phrases:
        for k in range(ph[0] + 1, ph[-1] + 1):
            inner.add(k)
    seen: dict[int, str] = {}
    for b in boundaries:
        k, t = b.get("index"), b.get("type")
        if t not in ("must_break", "no_break"):
            raise ValueError(f"未知边界约束类型: {t}")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > n_words - 1:
            raise ValueError(f"边界下标越界: {k}(合法范围 1–{n_words - 1})")
        if k in inner:
            raise ValueError(f"边界 {k} 位于不可拆短语内部, 不能设置边界约束")
        if k in seen and seen[k] != t:
            raise ValueError(f"边界 {k} 同时被设置为必须断开与禁止断开")
        seen[k] = t
    # 词 k 不在短语内部, 必为某单元首词
    start_to_unit = {u.word_indices[0]: i for i, u in enumerate(units)}
    for k, t in seen.items():
        (mb if t == "must_break" else nb).add(start_to_unit[k])
    return mb, nb


def layout_segment(units: list[Unit], a: int, b: int, max_chars: int):
    """尝试把 units[a:b] 排成至多两行, 返回 (lines, imbalance, split) 或 None。

    单元不可拆行; 枚举所有换行位置, 取行长差最小者。split 为换行处的单元
    边界下标(相对 units 的全局下标): 两行时第一行包含 units[a:split],
    单行时为 None。
    """
    texts = [u.text for u in units[a:b]]
    if any(len(t) > max_chars for t in texts):
        return None

    def join(ts):
        return " ".join(ts)

    best = None
    # 单行
    one = join(texts)
    if len(one) <= max_chars:
        best = ([one], 0.0, None)
    # 两行
    for k in range(1, len(texts)):
        l1, l2 = join(texts[:k]), join(texts[k:])
        if len(l1) <= max_chars and len(l2) <= max_chars:
            imb = abs(len(l1) - len(l2)) / max_chars
            if best is None or imb < best[1]:
                best = ([l1, l2], imb, a + k)
    return best


def _timing_violations(units: list[Unit], a: int, b: int, cfg: Config) -> tuple[list[str], float, int]:
    """段 units[a..b](闭区间)的时长/阅读速度违例; 同时返回时长与字符数。"""
    seg = units[a:b + 1]
    dur = seg[-1].end - seg[0].start
    chars = sum(len(u.text) for u in seg) + len(seg) - 1
    v = []
    if dur > cfg.max_duration + 1e-9:
        v.append(f"时长 {dur:.2f}s 超过上限 {cfg.max_duration}s")
    if dur < cfg.min_duration - 1e-9:
        v.append(f"时长 {dur:.2f}s 低于下限 {cfg.min_duration}s")
    if dur > 1e-9 and chars / dur > cfg.max_cps + 1e-9:
        v.append(f"阅读速度 {chars / dur:.1f} 字/秒 超过上限 {cfg.max_cps}")
    return v, dur, chars


def segment_violations(units: list[Unit], a: int, b: int, cfg: Config) -> list[str]:
    """列出段 units[a..b](闭区间)违反的硬约束(空列表表示可行)。"""
    v, _, _ = _timing_violations(units, a, b, cfg)
    if layout_segment(units, a, b + 1, cfg.max_chars_per_line) is None:
        v.append(f"无法在 2 行 x {cfg.max_chars_per_line} 字符内排下")
    return v


def boundary_shot_cost(t: float, shots: list[float], cfg: Config) -> float:
    if not shots:
        return 0.0
    d = min(abs(t - s) for s in shots)
    return min(d, cfg.shot_dev_cap) / cfg.shot_dev_cap


def crossing_count(start: float, end: float, shots: list[float]) -> int:
    return sum(1 for s in shots if start + 1e-9 < s < end - 1e-9)


def pause_cost(gap: float, cfg: Config) -> float:
    if gap >= cfg.pause_threshold:
        return 0.0
    return min(1.0, (cfg.pause_threshold - gap) / cfg.pause_threshold)


def _edge_cost(units: list[Unit], i: int, j: int, p: dict,
               cfg: Config, shots: list[float]) -> float:
    """转移边 i→j(段 units[i:j])的加权代价。段首边界(非全文开头)计入切点偏差与停顿。"""
    c_shot = p["crossing"] * cfg.crossing_penalty
    c_pause = 0.0
    if i > 0:
        c_shot += boundary_shot_cost(units[i].start, shots, cfg)
        c_pause = pause_cost(units[i].start - units[i - 1].end, cfg)
    return (cfg.w_shot * c_shot
            + cfg.w_imbalance * p["imbalance"]
            + cfg.w_pause * c_pause)


def _run_dp_kbest(units: list[Unit], feasible: list[list[bool]], cost_parts: list[list[dict]],
                  cfg: Config, shots: list[float], mb: set[int], nb: set[int],
                  k: int = 1):
    """K-best 全局 DP: 在同一个分段 DAG(切点 = 单元边界)上求总代价最低的 k 条路径。

    路径节点为单元边界位置 0..n, 边 i→j 对应段 units[i:j]; 路径完全由其节点
    (分段边界)序列决定。每个位置 j 保留至多 k 个 (累计代价, 前驱位置, 前驱名次),
    枚举全部合法入边后取代价最低的 k 个。DAG 中每条不同路径的生成方式唯一
    (前驱位置不同则路径不同; 前驱位置相同时前驱名次不同则前缀路径不同),
    因此候选天然互不重复, 无需再按边界序列判重。
    """
    INF = float("inf")
    n = len(units)
    # lo[j]: 转移 i→j 合法要求 i >= lo[j](不得跨过小于 j 的 must_break 边界)
    lo = [0] * (n + 1)
    for j in range(1, n + 1):
        lo[j] = max(lo[j - 1], j - 1 if (j - 1) in mb else 0)
    # states[j] = 到达位置 j 的至多 k 个最优状态, 按累计代价升序
    states: list[list[tuple[float, int, int]]] = [[] for _ in range(n + 1)]
    states[0] = [(0.0, -1, -1)]
    for j in range(1, n + 1):
        cand: list[tuple[float, int, int]] = []
        for i in range(lo[j], j):
            if i and i in nb:
                continue
            if not feasible[i][j - 1]:
                continue
            edge = _edge_cost(units, i, j, cost_parts[i][j - 1], cfg, shots)
            for rank, (cost, _pi, _pr) in enumerate(states[i]):
                cand.append((cost + edge, i, rank))
        cand.sort(key=lambda x: x[0])
        states[j] = cand[:k]
    return states


def _run_dp(units: list[Unit], feasible: list[list[bool]], cost_parts: list[list[dict]],
            cfg: Config, shots: list[float], mb: set[int], nb: set[int]):
    """单最优 DP(供冲突诊断使用): 返回与旧版兼容的 dp / prev 数组。

    边界约束(单元边界 u 位于单元 u-1 与单元 u 之间)作为硬条件:
      - u ∈ mb: 必须断开 —— 任何分段不得跨越该边界(转移 i→j 不允许 i < u < j)
      - u ∈ nb: 禁止断开 —— 单元 u-1 与 u 必须同段(不允许以 u 为段首)
    """
    states = _run_dp_kbest(units, feasible, cost_parts, cfg, shots, mb, nb, k=1)
    INF = float("inf")
    dp = [st[0][0] if st else INF for st in states]
    prev = [st[0][1] if st else -1 for st in states]
    return dp, prev


def _backtrack(states, rank: int = 0) -> list[int]:
    """从 K-best DP 终态(states[n][rank])回溯分段边界节点序列 [0, ..., n]。"""
    n = len(states) - 1
    path = [n]
    j, r = n, rank
    while j > 0:
        _, i, pr = states[j][r]
        path.append(i)
        j, r = i, pr
    path.reverse()
    return path


def _build_segments(units: list[Unit], path: list[int], cost_parts: list[list[dict]],
                    cfg: Config, shots: list[float], lock_info: dict | None = None):
    """按边界节点序列构造分段结果, 返回 (segments, 未取整总代价)。

    锁定边(lock_info["edges"])对应的段固定使用锁定换行, 并带 locked 标记;
    每段额外返回 line_break(内部换行处的词间隙下标, 单行段为 null), 供前端
    直接以该段发起锁定。
    """
    lock_edges = lock_info["edges"] if lock_info else {}
    segments = []
    total = 0.0
    for x, y in zip(path, path[1:]):
        a, b = x, y - 1  # 边 x→y 对应段 units[x:y]
        p = cost_parts[a][b]
        start, end = units[a].start, units[b].end
        c_shot = p["crossing"] * cfg.crossing_penalty
        c_pause = 0.0
        if a > 0:
            c_shot += boundary_shot_cost(start, shots, cfg)
            c_pause = pause_cost(units[a].start - units[a - 1].end, cfg)
        c_shot_w = cfg.w_shot * c_shot
        c_imb_w = cfg.w_imbalance * p["imbalance"]
        c_pause_w = cfg.w_pause * c_pause
        seg_total = c_shot_w + c_imb_w + c_pause_w
        total += seg_total
        su = p.get("split_unit")
        seg = {
            "word_range": [units[a].word_indices[0], units[b].word_indices[-1]],
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(p["duration"], 3),
            "cps": round(p["cps"], 1),
            "lines": p["lines"],
            "line_break": units[su].word_indices[0] if su is not None else None,
            "cost": {
                "shot": round(c_shot_w, 4),
                "imbalance": round(c_imb_w, 4),
                "pause": round(c_pause_w, 4),
                "total": round(seg_total, 4),
            },
        }
        if lock_edges.get(a) == y:
            seg["locked"] = True
        segments.append(seg)
    return segments, total


def _lock_conflict(lock: dict, units: list[Unit], n_words: int,
                   reasons: list[str], ua: int | None = None) -> dict:
    """构造锁定段失效的冲突响应。ua 给出可定位的单元下标, 越界时按词范围兜底。"""
    wa, wb = int(lock["word_range"][0]), int(lock["word_range"][1])
    if ua is not None:
        wi = units[ua].word_indices
        word_range = [wi[0], wi[-1]]
        start, end, text = units[ua].start, units[ua].end, units[ua].text
    else:
        wa_c = min(max(wa, 0), n_words - 1)
        wb_c = min(max(wb, wa_c), n_words - 1)
        word_range = [wa_c, wb_c]
        start = end = units[wa_c].start
        text = units[wa_c].text
    return {
        "ok": False,
        "conflict": {
            "kind": "lock",
            "word_range": word_range,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "reasons": reasons,
            "lock": {
                "word_range": [wa, wb],
                "line_break": lock.get("line_break"),
            },
        },
    }


def validate_locks(locks: list[dict], words: list[dict],
                   units: list[Unit], cfg: Config, user_mb: set[int], user_nb: set[int]):
    """解析并校验锁定段, 返回 (lock_info, conflict) 二选一(无锁定时 lock_info 为 None)。

    成功时 lock_info = {"mb": 锁边界 must_break 单元集合, "edges": {a: b, ...},
    "layout": {(a, b): {...}}}, 其中 (a, b) 为锁定段对应的唯一分段边
    (单元下标, 段为 units[a:b]); layout 给出锁定换行的 lines/imbalance,
    由调用方写入 cost_parts[a][b-1]。

    依次检查(返回首个问题锁): 词范围合法且与短语/单元边界对齐、内部换行合法、
    锁之间不重叠、与用户边界约束不冲突、锁定段在当前硬约束下仍可排下。
    """
    if not locks:
        return None, None
    n_words = len(words)
    n_units = len(units)
    # 词下标 -> 所属单元下标; unit_start = 作为单元首词的词下标(合法段/行边界)
    word_unit = [0] * n_words
    unit_start: set[int] = set()
    for ui, u in enumerate(units):
        for w in u.word_indices:
            word_unit[w] = ui
        unit_start.add(u.word_indices[0])
    mb_words = {units[u].word_indices[0] for u in user_mb}
    nb_words = {units[u].word_indices[0] for u in user_nb}

    parsed = []
    for raw in locks:
        rng = raw.get("word_range")
        lb = raw.get("line_break")
        wa, wb = int(rng[0]), int(rng[1])
        lock = {"word_range": [wa, wb], "line_break": lb}
        parsed.append((wa, wb, lb, lock))
    parsed.sort(key=lambda t: (t[0], t[1]))

    taken: list[tuple[int, int]] = []  # 已占用词区间
    edges: dict[int, int] = {}
    layout: dict[tuple[int, int], dict] = {}
    lock_mb: set[int] = set()
    for wa, wb, lb, lock in parsed:
        reasons: list[str] = []
        ua = None
        if 0 <= wa < n_words and 0 <= wb < n_words:
            ua = word_unit[wa]
        # 1) 词范围合法性
        if wa < 0 or wb < 0 or wa >= n_words or wb >= n_words:
            reasons.append(f"锁定段词范围 [{wa}, {wb}] 越界(合法范围 0–{n_words - 1}),"
                           f"词序列已变化,请解锁后重新锁定")
        elif wa > wb:
            reasons.append(f"锁定段词范围 [{wa}, {wb}] 非法:起点 {wa} 大于终点 {wb}")
        elif wa not in unit_start:
            reasons.append(f"锁定段起点词 {wa} 位于不可拆短语内部,不能作为段边界,"
                           f"请解锁后调整(短语已变化)")
        elif (wb + 1) < n_words and (wb + 1) not in unit_start:
            reasons.append(f"锁定段终点词 {wb} 位于不可拆短语内部,不能作为段边界,"
                           f"请解锁后调整(短语已变化)")
        if reasons:
            return None, _lock_conflict(lock, units, n_words, reasons, ua)

        a, b = word_unit[wa], word_unit[wb]  # 单元下标(闭区间段 units[a..b])
        # 2) 内部换行
        split_unit = None
        if lb is not None:
            lb = int(lb)
            lock["line_break"] = lb
            if lb <= wa or lb > wb:
                reasons.append(f"锁定段(词 {wa}–{wb})的内部换行 {lb} 不在其词间隙范围"
                               f"(合法为词 {wa + 1}–{wb} 左侧间隙),词序列已变化,请解锁后重锁")
            elif lb not in unit_start:
                reasons.append(f"锁定段的内部换行词间隙 {lb}(词 {lb - 1} 与词 {lb} 之间)"
                               f"位于不可拆短语内部,不能换行,请解锁后调整")
            else:
                split_unit = word_unit[lb]
        if reasons:
            return None, _lock_conflict(lock, units, n_words, reasons, a)

        # 3) 与已存在的锁重叠(首尾相接不算重叠)
        for pa, pb in taken:
            if wa <= pb and pa <= wb:
                reasons.append(f"锁定段(词 {wa}–{wb})与已锁定段(词 {pa}–{pb})重叠,"
                               f"请先解锁其中一段")
                break
        if reasons:
            return None, _lock_conflict(lock, units, n_words, reasons, a)

        # 4) 与用户边界约束冲突
        if wa in nb_words:
            reasons.append(f"锁定段起点(词 {wa - 1} 与词 {wa} 之间)被设置为禁止断开,"
                           f"与锁定段边界必须断开冲突")
        if (wb + 1) < n_words and (wb + 1) in nb_words:
            k = wb + 1
            reasons.append(f"锁定段终点(词 {k - 1} 与词 {k} 之间)被设置为禁止断开,"
                           f"与锁定段边界必须断开冲突")
        for k in range(wa + 1, wb + 1):
            if k in mb_words:
                reasons.append(f"锁定段内部词间隙 {k}(词 {k - 1} 与词 {k} 之间)"
                               f"被设置为必须断开,与锁定段词范围冲突")
        if lb is not None and int(lb) in nb_words:
            reasons.append(f"锁定的内部换行处(词 {int(lb) - 1} 与词 {int(lb)} 之间)"
                           f"被设置为禁止断开,与锁定换行冲突")
        if reasons:
            return None, _lock_conflict(lock, units, n_words, reasons, a)

        # 5) 锁定段在当前硬约束(行宽/行数、时长、阅读速度)下仍可行
        tv, _, _ = _timing_violations(units, a, b, cfg)
        reasons.extend(tv)
        if split_unit is None:
            one = " ".join(u.text for u in units[a:b + 1])
            if len(one) <= cfg.max_chars_per_line:
                lines, imb = [one], 0.0
            else:
                reasons.append(f"锁定为单行后长度 {len(one)} 超过单行上限 "
                               f"{cfg.max_chars_per_line} 字符")
                lines, imb = None, None
        else:
            su = split_unit
            l1 = " ".join(u.text for u in units[a:su])
            l2 = " ".join(u.text for u in units[su:b + 1])
            if any(len(u.text) > cfg.max_chars_per_line for u in units[a:b + 1]):
                reasons.append(f"锁定段内存在长度超过单行上限 {cfg.max_chars_per_line} "
                               f"字符的词/短语")
            elif len(l1) > cfg.max_chars_per_line or len(l2) > cfg.max_chars_per_line:
                reasons.append(f"锁定的内部换行(词间隙 {lb})使行长 "
                               f"{max(len(l1), len(l2))} 超过单行上限 "
                               f"{cfg.max_chars_per_line} 字符")
            else:
                lines, imb = [l1, l2], abs(len(l1) - len(l2)) / cfg.max_chars_per_line
        if reasons:
            return None, _lock_conflict(lock, units, n_words, reasons, a)

        edges[a] = b + 1  # 段 units[a:b+1] 对应边 a→(b+1)
        lock_mb.add(a)
        if b + 1 < n_units:
            lock_mb.add(b + 1)
        taken.append((wa, wb))
        layout[(a, b + 1)] = {"lines": lines, "imbalance": imb,
                              "split_unit": split_unit}

    return {"mb": lock_mb, "edges": edges, "layout": layout}, None


def solve(words: list[dict], phrases: list[list[int]], shots: list[float],
          cfg: Config, boundaries: list[dict] | None = None,
          alternatives: int = 1, locks: list[dict] | None = None) -> dict:
    if not words:
        if locks:
            # 空词序列上不存在可锁定的词范围: 返回首个锁的失效冲突
            first = min(locks, key=lambda l: (int(l["word_range"][0]),
                                              int(l["word_range"][1])))
            wa, wb = int(first["word_range"][0]), int(first["word_range"][1])
            return {
                "ok": False,
                "conflict": {
                    "kind": "lock",
                    "word_range": [0, 0],
                    "start": 0.0, "end": 0.0, "text": "",
                    "reasons": [f"锁定段词范围 [{wa}, {wb}] 越界:词序列为空,请先解锁"],
                    "lock": {"word_range": [wa, wb],
                             "line_break": first.get("line_break")},
                },
            }
        result = {"ok": True, "total_cost": 0.0, "segments": []}
        if alternatives > 1:
            result["alternatives"] = [{
                "segments": [], "total_cost": 0.0, "cost_delta": 0.0,
            }]
        return result
    for i, w in enumerate(words):
        if w["end"] < w["start"]:
            raise ValueError(f"词 {i} 的结束时间早于开始时间")
        if i and w["start"] < words[i - 1]["start"] - 1e-9:
            raise ValueError(f"词 {i} 的开始时间早于前一个词")

    units = build_units(words, phrases)
    n = len(units)
    shots = sorted(shots)
    mb, nb = validate_boundaries(boundaries or [], len(words), phrases, units)

    # 锁定段: 固定词范围与内部换行。校验失败(重叠/越界/与短语或新硬约束冲突)
    # 时直接返回首个冲突, 供页面高亮并保留锁定状态供修正。
    lock_info, lock_conflict = validate_locks(
        locks or [], words, units, cfg, mb, nb)
    if lock_conflict is not None:
        return lock_conflict
    if lock_info is not None:
        # 锁定边界转 must_break(段边界), 锁内间隙转 no_break(不得分段);
        # 锁之间不重叠且首尾相接时相接边界同属两个锁, 集合天然去重。
        mb = mb | lock_info["mb"]
        for a, b in lock_info["edges"].items():
            for u in range(a + 1, b):
                nb.add(u)

    # seg_cost[a][b]: 段 units[a:b+1] 的可行性与代价分解
    feasible = [[False] * n for _ in range(n)]
    cost_parts = [[None] * n for _ in range(n)]
    for a in range(n):
        for b in range(a, n):
            if units[b].end - units[a].start > cfg.max_duration + 1e-9:
                break  # 再延长只会更超时长
            if segment_violations(units, a, b, cfg):
                continue
            lay = layout_segment(units, a, b + 1, cfg.max_chars_per_line)
            lines, imb, split_unit = lay
            start, end = units[a].start, units[b].end
            dur = end - start
            chars = sum(len(u.text) for u in units[a:b + 1]) + (b - a)
            cross = crossing_count(start, end, shots)
            feasible[a][b] = True
            cost_parts[a][b] = {
                "lines": lines,
                "imbalance": imb,
                "split_unit": split_unit,
                "crossing": cross,
                "duration": dur,
                "cps": chars / dur if dur > 1e-9 else 0.0,
            }

    if lock_info is not None:
        # 用锁定换行覆盖锁定边的布局与行长不均代价(可行性已在 validate_locks 保证)。
        # 跨越多个自由区间与锁定段的 K-best 组合仍在同一张分段 DAG 上求解,
        # 按全局总代价统一排序, 并按完整断点序列去重(见下)。
        for (a, b), lay in lock_info["layout"].items():
            feasible[a][b - 1] = True
            cost_parts[a][b - 1]["lines"] = lay["lines"]
            cost_parts[a][b - 1]["imbalance"] = lay["imbalance"]
            cost_parts[a][b - 1]["split_unit"] = lay["split_unit"]

    states = _run_dp_kbest(units, feasible, cost_parts, cfg, shots, mb, nb,
                           k=alternatives)

    if not states[n]:
        return _conflict(units, feasible, cost_parts, cfg, shots, mb, nb,
                         lock_info=lock_info)

    # K-best 回溯: 每条路径即一个互不重复的分段边界序列
    plans = []
    seen_bounds: set[tuple[int, ...]] = set()
    for rank in range(len(states[n])):
        path = _backtrack(states, rank)
        key = tuple(path)
        if key in seen_bounds:  # 按分段边界序列判重(理论上不会出现, 兜底)
            continue
        seen_bounds.add(key)
        segments, raw_total = _build_segments(units, path, cost_parts, cfg, shots,
                                             lock_info=lock_info)
        plans.append({"path": path, "segments": segments, "raw_total": raw_total})

    best_raw = plans[0]["raw_total"]
    result = {
        "ok": True,
        "total_cost": round(best_raw, 4),
        "segments": plans[0]["segments"],
    }
    if alternatives > 1:
        result["alternatives"] = [{
            "segments": pl["segments"],
            "total_cost": round(pl["raw_total"], 4),
            "cost_delta": round(pl["raw_total"] - best_raw, 4),
        } for pl in plans]
    return result


def _lock_echo(lock_info: dict, units: list[Unit], u: int):
    """由锁定边界/锁内单元下标 u 反查锁定段回显信息 {word_range, line_break}。"""
    layout = lock_info.get("layout", {})
    for a, b in lock_info["edges"].items():
        if u == a or u == b or a < u < b:
            lay = layout.get((a, b), {})
            su = lay.get("split_unit")
            return {"word_range": [units[a].word_indices[0],
                                   units[b - 1].word_indices[-1]],
                    "line_break": units[su].word_indices[0] if su is not None else None}
    return None


def _conflict(units: list[Unit], feasible: list[list[bool]], cost_parts: list[list[dict]],
              cfg: Config, shots: list[float], mb: set[int], nb: set[int],
              lock_info: dict | None = None) -> dict:
    """无可行方案时定位冲突: 若由边界约束所致, 返回首个不可行边界及具体原因。

    有锁定段时, 锁边界/锁内间隙派生出的 must_break / no_break 归属于对应锁;
    首个冲突边界若属于锁, 冲突带 kind="lock" 与该锁回显, 供页面高亮并保留锁定状态。
    """
    n = len(units)
    INF = float("inf")
    dp0, _ = _run_dp(units, feasible, cost_parts, cfg, shots, set(), set())
    if dp0[n] == INF:
        # 去掉全部边界约束仍不可行: 与约束无关, 走通用冲突定位
        return _plain_conflict(units, dp0, cfg)

    # 锁定派生约束与用户约束的归属表
    lock_mb = lock_info["mb"] if lock_info else set()
    owner_nb: dict[int, tuple[int, int]] = {}
    if lock_info:
        for a, b in lock_info["edges"].items():
            for u in range(a + 1, b):
                owner_nb[u] = (a, b)

    # 边界约束导致不可行: 按单元边界升序逐个加入, 找首个使问题不可行的约束
    ordered = sorted([(u, "must_break") for u in mb] + [(u, "no_break") for u in nb])
    cur_mb: set[int] = set()
    cur_nb: set[int] = set()
    culprit = None
    for u, t in ordered:
        (cur_mb if t == "must_break" else cur_nb).add(u)
        dp, _ = _run_dp(units, feasible, cost_parts, cfg, shots, cur_mb, cur_nb)
        if dp[n] == INF:
            culprit = (u, t)
            break
    if culprit is None:  # 完整集合已知不可行, 理论上不会到达
        return _plain_conflict(units, dp0, cfg)

    u, t = culprit
    k = units[u].word_indices[0]  # 词间隙下标
    owned_lock = None
    if lock_info and (t == "must_break" and u in lock_mb
                      or t == "no_break" and u in owner_nb):
        owned_lock = _lock_echo(lock_info, units, u)
    if t == "must_break":
        s, e, a, reasons = _must_break_range(units, cfg, cur_mb, cur_nb)
        side = "右侧" if s == u else "左侧"
        ws, we = units[s].word_indices[0], units[e - 1].word_indices[-1]
        head = f"词 {k - 1} 与词 {k} 之间必须断开后, {side}区间(词 {ws}–{we})无可行分段"
        if owned_lock:
            head = (f"锁定段(词 {owned_lock['word_range'][0]}–"
                    f"{owned_lock['word_range'][1]})固定了其边界(词 {k - 1} 与词 {k} "
                    f"之间必须断开), 但{side}区间(词 {ws}–{we})在当前参数下无可行分段; "
                    f"请调整权重/约束或解锁该段")
    else:
        a, reasons = _no_break_reason(units, cfg, u, cur_mb, cur_nb)
        head = f"词 {k - 1} 与词 {k} 之间禁止断开, 两侧内容必须落在同一段"
        if owned_lock:
            head = (f"锁定段(词 {owned_lock['word_range'][0]}–"
                    f"{owned_lock['word_range'][1]})要求词 {k - 1} 与词 {k} 落在同一段, "
                    f"但与其他约束组合后无可行分段; 请调整权重/约束或解锁该段")
    wi = units[a].word_indices
    conflict = {
        "word_range": [wi[0], wi[-1]],
        "start": round(units[a].start, 3),
        "end": round(units[a].end, 3),
        "text": units[a].text,
        "reasons": [head] + reasons,
        "boundary": {"index": k, "type": t},
    }
    if owned_lock:
        conflict["kind"] = "lock"
        conflict["lock"] = owned_lock
    return {"ok": False, "conflict": conflict}


def _range_conflict(units: list[Unit], cfg: Config, s: int, e: int, nb: set[int]):
    """区间 [s, e) 必须独立分段(内部遵守禁止断开 nb)时,
    返回 (首个不可覆盖单元下标, 违例列表); 区间可行则返回 None。"""
    INF = float("inf")
    m = e - s
    dp = [INF] * (m + 1)
    dp[0] = 0.0
    for j in range(1, m + 1):
        for i in range(j):
            if i and (s + i) in nb:
                continue
            if dp[i] == INF or segment_violations(units, s + i, s + j - 1, cfg):
                continue
            dp[j] = 0.0
            break
    if dp[m] != INF:
        return None
    # 首个不可覆盖单元 = 最后一个可达切点之后的单元
    # (禁止断开可能使中间切点暂时不可达、随后又恢复, 不能取首个不可达位置)
    r = max(j for j in range(m + 1) if dp[j] != INF)
    a = s + r
    reasons = segment_violations(units, a, a, cfg)
    if not reasons:
        # 单元单独可行, 但与前方可达边界拼接均失败: 汇总最近一段的违例
        for i in range(a - 1, s - 1, -1):
            if dp[i - s] != INF:
                reasons = segment_violations(units, i, a, cfg)
                break
        if not reasons:
            reasons = ["与相邻单元组合后无可行布局"]
    return a, reasons


def _must_break_range(units: list[Unit], cfg: Config, mb: set[int], nb: set[int]):
    """must_break 边界把序列切成必须各自独立分段的区间, 返回首个不可行区间
    (s, e, 冲突单元下标, 违例列表)。"""
    n = len(units)
    cuts = [0] + sorted(mb) + [n]
    for s, e in zip(cuts, cuts[1:]):
        r = _range_conflict(units, cfg, s, e, nb)
        if r:
            a, reasons = r
            return s, e, a, reasons
    return 0, n, 0, ["无可行分段"]  # 完整集合已知不可行, 理论上不会到达


def _no_break_reason(units: list[Unit], cfg: Config, u: int, mb: set[int], nb: set[int]):
    """禁止断开边界 u: 沿连续 no_break 链扩展出必须同段的单元块, 返回冲突单元与违例。"""
    lo, hi = u, u
    while lo - 1 in nb:
        lo -= 1
    while hi + 1 in nb:
        hi += 1
    a = lo - 1  # 单元 lo-1 .. hi 必须落在同一段
    reasons = segment_violations(units, a, hi, cfg)
    if reasons:
        return a, reasons
    # 块本身可行, 是与必须断开等约束组合后不可行: 回退到所在强制区间定位
    n = len(units)
    cuts = [0] + sorted(mb) + [n]
    for s, e in zip(cuts, cuts[1:]):
        if s <= u <= e:
            r = _range_conflict(units, cfg, s, e, nb)
            if r:
                return r
    return a, ["与相邻边界约束组合后无可行分段"]


def _plain_conflict(units: list[Unit], dp: list[float], cfg: Config) -> dict:
    """与边界约束无关的不可行: 定位首个无法被任何可行分段覆盖的单元区间。"""
    INF = float("inf")
    n = len(units)
    # 首个不可覆盖单元 = 最后一个可达切点之后的单元(中间不可达位置可能随后恢复)
    a = max(j for j in range(n + 1) if dp[j] != INF)
    reasons = segment_violations(units, a, a, cfg)
    if not reasons:
        # 单元单独可行, 但与前方可达边界拼接均失败: 汇总最近一段的违例
        for i in range(a - 1, -1, -1):
            if dp[i] != INF:
                reasons = segment_violations(units, i, a, cfg)
                break
        if not reasons:
            reasons = ["与相邻单元组合后无可行布局"]
    wi = units[a].word_indices
    return {
        "ok": False,
        "conflict": {
            "word_range": [wi[0], wi[-1]],
            "start": round(units[a].start, 3),
            "end": round(units[a].end, 3),
            "text": units[a].text,
            "reasons": reasons,
        },
    }
