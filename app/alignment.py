"""译文对齐: 对当前断句方案与按行粘贴的译文做单调动态规划对齐。

对齐在「当前选中的断句方案」的源段序列与译文非空行序列之间进行,
要求单调(两侧均按序消费、不交叉)且完整覆盖两侧。允许的组型:
  - 1:1  一个源段 对 一行译文
  - 1:2  一个源段 对 相邻两行译文
  - 2:1  相邻两个源段 对 一行译文

每组代价 = 两侧长度占比差绝对值 + 句末标点惩罚:
  占比 = 组内该侧去空白字符数 / 该侧全文去空白字符数
  惩罚 = 组末句末标点类别不一致时 +0.15(类别见 punct_category)
总代价最小的完整覆盖即所求; 同分时把路径看作组型序列, 从第一组起
依次比较, 首个不同组按 1:1 → 1:2 → 2:1 取优先 —— 即优先 1:1,
其次源侧结束位置更早的组(1:2 的源侧结束位置早于 2:1)。

实现: 每组代价改写为整数后在 (源段数 × 译文行数) 网格上做后缀 DP,
再按组型优先序从起点贪心重建最优路径:
  每组整数代价 = 20·|a·T − b·S| + (标点不一致 ? 3·S·T : 0)
  总代价       = 整数代价之和 / (20·S·T)
其中 a/b 为组内两侧去空白字符数, S/T 为两侧全文去空白字符数(为 0 时
按 1 计, 此时该侧各组占比均为 0, 相对比较不受影响)。0.15 = 3/20,
故整数代价之和与真实总代价一一对应, 同分判定精确、无浮点误差。
一侧为空或无法完整覆盖(一侧项数超过另一侧两倍)时不进入 DP,
另做前向可达性扫描, 定位首个无法被任何合法对齐组覆盖的位置。
"""

from __future__ import annotations

# 组型: (源段步长, 译文行步长); 同分重建时按此顺序取优先(见模块文档字符串)。
MOVES = ((1, 1), (1, 2), (2, 1))

_KIND = {(1, 1): "1:1", (1, 2): "1:2", (2, 1): "2:1"}

# 句末标点类别: 取文本最后一个非空白字符
_PUNCT: dict[str, str] = {}
for _ch in ".。｡．":
    _PUNCT[_ch] = "period"
for _ch in "?？":
    _PUNCT[_ch] = "question"
for _ch in "!！":
    _PUNCT[_ch] = "exclaim"
for _ch in "…‥":
    _PUNCT[_ch] = "ellipsis"


def split_translation(text: str) -> list[str]:
    """译文按行拆分: 各行去除首尾空白, 忽略空行。"""
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def non_ws(s: str) -> int:
    """去空白字符数(空白含空格/制表/换行及 Unicode 空白)。"""
    return sum(1 for ch in s if not ch.isspace())


def punct_category(s: str) -> str:
    """句末标点类别: 最后一个非空白字符所属类别, 无标点或其他符号为 other。"""
    for ch in reversed(s):
        if not ch.isspace():
            return _PUNCT.get(ch, "other")
    return "other"


def _validate_segments(words: list[dict], segments: list[dict]) -> list[tuple[int, int]]:
    """校验断句方案: 段按词范围连续、不重叠、完整覆盖词序列; 返回 (首词, 末词) 列表。"""
    n_words = len(words)
    parsed: list[tuple[int, int]] = []
    for si, seg in enumerate(segments):
        rng = seg.get("word_range")
        wa, wb = int(rng[0]), int(rng[1])
        if wa < 0 or wb < 0 or wa >= n_words or wb >= n_words:
            raise ValueError(f"第 {si + 1} 段词范围 [{wa}, {wb}] 越界"
                             f"(合法范围 0–{n_words - 1})")
        if wa > wb:
            raise ValueError(f"第 {si + 1} 段词范围 [{wa}, {wb}] 非法:起点大于终点")
        parsed.append((wa, wb))
    if not parsed or parsed[0][0] != 0:
        raise ValueError("断句方案必须从词 0 开始完整覆盖词序列")
    if parsed[-1][1] != n_words - 1:
        raise ValueError("断句方案必须完整覆盖到最后一个词")
    for (a1, b1), (a2, b2) in zip(parsed, parsed[1:]):
        if a2 != b1 + 1:
            raise ValueError(f"断句方案的字幕段不连续:词 {b1} 与词 {a2} 之间存在空隙或重叠")
    return parsed


def _solve(sa: list[int], pc_src: list[str], tb: list[int], pc_tgt: list[str]):
    """后缀 DP + 同分贪心重建。返回 (组型序列, 整数总代价, 分母 S·T); 无法完整覆盖返回 None。"""
    n, m = len(sa), len(tb)
    S, T = max(sum(sa), 1), max(sum(tb), 1)
    den = S * T
    pa = [0] * (n + 1)
    for i, v in enumerate(sa):
        pa[i + 1] = pa[i] + v
    pb = [0] * (m + 1)
    for j, v in enumerate(tb):
        pb[j + 1] = pb[j] + v

    def edge(i: int, j: int, di: int, dj: int) -> int:
        a = pa[i + di] - pa[i]
        b = pb[j + dj] - pb[j]
        c = 20 * abs(a * T - b * S)
        if pc_src[i + di - 1] != pc_tgt[j + dj - 1]:
            c += 3 * den
        return c

    # suf[i][j]: 从 (i, j) 覆盖到 (n, m) 的最小整数代价; None = 不可达
    suf: list[list[int | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    suf[n][m] = 0
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i == n and j == m:
                continue
            best = None
            for di, dj in MOVES:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                sub = suf[ni][nj]
                if sub is None:
                    continue
                c = edge(i, j, di, dj) + sub
                if best is None or c < best:
                    best = c
            suf[i][j] = best
    if suf[0][0] is None:
        return None
    # 同分按组型优先序贪心重建: 每步取仍能达成全局最优的最优先组型
    moves: list[tuple[int, int]] = []
    i = j = 0
    while i < n or j < m:
        for di, dj in MOVES:
            ni, nj = i + di, j + dj
            if ni > n or nj > m:
                continue
            sub = suf[ni][nj]
            if sub is not None and suf[i][j] == edge(i, j, di, dj) + sub:
                moves.append((di, dj))
                i, j = ni, nj
                break
    return moves, suf[0][0], den


def _first_uncovered(n: int, m: int) -> tuple[str, int]:
    """无法完整覆盖时, 定位首个无法被任何合法对齐组覆盖的位置, 返回 (侧, 0 起下标)。

    前向可达: 状态 (i, j) 表示已覆盖 i 个源段、j 行译文; 源段/译文行可被覆盖
    ⟺ 存在可达状态能发出覆盖它的合法移动。
    """
    reach = [[False] * (m + 1) for _ in range(n + 1)]
    reach[0][0] = True
    cover_s = [False] * n
    cover_t = [False] * m
    for i in range(n + 1):
        for j in range(m + 1):
            if not reach[i][j]:
                continue
            for di, dj in MOVES:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                reach[ni][nj] = True
                for k in range(i, ni):
                    cover_s[k] = True
                for k in range(j, nj):
                    cover_t[k] = True
    for i in range(n):
        if not cover_s[i]:
            return "source", i
    for j in range(m):
        if not cover_t[j]:
            return "translation", j
    return "source", n  # 完整集合已知不可覆盖, 理论上不会到达


def _error(etype: str, side: str, index: int, head: str) -> dict:
    pos = f"源段第 {index + 1} 段" if side == "source" else f"译文第 {index + 1} 行"
    return {
        "type": etype,
        "first_uncovered": {"side": side, "index": index},
        "message": f"{head},首个未覆盖位置为{pos};输入已保留",
    }


def align_translation(words: list[dict], segments: list[dict], translation: str) -> dict:
    """对给定断句方案与译文(按行原文)执行单调对齐, 见模块文档字符串。"""
    lines = split_translation(translation)
    n, m = len(segments), len(lines)
    base = {"n_segments": n, "n_lines": m}
    if n == 0:
        if m == 0:
            return {"ok": True, **base, "src_chars": 0, "tgt_chars": 0,
                    "total_cost": 0.0, "groups": []}
        return {"ok": False, **base,
                "error": _error("empty_source", "translation", 0,
                                "源侧为空:当前没有可对齐的断句方案")}
    if m == 0:
        return {"ok": False, **base,
                "error": _error("empty_translation", "source", 0,
                                "译文为空:请按行粘贴译文")}

    parsed = _validate_segments(words, segments)
    seg_texts = [" ".join(words[k]["text"] for k in range(wa, wb + 1))
                 for wa, wb in parsed]
    sa = [non_ws(t) for t in seg_texts]
    tb = [non_ws(t) for t in lines]
    pc_src = [punct_category(t) for t in seg_texts]
    pc_tgt = [punct_category(t) for t in lines]

    solved = _solve(sa, pc_src, tb, pc_tgt)
    if solved is None:
        side, idx = _first_uncovered(n, m)
        return {"ok": False, **base,
                "error": _error("no_cover", side, idx,
                                f"无法完整覆盖(源段 {n} 段 / 译文 {m} 行,"
                                f"每组至多为 1:2 或 2:1)")}

    moves, num, den = solved
    S, T = sum(sa), sum(tb)
    Sd, Td = max(S, 1), max(T, 1)
    pa = [0]
    for v in sa:
        pa.append(pa[-1] + v)
    pb = [0]
    for v in tb:
        pb.append(pb[-1] + v)
    groups = []
    i = j = 0
    for di, dj in moves:
        a = pa[i + di] - pa[i]
        b = pb[j + dj] - pb[j]
        pm = pc_src[i + di - 1] != pc_tgt[j + dj - 1]
        diff = abs(a * Td - b * Sd) / den
        groups.append({
            "index": len(groups),
            "kind": _KIND[(di, dj)],
            "src_segments": list(range(i, i + di)),
            "tgt_lines": list(range(j, j + dj)),
            "src_text": " ".join(seg_texts[i:i + di]),
            "tgt_text": " ".join(lines[j:j + dj]),
            "start": round(words[parsed[i][0]]["start"], 3),
            "end": round(words[parsed[i + di - 1][1]]["end"], 3),
            "src_chars": a,
            "tgt_chars": b,
            "ratio_diff": round(diff, 4),
            "punct_mismatch": pm,
            "cost": round(diff + (0.15 if pm else 0.0), 4),
        })
        i += di
        j += dj
    return {
        "ok": True,
        **base,
        "src_chars": S,
        "tgt_chars": T,
        "total_cost": round(num / (20 * den), 4),
        "groups": groups,
    }
