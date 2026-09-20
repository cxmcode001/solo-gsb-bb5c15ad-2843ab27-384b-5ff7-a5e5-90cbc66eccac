"""说话人标注: 对当前断句方案按重叠时长做全局动态规划标注。

输入为按行录入的「开始 结束 说话人」区间(时间支持 HH:MM:SS.mmm 或秒数,
与词序列一致)。不同说话人的区间允许互相重叠; 同一说话人自身的区间不得
重叠(首尾相接允许), 否则返回首个问题区间。

每段与各说话人的重叠秒数 = 段时间范围与该说话人全部区间的交集长度之和。
全局动态规划最小化:
    Σ(段时长 − 所选说话人的重叠秒数) + 换人惩罚 × 相邻段换人次数
其中「换人」只统计相邻两段均为真实说话人且标签不同的情况: 与 unknown
相邻(或跨过 unknown)均不计换人惩罚。与所有说话人重叠秒数均为 0 的段
强制标为 unknown(不是 DP 的可选项); 其余段可选任一说话人(允许选择
重叠为 0 的说话人, 代价为整段时长, 由换人惩罚权衡是否值得)。

同分决胜(依次):
  1. 说话人首次出现顺序: 把标注结果看作标签序列, 从第一段起逐一比较,
     首个不同段中说话人在输入区间里首次出现更早者胜;
  2. 完整标签序列: 按完整标签文本序列的字典序决胜。
实现上每个标签取名次(说话人首次出现名次, unknown 排在所有说话人之后),
名次序列即标签序列的秩, 按 (总代价, 名次序列) 字典序比较即同时覆盖两级
规则(名次相同当且仅当标签文本相同)。所有时间量化到微秒整数再进入 DP,
同分判定精确、无浮点误差。

人工覆盖: overrides 把指定段的标签钉死(可为任一说话人或 unknown),
其余段仍在同一目标下重新全局优化; 被钉死的段照常计入未覆盖时长,
钉为真实说话人时照常参与相邻换人惩罚, 钉为 unknown 时不计。
"""

from __future__ import annotations

import re

UNKNOWN = "unknown"

# 时间量化单位: 微秒(整数运算, 同分判定精确)
_SCALE = 1_000_000

_NUM_RE = re.compile(r"\d+(\.\d+)?")
_TIME_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{1,2})[,.](\d{1,3})")


class SpeakerError(Exception):
    """说话人区间或人工覆盖本身的错误, payload 供接口结构化返回(HTTP 422)。"""

    def __init__(self, payload: dict):
        super().__init__(payload.get("type", "speaker_error"))
        self.payload = payload


def parse_time(s: str) -> float:
    """解析时间点: 秒数或 HH:MM:SS.mmm(也接受 MM:SS.mmm 与逗号毫秒分隔)。"""
    s = s.strip()
    if _NUM_RE.fullmatch(s):
        return float(s)
    m = _TIME_RE.fullmatch(s)
    if not m:
        raise ValueError(s)
    h, mm, ss, ms = m.groups()
    return (int(h or 0) * 3600 + int(mm) * 60 + int(ss)
            + int(ms.ljust(3, "0")) / 1000)


def parse_speakers(text: str) -> tuple[list[str], list[list[tuple[float, float]]]]:
    """解析按行录入的说话人区间, 返回 (说话人名单(首次出现顺序), 每人区间列表)。

    每行「开始 结束 说话人」, 空行忽略; 行号为输入中的实际行号(空行也计数)。
    首个问题(格式/起止非法、同一说话人区间自身重叠)即抛 SpeakerError。
    """
    names: list[str] = []
    index_of: dict[str, int] = {}
    intervals: list[list[tuple[float, float]]] = []
    for lineno, raw in enumerate((text or "").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[2].strip():
            raise SpeakerError({
                "type": "bad_line",
                "line": lineno,
                "message": f"第 {lineno} 行格式错误: 应为「开始 结束 说话人」三个字段;输入已保留",
            })
        try:
            start = parse_time(parts[0])
            end = parse_time(parts[1])
        except ValueError:
            raise SpeakerError({
                "type": "bad_line",
                "line": lineno,
                "message": f"第 {lineno} 行时间无法解析(支持秒数或 HH:MM:SS.mmm):{line};输入已保留",
            }) from None
        if not start < end:
            raise SpeakerError({
                "type": "bad_interval",
                "line": lineno,
                "start": start,
                "end": end,
                "message": f"第 {lineno} 行区间起止非法: 开始 {start} 不早于结束 {end};输入已保留",
            })
        name = parts[2].strip()
        si = index_of.get(name)
        if si is None:
            si = len(names)
            index_of[name] = si
            names.append(name)
            intervals.append([])
        intervals[si].append((start, end))

    # 同一说话人自身区间不得重叠(首尾相接允许); 按首次出现顺序逐个检查, 返回首个重叠对
    for si, name in enumerate(names):
        ivs = sorted(intervals[si])
        for (s1, e1), (s2, e2) in zip(ivs, ivs[1:]):
            if s2 < e1:
                raise SpeakerError({
                    "type": "speaker_overlap",
                    "speaker": name,
                    "intervals": [[round(s1, 3), round(e1, 3)],
                                  [round(s2, 3), round(e2, 3)]],
                    "message": f"说话人「{name}」的区间 [{round(s1, 3)}, {round(e1, 3)}] "
                               f"与 [{round(s2, 3)}, {round(e2, 3)}] 自身重叠,"
                               f"请修正为不重叠的区间;输入已保留",
                })
    return names, intervals


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


def _ticks(x: float) -> int:
    return int(round(x * _SCALE))


def assign_speakers(words: list[dict], segments: list[dict], speakers_text: str,
                    change_penalty: float = 1.0, coverage_threshold: float = 0.5,
                    overrides: list[dict] | None = None) -> dict:
    """对给定断句方案执行说话人标注, 见模块文档字符串。"""
    names, intervals = parse_speakers(speakers_text)
    n = len(segments)
    k = len(names)
    base = {"n_segments": n, "speakers": names,
            "change_penalty": change_penalty,
            "coverage_threshold": coverage_threshold}
    if n == 0:
        if k == 0:
            return {"ok": True, **base, "uncovered": 0.0, "changes": 0,
                    "total_cost": 0.0, "review_count": 0, "segments": []}
        return {"ok": False, **base,
                "error": {"type": "empty_segments",
                          "message": "源侧为空:当前没有可标注的断句方案;输入已保留"}}
    if k == 0:
        return {"ok": False, **base,
                "error": {"type": "empty_speakers",
                          "message": "说话人区间为空:请按行输入「开始 结束 说话人」;输入已保留"}}

    parsed = _validate_segments(words, segments)

    # 人工覆盖: 段下标 -> 标签(说话人或 unknown); 同一段重复覆盖以首个为准
    pinned: dict[int, str] = {}
    for ov in overrides or []:
        idx = int(ov.get("segment", -1))
        label = str(ov.get("label", ""))
        if idx < 0 or idx >= n:
            raise SpeakerError({
                "type": "bad_override",
                "segment": idx,
                "message": f"人工覆盖的段下标 {idx} 越界(合法范围 0–{n - 1}),"
                           f"断句方案已变化,请清除后重新覆盖;输入已保留",
            })
        if label != UNKNOWN and label not in names:
            raise SpeakerError({
                "type": "bad_override",
                "segment": idx,
                "label": label,
                "message": f"人工覆盖的标签「{label}」不在说话人区间中,"
                           f"请修正区间或清除该覆盖;输入已保留",
            })
        if idx not in pinned:
            pinned[idx] = label

    # 每段时间范围与每段×每说话人的重叠秒数(微秒整数)
    seg_range = [(_ticks(words[wa]["start"]), _ticks(words[wb]["end"]))
                 for wa, wb in parsed]
    iv_ticks = [[(_ticks(s), _ticks(e)) for s, e in ivs] for ivs in intervals]
    overlap: list[list[int]] = []
    for st, et in seg_range:
        row = []
        for si in range(k):
            tot = 0
            for a, b in iv_ticks[si]:
                if a < et and st < b:
                    tot += min(et, b) - max(st, a)
            row.append(tot)
        overlap.append(row)

    pen = _ticks(change_penalty)
    UNK = -1  # 内部标签: unknown

    # 全局 DP: dp[i][标签] = (整数总代价, 名次序列); 名次 = 说话人首次出现名次,
    # unknown 名次为 k(排在所有说话人之后)。名次序列即完整标签序列的秩,
    # 按 (总代价, 名次序列) 字典序比较即实现两级同分决胜。
    dp: list[dict[int, tuple[int, tuple[int, ...]]]] = []
    for i in range(n):
        st, et = seg_range[i]
        dur = max(et - st, 0)
        if i in pinned:
            lab = pinned[i]
            allowed = [UNK] if lab == UNKNOWN else [names.index(lab)]
        elif sum(overlap[i]) == 0:
            allowed = [UNK]  # 无任何重叠的段强制 unknown
        else:
            allowed = list(range(k))
        cur: dict[int, tuple[int, tuple[int, ...]]] = {}
        for s in allowed:
            c_i = dur if s == UNK else dur - overlap[i][s]
            rank = k if s == UNK else s
            best = None
            if i == 0:
                best = (c_i, (rank,))
            else:
                for p, (pc, pkey) in dp[-1].items():
                    sw = pen if (p != UNK and s != UNK and p != s) else 0
                    cand = (pc + sw + c_i, pkey + (rank,))
                    if best is None or cand < best:
                        best = cand
            cur[s] = best
        dp.append(cur)

    _, (cost_ticks, key) = min(dp[-1].items(), key=lambda kv: kv[1])
    labels = [UNK if r == k else r for r in key]

    changes = sum(1 for i in range(1, n)
                  if labels[i - 1] != UNK and labels[i] != UNK
                  and labels[i - 1] != labels[i])

    out_segments: list[dict] = []
    uncovered_ticks = 0
    review_count = 0
    for i in range(n):
        st, et = seg_range[i]
        dur = max(et - st, 0)
        lab = labels[i]
        ov_chosen = 0 if lab == UNK else overlap[i][lab]
        uncovered_ticks += dur - ov_chosen
        coverage = (ov_chosen / dur) if dur > 0 else 0.0
        needs_review = coverage < coverage_threshold
        if needs_review:
            review_count += 1
        out_segments.append({
            "index": i,
            "word_range": [parsed[i][0], parsed[i][1]],
            "start": round(st / _SCALE, 3),
            "end": round(et / _SCALE, 3),
            "duration": round(dur / _SCALE, 3),
            "label": UNKNOWN if lab == UNK else names[lab],
            "overridden": i in pinned,
            "overlap": round(ov_chosen / _SCALE, 4),
            "uncovered": round((dur - ov_chosen) / _SCALE, 4),
            "coverage": round(coverage, 4),
            "overlaps": {names[si]: round(overlap[i][si] / _SCALE, 4)
                         for si in range(k)},
            "needs_review": needs_review,
        })

    return {
        "ok": True,
        **base,
        "uncovered": round(uncovered_ticks / _SCALE, 4),
        "changes": changes,
        "total_cost": round(cost_ticks / _SCALE, 4),
        "review_count": review_count,
        "segments": out_segments,
    }
