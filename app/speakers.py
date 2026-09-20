"""说话人标注: 对当前断句方案按说话人区间做全局动态规划标注。

输入为「当前选中的断句方案」与按行录入的说话人区间(开始 结束 说话人)。
服务端把每段与各说话人的区间求交, 得到重叠秒数, 再在段序列上用全局动态
规划为每段选择一个说话人, 最小化:

    未被所选说话人覆盖的时长 + 相邻段换人惩罚

  - 未被覆盖时长: 段时长 − 该段与所选说话人区间的重叠时长(时间按微秒量化
    为整数, 同分判定精确、无浮点误差);
  - 换人惩罚: 相邻两段均被标注且说话人不同, 每次 +switch_penalty 秒。

同分裁决: 依次按说话人首次出现顺序(区间录入顺序)、完整标签序列决胜 ——
从第一段起逐个比较, 首个不同位置取首次出现更早的说话人。实现上先做整数
后缀 DP 求出最小代价, 再从第一段起按说话人优先序贪心重建: 每段取仍能
达成全局最小代价的最优先说话人, 即得完整标签序列的唯一最优解。

特殊规则:
  - 与任何说话人区间都无重叠的段标为 unknown, 不参与换人惩罚(两侧相邻边
    均豁免), 其时长不计入优化目标, 单独汇总为 unknown_duration;
  - 人工覆盖(overrides)可强制某段为指定说话人或 unknown(speaker 为 null),
    强制 unknown 同样豁免换人惩罚; 覆盖引入的新说话人只出现在被覆盖的段上,
    不作为其他段的候选;
  - 区间起止非法(结束不晚于开始、开始为负、时间非有限数值、说话人为空)或
    同一说话人自身区间重叠时, 返回首个问题(结构化错误, 页面保留输入)。
"""

from __future__ import annotations

import math

QUANT = 1_000_000  # 时间量化: 微秒(整数运算, 同分判定精确)


class SpeakerError(Exception):
    """说话人区间/人工覆盖输入错误, payload 供接口结构化返回(首个问题)。"""

    def __init__(self, payload: dict):
        super().__init__(payload.get("type", "speaker_error"))
        self.payload = payload


def _q(t: float) -> int:
    return int(round(t * QUANT))


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


def _validate_intervals(intervals: list[dict]):
    """校验说话人区间, 返回 (说话人列表(首次出现顺序), {说话人: [(start_q, end_q)]})。

    依次返回首个问题: 起止非法(按录入顺序), 或同一说话人自身区间重叠
    (按说话人首次出现顺序、同一说话人内按区间开始时间排序, 首尾相接不算重叠)。
    """
    speakers: list[str] = []
    by_spk: dict[str, list[tuple[int, int, int, float, float]]] = {}
    for i, it in enumerate(intervals):
        st, en = it["start"], it["end"]
        sp = str(it["speaker"]).strip()
        bad = None
        if not (isinstance(st, (int, float)) and isinstance(en, (int, float))
                and not isinstance(st, bool) and not isinstance(en, bool)
                and math.isfinite(st) and math.isfinite(en)):
            bad = "时间不是有限数值"
        elif st < 0:
            bad = f"开始时间 {st} 为负"
        elif en <= st:
            bad = f"结束时间 {en} 不晚于开始时间 {st}"
        elif not sp:
            bad = "说话人为空"
        if bad:
            raise SpeakerError({
                "type": "invalid_interval",
                "index": i,
                "interval": {"start": st, "end": en, "speaker": it["speaker"]},
                "message": f"说话人区间第 {i + 1} 行起止非法:{bad};输入已保留",
            })
        if sp not in by_spk:
            speakers.append(sp)
            by_spk[sp] = []
        by_spk[sp].append((_q(st), _q(en), i, st, en))
    for sp in speakers:
        arr = sorted(by_spk[sp])  # (开始, 结束, 录入下标, 原始开始, 原始结束)
        for a, b in zip(arr, arr[1:]):
            if b[0] < a[1]:  # 正长度重叠(首尾相接不算)
                raise SpeakerError({
                    "type": "speaker_overlap",
                    "speaker": sp,
                    "first": {"index": a[2], "start": a[3], "end": a[4]},
                    "second": {"index": b[2], "start": b[3], "end": b[4]},
                    "message": (f"说话人「{sp}」的区间自相重叠:第 {a[2] + 1} 行"
                                f"({a[3]:g}–{a[4]:g} 秒)与第 {b[2] + 1} 行"
                                f"({b[3]:g}–{b[4]:g} 秒);输入已保留"),
                })
    return speakers, {sp: [(s, e) for s, e, *_ in by_spk[sp]] for sp in speakers}


def _parse_overrides(overrides: list[dict], n: int) -> dict[int, str | None]:
    """校验人工覆盖, 返回 {段下标: 说话人名 或 None(强制 unknown)}(返回首个问题)。"""
    forced: dict[int, str | None] = {}
    for o in overrides:
        idx = o.get("index")
        sp = o.get("speaker")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0 or idx >= n:
            raise SpeakerError({
                "type": "invalid_override", "index": idx,
                "message": f"人工覆盖的段下标 {idx} 越界(合法范围 0–{max(n - 1, 0)});输入已保留",
            })
        if idx in forced:
            raise SpeakerError({
                "type": "duplicate_override", "index": idx,
                "message": f"第 {idx + 1} 段被重复人工覆盖;输入已保留",
            })
        if sp is not None:
            sp = str(sp).strip()
            if not sp:
                raise SpeakerError({
                    "type": "invalid_override", "index": idx,
                    "message": f"第 {idx + 1} 段的人工覆盖说话人为空;输入已保留",
                })
        forced[idx] = sp
    return forced


def assign_speakers(words: list[dict], segments: list[dict], intervals: list[dict],
                    switch_penalty: float = 0.5, review_threshold: float = 0.6,
                    overrides: list[dict] | None = None) -> dict:
    """对给定断句方案与说话人区间执行说话人标注, 见模块文档字符串。"""
    overrides = overrides or []
    if not math.isfinite(switch_penalty) or switch_penalty < 0:
        raise ValueError(f"换人惩罚必须是不小于 0 的有限数值(当前 {switch_penalty})")
    if not math.isfinite(review_threshold) or not 0 <= review_threshold <= 1:
        raise ValueError(f"覆盖率阈值必须在 0–1 之间(当前 {review_threshold})")

    # 词时间校验(与断句接口同一口径)
    for i, w in enumerate(words):
        if w["end"] < w["start"]:
            raise ValueError(f"词 {i} 的结束时间早于开始时间")
        if i and w["start"] < words[i - 1]["start"] - 1e-9:
            raise ValueError(f"词 {i} 的开始时间早于前一个词")

    # 区间校验与断句方案无关: 即使当前无断句方案也先报告区间问题
    speakers, spk_intervals = _validate_intervals(intervals)
    spk_index = {sp: k for k, sp in enumerate(speakers)}

    base = {
        "ok": True,
        "speakers": speakers,
        "switch_penalty": switch_penalty,
        "review_threshold": review_threshold,
    }
    if not segments:
        # 无断句方案(词序列为空或断句冲突): 仅校验区间, 返回空标注结果
        return {**base, "n_segments": 0, "n_unknown": 0, "switches": 0,
                "uncovered": 0.0, "unknown_duration": 0.0, "total_cost": 0.0,
                "segments": [], "review": []}

    parsed = _validate_segments(words, segments)
    n = len(parsed)
    forced = _parse_overrides(overrides, n)

    # 覆盖引入的新说话人追加在区间说话人之后, 只作为被覆盖段的强制标签
    all_speakers = list(speakers)
    for sp in forced.values():
        if sp is not None and sp not in spk_index:
            spk_index[sp] = len(all_speakers)
            all_speakers.append(sp)
    m = len(speakers)  # 区间说话人数(自由候选只取前 m 个)

    # 每段时长(微秒)与每段 × 各说话人的重叠时长(微秒)
    dur_q: list[int] = []
    ov_q: list[list[int]] = []
    for wa, wb in parsed:
        ss, ee = _q(words[wa]["start"]), _q(words[wb]["end"])
        dur_q.append(max(0, ee - ss))
        row = []
        for k in range(m):
            tot = 0
            for s, e in spk_intervals[all_speakers[k]]:
                lo = ss if ss > s else s
                hi = ee if ee < e else e
                if hi > lo:
                    tot += hi - lo
            row.append(tot)
        ov_q.append(row)

    pen_q = _q(switch_penalty)

    # 每段候选说话人: 被覆盖段取强制标签; 无任何重叠的段强制 unknown;
    # 其余段取全部区间说话人(按首次出现顺序, 即同分裁决的优先序)
    is_unk = [False] * n
    cands: list[list[int] | None] = [None] * n
    for s in range(n):
        if s in forced:
            sp = forced[s]
            if sp is None:
                is_unk[s] = True
            else:
                cands[s] = [spk_index[sp]]
        elif m == 0 or all(v == 0 for v in ov_q[s]):
            is_unk[s] = True
        else:
            cands[s] = list(range(m))

    def cost(s: int, k: int) -> int:
        """段 s 标注为说话人 k 的未覆盖时长(微秒)。"""
        return dur_q[s] - (ov_q[s][k] if k < m else 0)

    # 后缀 DP: suf[s][k] = 段 s 标为 k 时, s..n-1 的最小代价(含 s 本身与其后换人);
    # usuf[s] = 段 s 为 unknown 时, s..n-1 的最小代价(unknown 不计代价与换人)
    suf: list[dict[int, int] | None] = [None] * n
    usuf = [0] * n

    def after(p: int, prev: int | None) -> int:
        """段 p 起的最小代价, prev 为段 p-1 的标签(None = unknown 或开头)。"""
        if p >= n:
            return 0
        if is_unk[p]:
            return usuf[p]
        best = None
        for k, v in suf[p].items():
            c = v + (pen_q if (prev is not None and k != prev) else 0)
            if best is None or c < best:
                best = c
        return best

    for s in range(n - 1, -1, -1):
        if is_unk[s]:
            usuf[s] = after(s + 1, None)
        else:
            suf[s] = {k: cost(s, k) + after(s + 1, k) for k in cands[s]}
    total_q = after(0, None)

    # 同分重建: 从第一段起按说话人优先序取仍能达成全局最小代价的标签,
    # 即完整标签序列中首个不同位置取首次出现更早说话人的唯一解
    labels: list[int | None] = [None] * n
    budget = total_q
    prev = None
    for s in range(n):
        if is_unk[s]:
            prev = None
            continue
        chosen = None
        for k in cands[s]:
            edge = pen_q if (prev is not None and k != prev) else 0
            if edge + suf[s][k] == budget:
                chosen = k
                budget = suf[s][k] - cost(s, k)
                break
        if chosen is None:  # 整数 DP 不变量保证必有候选, 兜底取代价最小者
            chosen = min(cands[s], key=lambda k: suf[s][k])
            budget = suf[s][chosen] - cost(s, chosen)
        labels[s] = chosen
        prev = chosen

    seg_out: list[dict] = []
    uncovered_q = 0
    unknown_q = 0
    switches = 0
    review: list[int] = []
    for s in range(n):
        k = labels[s]
        wa, wb = parsed[s]
        dur = dur_q[s]
        unk = k is None
        ov = 0 if unk else (ov_q[s][k] if k < m else 0)
        if unk:
            unknown_q += dur
        else:
            uncovered_q += dur - ov
        cov = ov / dur if dur > 0 else 0.0
        rv = cov < review_threshold
        if rv:
            review.append(s)
        if s and labels[s - 1] is not None and k is not None and labels[s - 1] != k:
            switches += 1
        seg_out.append({
            "index": s,
            "word_range": [wa, wb],
            "start": round(words[wa]["start"], 3),
            "end": round(words[wb]["end"], 3),
            "duration": round(dur / QUANT, 3),
            "speaker": "unknown" if unk else all_speakers[k],
            "unknown": unk,
            "override": s in forced,
            "overlap": round(ov / QUANT, 3),
            "coverage": round(cov, 4),
            "overlaps": {all_speakers[k2]: round(ov_q[s][k2] / QUANT, 3)
                         for k2 in range(m)},
            "review": rv,
        })

    return {**base,
            "n_segments": n,
            "n_unknown": sum(1 for k in labels if k is None),
            "switches": switches,
            "uncovered": round(uncovered_q / QUANT, 4),
            "unknown_duration": round(unknown_q / QUANT, 4),
            "total_cost": round(total_q / QUANT, 4),
            "segments": seg_out,
            "review": review}
