"""术语一致性审校。

流程:
  1. 规范词与识别变体先做 NFKC + 大小写归一(casefold), 建立归一形 -> 术语归属;
     同一归一形被两个不同术语(规范词或变体)声明时归属冲突, 直接报错。
  2. 对当前断句方案的每个字幕段, 把段内词按空格拼接, 在归一化后的文本上用
     Aho-Corasick 一次性收集全部命中(允许嵌套、重叠、跨词, 但不跨字幕段),
     不按术语逐个匹配。
  3. 同一段内候选相互重叠/嵌套时, 用区间动态规划在非重叠候选集合中按
     (覆盖字符数, -非规范变体数, 录入顺序位权和) 取字典序最优且唯一解。
"""

from __future__ import annotations

import unicodedata
from collections import deque
from dataclasses import dataclass, field


class TermError(Exception):
    """术语表本身的错误(空规范词 / 变体归属冲突), payload 供接口结构化返回。"""

    def __init__(self, payload: dict):
        super().__init__(payload.get("type", "term_error"))
        self.payload = payload


def normalize(text: str) -> str:
    """NFKC + 大小写归一。逐字符归一(再拼接), 与带位置映射的归一化保持同一口径。"""
    return "".join(unicodedata.normalize("NFKC", ch).casefold() for ch in text)


def normalize_mapped(text: str) -> tuple[str, list[int], set[int]]:
    """归一化文本, 同时返回 归一字符下标 -> 原文字符下标 映射与"可作为边界的位置"。

    每个原字符的 NFKC+casefold 展开(可能为多个字符, 如 ﬁ -> fi)作为一个原子组:
    组首位置记入 group_starts。命中间隔只允许落在组边界上, 避免在一个原字符的
    展开内部切开(导致映射回原文时重叠)。
    """
    out: list[str] = []
    orig: list[int] = []
    group_starts: set[int] = set()
    for i, ch in enumerate(text):
        t = unicodedata.normalize("NFKC", ch).casefold()
        if not t:
            continue
        group_starts.add(len(out))
        for c in t:
            out.append(c)
            orig.append(i)
    return "".join(out), orig, group_starts


def compact_normalize(text: str) -> str:
    """术语模式归一: NFKC + 大小写归一后去除全部空白。

    段文本以词为单位拼接, 词间空白不参与匹配 —— 术语因此可以跨词(如词序列
    "人/工/智/能" 命中术语 "人工智能", "AI/model" 命中 "AI model"), 但匹配仍在
    同一段文本内进行, 不跨字幕段。
    """
    return "".join(c for c in normalize(text) if not c.isspace())


@dataclass
class Term:
    index: int
    canonical: str                       # 用户录入的规范写法原文
    canon_norm: str
    variants: list[tuple[str, str]] = field(default_factory=list)  # (录入原文, 归一形)


def prepare_terms(raw_terms: list[dict]) -> tuple[list[Term], dict[str, tuple[int, bool, str | None]]]:
    """解析术语表, 返回 (术语列表, 归一形归属表)。

    归属表: norm_surface -> (术语下标, 是否规范词, 变体录入原文(规范词时为 None))。
    同一术语内变体归一形与规范词相同者按规范词处理; 跨术语的同归一形声明报归属冲突。
    """
    terms: list[Term] = []
    owner: dict[str, tuple[int, bool, str | None]] = {}

    def claim(norm_s: str, term_idx: int, is_canon: bool, variant_raw: str | None):
        old = owner.get(norm_s)
        if old is not None and old[0] != term_idx:
            idxs = sorted({old[0], term_idx})
            raise TermError({
                "type": "variant_conflict",
                "term_indices": idxs,
                "surface": norm_s,
            })
        if old is None:  # 同术语重复声明(含变体归一后等于规范词)以首次为准
            owner[norm_s] = (term_idx, is_canon, variant_raw)

    for idx, t in enumerate(raw_terms):
        canonical = t.get("canonical") or ""
        canon_norm = compact_normalize(canonical)
        if not canon_norm:
            raise TermError({"type": "empty_canonical", "term_index": idx})
        term = Term(index=idx, canonical=canonical, canon_norm=canon_norm)
        terms.append(term)
        claim(canon_norm, idx, True, None)
        seen: set[str] = {canon_norm}
        for v in t.get("variants") or []:
            v_norm = compact_normalize(v)
            if not v_norm or v_norm in seen:
                continue  # 空白变体与术语内重复变体忽略
            seen.add(v_norm)
            term.variants.append((v, v_norm))
            claim(v_norm, idx, False, v)
    return terms, owner


def build_automaton(patterns: list[str]):
    """Aho-Corasick: 一次扫描收集全部模式的全部命中(含嵌套/重叠)。"""
    nxt: list[dict[str, int]] = [{}]
    fail: list[int] = [0]
    out: list[list[str]] = [[]]
    for p in patterns:
        node = 0
        for ch in p:
            nxt_node = nxt[node].get(ch)
            if nxt_node is None:
                nxt[node][ch] = len(nxt)
                nxt.append({})
                fail.append(0)
                out.append([])
                nxt_node = len(nxt) - 1
            node = nxt_node
        if p not in out[node]:
            out[node].append(p)
    q: deque[int] = deque()
    for ch, u in nxt[0].items():
        fail[u] = 0
        q.append(u)
    while q:
        v = q.popleft()
        out[v].extend(out[fail[v]])  # BFS 按深度进行, fail 链输出已聚合
        for ch, u in nxt[v].items():
            f = fail[v]
            while f and ch not in nxt[f]:
                f = fail[f]
            f = nxt[f].get(ch, 0)
            fail[u] = 0 if f == u else f
            q.append(u)
    return nxt, fail, out


@dataclass
class Candidate:
    s: int           # 归一文本起始下标(闭)
    e: int           # 归一文本结束下标(开)
    term_index: int
    is_canon: bool
    surface: str
    cover: int       # 映射回原文后覆盖的字符数
    bit: int = 0     # 录入顺序稳定唯一的位权(由 select 填充)


def select_hits(cands: list[Candidate], norm_len: int) -> list[Candidate]:
    """区间动态规划: 在归一文本 [0, norm_len) 上选择互不重叠的候选集合。

    最大化 (覆盖字符数, -变体命中数, 位权和)。每个候选按 (术语录入顺序, 起点,
    终点) 编全局唯一名次, 名次 r 赋位权 2^(m-1-r); 互异二进位权之和唯一标识
    候选子集(二进制), 且早录入候选的位权严格大于其后所有候选位权之和, 因此完全
    同分时稳定偏向录入更早的术语, 结果唯一确定。
    """
    if not cands:
        return []
    order = sorted(range(len(cands)),
                   key=lambda i: (cands[i].term_index, cands[i].s, cands[i].e))
    m = len(cands)
    for rank, ci in enumerate(order):
        cands[ci].bit = 1 << (m - 1 - rank)

    by_end: dict[int, list[Candidate]] = {}
    for c in cands:
        by_end.setdefault(c.e, []).append(c)

    dp: list[tuple[int, int, int] | None] = [None] * (norm_len + 1)
    pick: list[Candidate | None] = [None] * (norm_len + 1)
    dp[0] = (0, 0, 0)
    for p in range(1, norm_len + 1):
        dp[p] = dp[p - 1]  # 跳过第 p-1 个归一字符(不被任何命中覆盖)
        for c in by_end.get(p, ()):  # c.s < c.e == p, dp[c.s] 在此刻已最终确定
            base = dp[c.s]
            key = (base[0] + c.cover,
                   base[1] - (0 if c.is_canon else 1),
                   base[2] + c.bit)
            if key > dp[p]:
                dp[p] = key
                pick[p] = c

    chosen: list[Candidate] = []
    p = norm_len
    while p > 0:
        c = pick[p]
        if c is None:
            p -= 1
        else:
            chosen.append(c)
            p = c.s
    chosen.sort(key=lambda c: c.s)
    return chosen


def _line_hits(line_text: str, seg_text: str, lo: int, hi: int,
               hits: list[dict]) -> list[dict]:
    """把全文区间命中裁剪到某一行的字符区间 [lo, hi), 转换为行内偏移与行内文本。

    跨词术语恰好跨过段内换行时, 会被拆成两行各一段(行偏移相互独立);
    全局命中(segments[].hits)仍保留完整的一条, 供下载 JSON 还原。
    """
    out = []
    for h in hits:
        s, e = max(h["start"], lo), min(h["end"], hi)
        if s < e:
            lh = dict(h)
            lh["start"], lh["end"] = s - lo, e - lo
            lh["text"] = line_text[s - lo:e - lo]
            out.append(lh)
    return out


def review_terms(words: list[dict], segments: list[dict],
                 raw_terms: list[dict]) -> dict:
    """对给定断句方案执行术语一致性审校, 见模块文档字符串。"""
    terms, owner = prepare_terms(raw_terms)
    n_words = len(words)

    # 校验断句方案: 段按词范围连续、不重叠、完整覆盖词序列
    parsed_seg: list[tuple[int, int, int | None]] = []
    for si, seg in enumerate(segments):
        rng = seg.get("word_range")
        wa, wb = int(rng[0]), int(rng[1])
        if wa < 0 or wb < 0 or wa >= n_words or wb >= n_words:
            raise ValueError(f"第 {si + 1} 段词范围 [{wa}, {wb}] 越界"
                             f"(合法范围 0–{n_words - 1})")
        if wa > wb:
            raise ValueError(f"第 {si + 1} 段词范围 [{wa}, {wb}] 非法:起点大于终点")
        lb = seg.get("line_break")
        if lb is not None:
            lb = int(lb)
            if lb <= wa or lb > wb:
                raise ValueError(f"第 {si + 1} 段的内部换行 {lb} 不在其词间隙范围"
                                 f"(词 {wa + 1}–{wb} 左侧间隙)")
        parsed_seg.append((wa, wb, lb))
    if n_words == 0:
        if parsed_seg:
            raise ValueError("词序列为空时不能携带字幕段")
    else:
        if not parsed_seg or parsed_seg[0][0] != 0:
            raise ValueError("断句方案必须从词 0 开始完整覆盖词序列")
        if parsed_seg[-1][1] != n_words - 1:
            raise ValueError("断句方案必须完整覆盖到最后一个词")
        for (a1, b1, _), (a2, b2, _) in zip(parsed_seg, parsed_seg[1:]):
            if a2 != b1 + 1:
                raise ValueError(f"断句方案的字幕段不连续:词 {b1} 与词 {a2} 之间存在空隙或重叠")

    patterns = [s for s, (_i, is_c, _v) in owner.items()]
    nxt, fail, out = build_automaton(patterns)

    total_canon = total_variant = 0
    by_term: dict[int, dict] = {}
    inconsistencies: list[dict] = []
    seg_results: list[dict] = []

    for si, (wa, wb, lb) in enumerate(parsed_seg):
        word_texts = [words[k]["text"] for k in range(wa, wb + 1)]
        text = " ".join(word_texts)
        ntext, orig, group_starts = normalize_mapped(text)
        # 去空白得到匹配文本: 术语可以跨词(词间空白不参与匹配), 但不跨字幕段。
        # kept[j] 为匹配文本第 j 个字符在 ntext 中的下标, 命中区间经它映射回原文。
        kept = [i for i, ch in enumerate(ntext) if not ch.isspace()]
        mtext = "".join(ntext[i] for i in kept)

        # 一次自动机扫描, 收集该段全部候选(嵌套、重叠均保留, 交由区间 DP 裁决)
        cands: list[Candidate] = []
        state = 0
        for j, ch in enumerate(mtext):
            while state and ch not in nxt[state]:
                state = fail[state]
            state = nxt[state].get(ch, 0)
            je = j + 1
            for surf in out[state]:
                js = je - len(surf)
                s = kept[js]
                e = kept[je - 1] + 1  # ntext 区间(含跨词时被忽略的词间空白)
                # 不允许在单个原字符的归一展开内部切开
                if s not in group_starts:
                    continue
                if e < len(ntext) and not ntext[e].isspace() and e not in group_starts:
                    continue
                term_idx, is_canon, _variant_raw = owner[surf]
                cands.append(Candidate(
                    s=s, e=e, term_index=term_idx, is_canon=is_canon,
                    surface=surf, cover=orig[e - 1] - orig[s] + 1))

        chosen = select_hits(cands, len(ntext))
        term_by_idx = {t.index: t for t in terms}

        hits: list[dict] = []
        for c in chosen:
            a, b = orig[c.s], orig[c.e - 1] + 1
            t = term_by_idx[c.term_index]
            variant_raw = None
            if not c.is_canon:
                variant_raw = owner[c.surface][2]
            h = {
                "start": a,
                "end": b,
                "term_index": c.term_index,
                "kind": "canonical" if c.is_canon else "variant",
                "text": text[a:b],
                "canonical": t.canonical,
                "variant": variant_raw,
            }
            hits.append(h)

            row = by_term.setdefault(c.term_index, {
                "term_index": c.term_index, "canonical": t.canonical,
                "hits": 0, "canonical_hits": 0, "variant_hits": 0,
                "variants": {},
            })
            row["hits"] += 1
            if c.is_canon:
                total_canon += 1
                row["canonical_hits"] += 1
            else:
                total_variant += 1
                row["variant_hits"] += 1
                row["variants"].setdefault(variant_raw, 0)
                row["variants"][variant_raw] += 1

        # 段内两行文本(与断句结果的 lines 一致): 行首偏移与段全文偏移对齐
        lines_out: list[dict] = []
        if lb is None:
            lines_out.append({"text": text,
                              "hits": _line_hits(text, text, 0, len(text), hits)})
        else:
            cut = lb - wa  # 第二行首个词在 word_texts 中的下标
            l1 = " ".join(word_texts[:cut])
            l2 = " ".join(word_texts[cut:])
            l1_lo, l1_hi = 0, len(l1)
            l2_lo, l2_hi = len(l1) + 1, len(l1) + 1 + len(l2)
            lines_out.append({"text": l1, "hits": _line_hits(l1, text, l1_lo, l1_hi, hits)})
            lines_out.append({"text": l2, "hits": _line_hits(l2, text, l2_lo, l2_hi, hits)})

        for li, line in enumerate(lines_out):
            for lh in line["hits"]:
                if lh["kind"] == "variant":
                    inconsistencies.append({
                        "term_index": lh["term_index"],
                        "canonical": lh["canonical"],
                        "variant": lh["variant"],
                        "text": lh["text"],
                        "segment_index": si,
                        "line": li + 1,
                        "start": lh["start"],
                        "end": lh["end"],
                    })

        seg_results.append({
            "index": si,
            "word_range": [wa, wb],
            "line_break": lb,
            "text": text,
            "hits": hits,
            "lines": lines_out,
        })

    by_term_out = []
    for idx in sorted(by_term):
        row = by_term[idx]
        row["variants"] = [{"variant": v, "count": n}
                           for v, n in sorted(row["variants"].items())]
        by_term_out.append(row)

    return {
        "ok": True,
        "segment_count": len(seg_results),
        "segments": seg_results,
        "summary": {
            "terms": len(terms),
            "segments": len(seg_results),
            "hits": total_canon + total_variant,
            "canonical_hits": total_canon,
            "variant_hits": total_variant,
            "by_term": by_term_out,
        },
        "inconsistencies": inconsistencies,
    }
