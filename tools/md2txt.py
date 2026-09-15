#!/usr/bin/env python3
# Markdown -> phone-readable plain text (CJK), tables flattened.
import re, sys, unicodedata

BT = chr(96)

def strip_inline(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(BT + r"([^" + BT + r"]+)" + BT, r"\1", s)
    s = s.replace("**", "")
    return s.strip()

def disp(s):
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)

def render_table(rows):
    rows = [[strip_inline(c) for c in r] for r in rows]
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    hdr, body = rows[0], rows[1:]
    out = []
    if ncol == 2:
        for r in body:
            out.append("\u30fb" + r[0] + "\uff1a" + r[1])
    else:
        sep = "  \uff5c  "
        out.append("\u3010" + sep.join(hdr) + "\u3011")
        for r in body:
            out.append("  " + sep.join(r))
    return out

def main(src, dst):
    lines = open(src, encoding="utf-8").read().split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        s = lines[i].rstrip()
        st = s.strip()
        if st == "":
            if out and out[-1] != "":
                out.append("")
            i += 1; continue
        if re.match(r"^-{3,}$", st):
            out.append(""); i += 1; continue
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            lvl, txt = len(m.group(1)), strip_inline(m.group(2))
            out.append("")
            if lvl == 1:
                out.append(txt)
                out.append("=" * min(disp(txt), 60))
            elif lvl == 2:
                out.append("\u3010" + txt + "\u3011")
                out.append("-" * min(disp(txt) + 2, 60))
            elif lvl == 3:
                out.append("\u25b8 " + txt)
            else:
                out.append("  \u00b7 " + txt)
            out.append("")
            i += 1; continue
        if s.lstrip().startswith("|"):
            rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            if len(rows) > 1 and all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) or c.strip() == "" for c in rows[1]):
                rows = [rows[0]] + rows[2:]
            out.extend(render_table(rows))
            out.append("")
            continue
        if s.startswith("    "):
            while i < n and lines[i].startswith("    "):
                out.append("    " + strip_inline(lines[i][4:]))
                i += 1
            out.append(""); continue
        if s.startswith(">"):
            while i < n and lines[i].startswith(">"):
                out.append("\u3000\u2502 " + strip_inline(lines[i].lstrip(">").strip()))
                i += 1
            out.append(""); continue
        if re.match(r"^\s*[-*]\s+", s):
            out.append("\u30fb" + strip_inline(re.sub(r"^\s*[-*]\s+", "", s)))
            i += 1; continue
        out.append(strip_inline(s))
        i += 1
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    open(dst, "w", encoding="utf-8").write(text)
    print("wrote", dst, len(text), "chars,", text.count("\n"), "lines")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
