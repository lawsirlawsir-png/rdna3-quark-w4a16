#!/usr/bin/env python3
# Markdown -> PDF renderer tuned for CJK technical documents (fpdf2).
import sys, re, unicodedata
from fpdf import FPDF

BT = chr(96)
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"

def disp_len(s):
    n = 0
    for ch in s:
        n += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return n

INLINE = re.compile(r"\*\*(.+?)\*\*|" + BT + r"([^" + BT + r"]+)" + BT)

def parse_inline(s):
    out, pos = [], 0
    for m in INLINE.finditer(s):
        if m.start() > pos:
            out.append((s[pos:m.start()], False, False))
        if m.group(1) is not None:
            out.append((m.group(1), True, False))
        else:
            out.append((m.group(2), False, True))
        pos = m.end()
    if pos < len(s):
        out.append((s[pos:], False, False))
    return [r for r in out if r[0] != ""] or [("", False, False)]

class Doc(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("main", size=8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 5, "COMMUNITY-POST-20260915  -  gfx1100 / W7800 48G  -  p. %d" % self.page_no(), align="C")
        self.set_text_color(0, 0, 0)

def draw_line(pdf, runs, x, y, size):
    for text, bold, code in runs:
        pdf.set_font("main", size=size)
        pdf.set_text_color(120, 70, 20) if code else pdf.set_text_color(0, 0, 0)
        w = pdf.get_string_width(text)
        pdf.text(x, y, text)
        if bold:
            pdf.text(x + 0.28, y, text)
        x += w
    pdf.set_text_color(0, 0, 0)

def wrap_runs(pdf, runs, maxw, size):
    pdf.set_font("main", size=size)
    lines, cur, curw, buf, bstyle = [], [], 0.0, "", None
    for text, bold, code in runs:
        style = (bold, code)
        for ch in text:
            w = pdf.get_string_width(ch)
            if curw + w > maxw and (cur or buf):
                if buf:
                    cur.append((buf, bstyle[0], bstyle[1])); buf = ""
                lines.append(cur); cur = []; curw = 0.0
            if bstyle is not None and bstyle != style and buf:
                cur.append((buf, bstyle[0], bstyle[1])); buf = ""
            bstyle = style
            buf += ch; curw += w
    if buf:
        cur.append((buf, bstyle[0], bstyle[1]))
    lines.append(cur)
    return lines

def emit_lines(pdf, lines, size, lh, indent=0.0, colorbar=None):
    lm = pdf.l_margin + indent
    avail = pdf.w - pdf.r_margin - lm
    for ln in lines:
        if pdf.get_y() + lh > pdf.h - 18:
            pdf.add_page()
        y = pdf.get_y() + lh * 0.75
        if colorbar is not None:
            pdf.set_fill_color(*colorbar)
            pdf.rect(pdf.l_margin + 1.0, pdf.get_y() + 1.0, 1.1, lh - 1.0, style="F")
        draw_line(pdf, ln, lm, y, size)
        pdf.set_y(pdf.get_y() + lh)
    return avail

def render_table(pdf, rows, size=8.0):
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    units = []
    for c in range(ncol):
        m = 1
        for r in rows:
            m = max(m, disp_len(re.sub(r"\*\*|" + BT, "", r[c])))
        units.append(min(m, 46))
    avail = pdf.w - pdf.l_margin - pdf.r_margin
    tot = float(sum(units))
    widths = [max(13.0, avail * u / tot) for u in units]
    sc = avail / sum(widths)
    widths = [w * sc for w in widths]
    pad, lh = 1.4, size * 0.52
    for i, r in enumerate(rows):
        cells = [wrap_runs(pdf, parse_inline(r[c]), widths[c] - 2 * pad, size) for c in range(ncol)]
        h = max(len(cl) for cl in cells) * lh + 2 * pad
        if pdf.get_y() + h > pdf.h - 18:
            pdf.add_page()
        y0, x = pdf.get_y(), pdf.l_margin
        if i == 0:
            pdf.set_fill_color(236, 240, 246)
            pdf.rect(x, y0, avail, h, style="F")
        if i == 1 and len(rows) > 2 and all(set(re.sub(r"[^A-Za-z-]", "", c)) <= set("-") and c.strip() for c in rows[1]):
            pass
        pdf.set_draw_color(170, 178, 190)
        for c in range(ncol):
            pdf.rect(x, y0, widths[c], h)
            ty = y0 + pad + lh * 0.75
            for ln in cells[c]:
                draw_line(pdf, ln, x + pad, ty, size)
                ty += lh
            x += widths[c]
        pdf.set_y(y0 + h + 1.2)

def main(src, dst):
    text = open(src, encoding="utf-8").read()
    lines = text.split("\n")
    pdf = Doc(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(True, margin=18)
    pdf.add_font("main", "", FONT)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_title("gfx1100 Quark INT4 on GIGABYTE W7800 48G")
    pdf.set_author("lawsirlawsir-png")

    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        s = ln.rstrip()
        if s.strip() == "":
            pdf.set_y(pdf.get_y() + 1.6); i += 1; continue
        if re.match(r"^-{3,}$", s.strip()):
            if pdf.get_y() + 4 > pdf.h - 18: pdf.add_page()
            pdf.set_draw_color(150, 150, 150)
            pdf.line(pdf.l_margin, pdf.get_y() + 2, pdf.w - pdf.r_margin, pdf.get_y() + 2)
            pdf.set_y(pdf.get_y() + 4.2); i += 1; continue
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            lvl, txt = len(m.group(1)), m.group(2)
            size = {1: 16.0, 2: 12.6, 3: 10.8, 4: 10.0}[lvl]
            before = {1: 0, 2: 4.4, 3: 3.0, 4: 2.2}[lvl]
            if pdf.get_y() + before + size * 0.8 > pdf.h - 18: pdf.add_page()
            pdf.set_y(pdf.get_y() + before)
            runs = parse_inline(txt)
            for r in runs: r = (r[0], True, r[2])
            wrapped = wrap_runs(pdf, [(t, True, c) for t, b, c in runs], pdf.w - 30, size)
            emit_lines(pdf, wrapped, size, size * 0.56)
            pdf.set_y(pdf.get_y() + 1.6)
            i += 1; continue
        if s.startswith("|"):
            rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                raw = lines[i].strip().strip("|")
                rows.append([c.strip() for c in raw.split("|")])
                i += 1
            body = [rows[0]] + [r for r in rows[2:]] if len(rows) > 1 else rows
            if len(rows) > 1 and all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) or c.strip() == "" for c in rows[1]):
                body = [rows[0]] + rows[2:]
            else:
                body = rows
            render_table(pdf, body)
            pdf.set_y(pdf.get_y() + 1.4); continue
        if s.startswith("    "):
            block = []
            while i < n and (lines[i].startswith("    ") or lines[i].strip() == ""):
                if lines[i].strip() == "" and (i + 1 >= n or not lines[i + 1].startswith("    ")):
                    break
                block.append(lines[i][4:]); i += 1
            size = 7.8
            wrapped = []
            for src_line in block:
                wrapped.extend(wrap_runs(pdf, [(src_line, False, True)], pdf.w - 34, size))
            for ln in wrapped:
                if pdf.get_y() + 4 > pdf.h - 18: pdf.add_page()
                pdf.set_fill_color(244, 245, 247)
                pdf.rect(pdf.l_margin + 1, pdf.get_y() + 0.4, pdf.w - 30 - 2, size * 0.5 + 0.6, style="F")
                draw_line(pdf, ln, pdf.l_margin + 3, pdf.get_y() + size * 0.45, size)
                pdf.set_y(pdf.get_y() + size * 0.5)
            pdf.set_y(pdf.get_y() + 2.0); continue
        if s.startswith(">"):
            block = []
            while i < n and lines[i].startswith(">"):
                block.append(lines[i].lstrip(">").strip()); i += 1
            runs = parse_inline(" ".join(block))
            wrapped = wrap_runs(pdf, runs, pdf.w - 38, 9.6)
            pdf.set_fill_color(250, 248, 240)
            y_before = pdf.get_y()
            emit_lines(pdf, wrapped, 9.6, 5.4, indent=6.0, colorbar=(212, 178, 90))
            pdf.rect(pdf.l_margin, y_before, pdf.w - 30, pdf.get_y() - y_before, style="F")
            emit_lines(pdf, wrapped, 9.6, 5.4, indent=6.0, colorbar=(212, 178, 90))
            pdf.set_y(pdf.get_y() + 1.8); continue
        if re.match(r"^\s*[-*]\s+", s):
            runs = parse_inline(re.sub(r"^\s*[-*]\s+", "", s))
            wrapped = wrap_runs(pdf, runs, pdf.w - 36, 9.8)
            draw_x = pdf.l_margin + 4.5
            if pdf.get_y() + 5 > pdf.h - 18: pdf.add_page()
            pdf.set_font("main", size=9.8); pdf.set_text_color(60, 60, 60)
            pdf.text(pdf.l_margin + 1.2, pdf.get_y() + 4.0, chr(0x2022))
            pdf.set_text_color(0, 0, 0)
            for k, ln2 in enumerate(wrapped):
                if pdf.get_y() + 5 > pdf.h - 18: pdf.add_page()
                draw_line(pdf, ln2, draw_x, pdf.get_y() + 4.0, 9.8)
                pdf.set_y(pdf.get_y() + 5.2)
            i += 1; continue
        runs = parse_inline(s)
        wrapped = wrap_runs(pdf, runs, pdf.w - 30, 9.8)
        emit_lines(pdf, wrapped, 9.8, 5.4)
        i += 1
    pdf.output(dst)
    print("wrote", dst)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
