"""Build HSR_phase_attention_20261013.pptx from slides_source.md (python-pptx).

Run:  <venv with python-pptx, Pillow>/bin/python build_deck.py
Inputs are read-only: figures under AG3S/ag3s/docs/figures and paper/figures.
Cropped figures live in ./crops (made from the originals, originals untouched).
"""
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "HSR_phase_attention_20261013.pptx"

FONT = "Noto Sans CJK KR"
INK = RGBColor(0x1E, 0x2A, 0x3A)
MUTED = RGBColor(0x5B, 0x65, 0x73)
ACCENT = RGBColor(0xA8, 0x4A, 0x22)      # rust, matches the C3 panel tone
TINT = RGBColor(0xFB, 0xF0, 0xEA)
HEAD = RGBColor(0x2E, 0x3B, 0x4E)
ROW_A = RGBColor(0xFF, 0xFF, 0xFF)
ROW_B = RGBColor(0xF2, 0xF4, 0xF7)
LINE = RGBColor(0xC9, 0xCE, 0xD6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK_BG = RGBColor(0x1E, 0x2A, 0x3A)

FIG = REPO / "AG3S/ag3s/docs/figures"
CROPS = HERE / "crops"


def ensure_crop(src, box, dst):
    """Crop src (read-only original) to box=(l, t, r, b) px and save as crops/dst (once)."""
    out = CROPS / dst
    if not out.exists():
        Image.open(src).crop(box).save(out)
    return out


# T48b: top "held-out primary" panel only (legend row + panel; original 1610x1552)
T48B_HELDOUT = ensure_crop(FIG / "t48/t48b-score-accuracy-bars.png", (0, 30, 1610, 576),
                           "t48b-score-accuracy-bars-heldout-primary.png")
# T59 diag: three panels without the figure-wide suptitle (original 2550x840)
T59_DIAG = ensure_crop(FIG / "t59/t59-diag-attention-stage.png", (0, 56, 2550, 836),
                       "t59-diag-attention-stage-notitle.png")

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
TOTAL = 15
_count = [0]


# ---------------------------------------------------------------- helpers
def set_font(run, size, bold=False, color=INK, italic=False):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    f.name = FONT
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rpr, qn(tag))
        el.set("typeface", FONT)


def set_bullet(par, level, size):
    pPr = par._p.get_or_add_pPr()
    indent_in = 0.22 if level == 0 else 0.2
    marl = 0.22 + 0.3 * level
    pPr.set("marL", str(int(Inches(marl))))
    pPr.set("indent", str(int(-Inches(indent_in))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    buf = etree.SubElement(pPr, qn("a:buFont"))
    buf.set("typeface", "Arial")
    bu = etree.SubElement(pPr, qn("a:buChar"))
    bu.set("char", "•" if level == 0 else "–")


def no_bullet(par):
    pPr = par._p.get_or_add_pPr()
    etree.SubElement(pPr, qn("a:buNone"))


def textbox(slide, x, y, w, h, margin=0.04, anchor=MSO_ANCHOR.TOP, name=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    if name:
        tb.name = name
    return tb


def add_runs(par, runs, size, color=INK, bold=False):
    """runs: str or list of (text, dict(bold=..., color=...))."""
    if isinstance(runs, str):
        runs = [(runs, {})]
    for text, opt in runs:
        r = par.add_run()
        r.text = text
        set_font(r, opt.get("size", size), bold=opt.get("bold", bold),
                 color=opt.get("color", color), italic=opt.get("italic", False))


def bullets(slide, x, y, w, h, items, size=14, space_after=6, name="Body"):
    """items: list of (level, runs). level -1 = plain paragraph (no bullet)."""
    tb = textbox(slide, x, y, w, h, name=name)
    tf = tb.text_frame
    first = True
    for level, runs in items:
        par = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        sz = size if level <= 0 else size - 1
        add_runs(par, runs, sz, color=INK if level <= 0 else RGBColor(0x33, 0x3D, 0x4A))
        if level >= 0:
            set_bullet(par, level, sz)
        else:
            no_bullet(par)
        par.space_after = Pt(space_after)
        par.line_spacing = 1.1
    return tb


def image_fit(slide, path, x, y, w, h, align="center", valign="top", name=None):
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    px = x + {"left": 0, "center": (w - pw) / 2, "right": w - pw}[align]
    py = y + {"top": 0, "middle": (h - ph) / 2, "bottom": h - ph}[valign]
    pic = slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
    pic.line.color.rgb = LINE
    pic.line.width = Pt(0.75)
    if name:
        pic.name = name
    return pic, (px, py, pw, ph)


def header(slide, title, message):
    tb = textbox(slide, 0.5, 0.32, 12.33, 0.62, margin=0, anchor=MSO_ANCHOR.MIDDLE, name="Title")
    p = tb.text_frame.paragraphs[0]
    add_runs(p, title, 24, bold=True)
    if message:
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.05),
                                     Inches(12.33), Inches(0.72))
        box.name = "Message"
        box.adjustments[0] = 0.12
        box.fill.solid()
        box.fill.fore_color.rgb = TINT
        box.line.fill.background()
        box.shadow.inherit = False
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = Inches(0.18)
        tf.margin_top = tf.margin_bottom = Inches(0.03)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        add_runs(p, message, 15, color=ACCENT, bold=True)


def footer(slide, dark=False):
    _count[0] += 1
    n = _count[0]
    col = RGBColor(0xB8, 0xC0, 0xCC) if dark else RGBColor(0x8A, 0x93, 0xA0)
    tb = textbox(slide, 0.5, 7.08, 8.0, 0.3, margin=0, anchor=MSO_ANCHOR.MIDDLE, name="Footer")
    add_runs(tb.text_frame.paragraphs[0], "AG3S 진행 보고 · HSR · phase별 attention 적용 · 2026-10-13", 9, color=col)
    tb = textbox(slide, 11.33, 7.08, 1.5, 0.3, margin=0, anchor=MSO_ANCHOR.MIDDLE, name="Page")
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    add_runs(p, f"{n} / {TOTAL}", 9, color=col)


def notes(slide, text):
    tf = slide.notes_slide.notes_text_frame
    lines = text.strip().split("\n")
    tf.text = lines[0]
    for line in lines[1:]:
        tf.add_paragraph().text = line
    for p in tf.paragraphs:
        for r in p.runs:
            set_font(r, 12)


def table(slide, x, y, w, col_w, rows, size=11, row_h=0.36, header_size=None,
          bold_cells=(), accent_cells=(), name="Table"):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    shp.name = name
    tbl = shp.table
    # plain style: no banding from the theme style; we paint fills ourselves
    tblPr = tbl._tbl.tblPr
    tblPr.set("firstRow", "1")
    tblPr.set("bandRow", "0")
    for i, cw in enumerate(col_w):
        tbl.columns[i].width = Inches(cw)
    for r in range(nr):
        tbl.rows[r].height = Inches(row_h)
        for c in range(nc):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = HEAD
            else:
                cell.fill.fore_color.rgb = ROW_A if r % 2 else ROW_B
            tf = cell.text_frame
            tf.word_wrap = True
            val = rows[r][c]
            p = tf.paragraphs[0]
            if r == 0:
                add_runs(p, val, header_size or size, color=WHITE, bold=True)
            else:
                b = (r, c) in bold_cells
                col = ACCENT if (r, c) in accent_cells else INK
                add_runs(p, val, size, color=col, bold=b or (r, c) in accent_cells)
    return shp


def caption(slide, x, y, w, text, size=11, color=MUTED, bold=False, h=0.3, name="Caption"):
    tb = textbox(slide, x, y, w, h, margin=0, name=name)
    add_runs(tb.text_frame.paragraphs[0], text, size, color=color, bold=bold)
    return tb


def new_slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    return s


# ---------------------------------------------------------------- cover
s = prs.slides.add_slide(BLANK)
s.background.fill.solid()
s.background.fill.fore_color.rgb = DARK_BG
tb = textbox(s, 0.9, 2.0, 11.5, 2.2, margin=0, anchor=MSO_ANCHOR.BOTTOM, name="Title")
p = tb.text_frame.paragraphs[0]
add_runs(p, "VLA attention의 단계별 적용", 40, color=WHITE, bold=True)
p = tb.text_frame.add_paragraph()
add_runs(p, "HSR (Hidden-state Sub-task Recognition)과 phase-aware target 관리", 26,
         color=RGBColor(0xF2, 0xC6, 0xA8), bold=False)
p.space_before = Pt(10)
tb = textbox(s, 0.9, 4.55, 11.5, 0.5, margin=0, name="Subtitle")
add_runs(tb.text_frame.paragraphs[0], "AG3S 진행 보고 · 2026-10-13", 18, color=RGBColor(0xC9, 0xD2, 0xDE))
# phase motif: pick -> place -> home chips
chips = [("pick", RGBColor(0xE9, 0xA1, 0x00)), ("place", RGBColor(0xE5, 0x7A, 0xA0)),
         ("home", RGBColor(0xE0, 0x4B, 0x4B))]
cx = 0.9
for i, (lab, col) in enumerate(chips):
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx), Inches(5.45), Inches(1.25), Inches(0.42))
    c.adjustments[0] = 0.5
    c.fill.solid(); c.fill.fore_color.rgb = col
    c.line.fill.background(); c.shadow.inherit = False
    c.name = f"Phase {lab}"
    tf = c.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    q = tf.paragraphs[0]; q.alignment = PP_ALIGN.CENTER
    add_runs(q, lab, 14, color=WHITE, bold=True)
    cx += 1.25
    if i < 2:
        a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(cx + 0.06), Inches(5.66),
                                   Inches(cx + 0.34), Inches(5.66))
        a.line.color.rgb = RGBColor(0xC9, 0xD2, 0xDE); a.line.width = Pt(1.5)
        a.line._get_or_add_ln().append(etree.fromstring(
            '<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="triangle"/>'))
        cx += 0.4
footer(s, dark=True)
notes(s, "표지. AG3S 진행 보고 (2026-10-13): VLA attention의 단계별 적용 — HSR (Hidden-state Sub-task Recognition)과 phase-aware target 관리.")


# ---------------------------------------------------------------- slide 1
s = new_slide()
header(s, "문제 — attention이 가리키는 물체는 작업 단계에 따라 바뀐다",
       "같은 attention이라도 \"지금 무엇을 하는 중인가\"를 모르면 들고 가는 중 · 놓은 뒤에 엉뚱한 물체를 target으로 잡는다.")
_, (px, py, pw, ph) = image_fit(s, FIG / "ppt-1013/t59-ppt-timeline-ep1809-W2.png", 0.5, 1.98, 8.2, 4.62, name="Figure timeline")
caption(s, 0.5, py + ph + 0.06, pw, "16D 실제 rollout 1개 (ep1809, banana), frame 0–39", size=10)
bullets(s, 8.95, 2.05, 3.9, 4.7, [
    (0, "매 frame attention만으로 target을 고르면 (phase 미분할)"),
    (1, [("들고 가는 중(held): ", {"bold": True}), ("10 frame 중 8 frame에서 table 위의 다른 과일 선택 (또는 target 없음)", {})]),
    (1, [("놓고 해제한 뒤: ", {"bold": True}), ("crate 안에 이미 놓인 banana를 다시 target으로 선택", {})]),
    (0, [("현재 시스템 (phase 분할): ", {"bold": True}), ("같은 구간에서 쥔 banana 유지, 놓은 뒤 target 해제", {})]),
], size=15, space_after=10)
footer(s)
notes(s, """교수님, 먼저 문제를 실제 기록으로 보여 드리겠습니다. 이 그림은 16D 모델이 banana를 crate에 넣는 rollout 하나를 frame 단위로 펼친 것입니다. 맨 위 띠는 시뮬레이터 기준의 실제 상태로, 쥐기 전, 쥐고 있음, 놓음 순서입니다.
아래에서 두 번째 줄은 phase를 구분하지 않고 매 frame attention이 가장 강한 물체를 target으로 고른 결과입니다. 쥐고 들어 올리는 순간부터 target을 놓쳐서, held 10 frame 중 8 frame에서 table 위의 다른 과일을 고르거나 아무것도 고르지 못합니다. 놓은 뒤에도 crate 안에 이미 놓인 banana를 다시 target으로 잡습니다.
맨 아래 줄은 현재 시스템입니다. 쥔 동안에는 banana를 유지하고, 놓은 뒤에는 target을 비웁니다. attention 자체는 같고, 그것을 언제 믿고 언제 믿지 않을지를 phase가 정해 줘야 한다는 것이 이번 설계의 출발점입니다.""")


# ---------------------------------------------------------------- slide 2
s = new_slide()
header(s, "설계 — phase 신호 두 개와 phase별 attention 사용",
       "attention은 \"무엇을 잡을지\"에만 쓰고, \"쥐었는지\"는 grasp latch가, \"작업이 어디까지 왔는지\"는 HSR이 정한다.")

PAL = {
    "in":    (RGBColor(0xEC, 0xEC, 0xEC), RGBColor(0x55, 0x55, 0x55)),
    "vla":   (RGBColor(0xD6, 0xE4, 0xF5), RGBColor(0x4A, 0x6E, 0x9A)),
    "att":   (RGBColor(0xFF, 0xF3, 0xCC), RGBColor(0xC9, 0xA2, 0x27)),
    "hid":   (RGBColor(0xFB, 0xE3, 0xD6), RGBColor(0xC0, 0x56, 0x2F)),
    "latch": (RGBColor(0xE8, 0xEB, 0xEF), RGBColor(0x5B, 0x65, 0x73)),
    "tc":    (RGBColor(0xE6, 0xE1, 0xF0), RGBColor(0x6E, 0x5C, 0x99)),
}


def node(slide, x, y, w, h, paras, kind, size=11, name=None):
    b = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    b.adjustments[0] = 0.14
    fill, line = PAL[kind]
    b.fill.solid(); b.fill.fore_color.rgb = fill
    b.line.color.rgb = line; b.line.width = Pt(1.25)
    b.shadow.inherit = False
    if name:
        b.name = name
    tf = b.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    first = True
    for runs in paras:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = PP_ALIGN.CENTER
        add_runs(p, runs, size)
    return b


def arrow(slide, x1, y1, x2, y2, color=RGBColor(0x55, 0x5E, 0x6B)):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(1.5)
    c.line._get_or_add_ln().append(etree.fromstring(
        '<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="triangle" w="med" len="med"/>'))
    return c


# block diagram, left region x 0.5..9.1, y 1.95..4.05
Y0 = 1.97
node(s, 0.5, Y0 + 0.28, 1.35, 0.66, ["3 camera +", "instruction"], "in", name="Input")
node(s, 2.1, Y0 + 0.1, 1.5, 1.02, [[("frozen π0.5", {"bold": True})], [("(prefix forward 1회)", {"size": 9.5})]], "vla", name="VLA")
node(s, 3.95, Y0, 1.45, 0.48, ["attention (L8H2)"], "att", name="Attention")
node(s, 3.95, Y0 + 0.74, 1.45, 0.48, ["KV cache (L4 V)"], "hid", name="KV cache")
node(s, 5.7, Y0, 1.55, 0.48, ["target 후보", "(3D grounding)"], "att", size=10, name="Target candidate")
node(s, 5.7, Y0 + 0.74, 1.55, 0.48, [[("HSR", {"bold": True}), (" → pick / place / home", {})]], "hid", size=10, name="HSR")
node(s, 0.5, Y0 + 1.47, 1.42, 0.6, ["gripper 명령 · 개도 ·", "실행 feedback"], "in", size=9, name="Gripper input")
node(s, 2.1, Y0 + 1.47, 5.15, 0.6, [[("grasp latch", {"bold": True}),
                                    ("  (SEARCHING→LATCHED→CLOSING→HELD→PLACED)", {})]], "latch", size=10.5, name="Grasp latch")
node(s, 7.55, Y0, 1.55, 2.07, [[("TargetConfirm", {"bold": True})], "target 채택 ·", "유지 · 해제",
                               [("⇒ target-free ESDF / attached sphere", {"size": 9.5})]], "tc", size=10.5, name="TargetConfirm")
arrow(s, 1.85, Y0 + 0.61, 2.1, Y0 + 0.61)
arrow(s, 3.6, Y0 + 0.4, 3.95, Y0 + 0.24)
arrow(s, 3.6, Y0 + 0.82, 3.95, Y0 + 0.98)
arrow(s, 5.4, Y0 + 0.24, 5.7, Y0 + 0.24)
arrow(s, 5.4, Y0 + 0.98, 5.7, Y0 + 0.98)
arrow(s, 7.25, Y0 + 0.24, 7.55, Y0 + 0.24)
arrow(s, 7.25, Y0 + 0.98, 7.55, Y0 + 0.98)
arrow(s, 7.25, Y0 + 1.77, 7.55, Y0 + 1.77)
arrow(s, 1.92, Y0 + 1.77, 2.1, Y0 + 1.77)

table(s, 0.5, 4.27, 8.6, [1.35, 1.5, 3.2, 2.55], [
    ["단계", "latch", "attention 사용", "그 물체의 field 처리"],
    ["집으러 감 (pick)", "SEARCHING · LATCHED", "attention 1위 cluster가 3 frame 연속이면 target 채택 · 교체", "target만 지운 target-free layer, 손가락만 질의"],
    ["손을 닫는 중", "CLOSING", "freeze: attention이 바뀌어도 target 고정", "유지"],
    ["들고 감 (place)", "HELD", "target 결정에 쓰지 않음, HSR이 새 target 채택을 막음", "쥔 물체 = attached sphere (robot의 일부)"],
    ["놓은 뒤 (home)", "PLACED", "HSR home이면 놓인 물체 해제", "다시 obstacle"],
], size=10.5, row_h=0.4, name="Phase table")
bullets(s, 0.5, 6.62, 8.6, 0.4, [
    (0, [("원칙: ", {"bold": True, "color": ACCENT}), ("HSR label은 carving을 막을 수만 있고 추가할 수는 없다.", {})]),
], size=13, space_after=0, name="Principle")
_, (px, py, pw, ph) = image_fit(s, CROPS / "c3_when_it_changes.png", 9.4, 1.97, 3.43, 4.75, align="right", name="Figure C3")
caption(s, px, py + ph + 0.05, pw, "paper Figure 2 (reference montage v2) C3 패널", size=9)
footer(s)
notes(s, """설계는 신호를 세 가지로 나눴습니다. attention은 무엇을 잡을지에만 씁니다. 실제로 쥐었는지는 gripper 개도와 실행 feedback으로 판단하는 grasp latch가 정합니다. 작업이 집기, 옮기기, 복귀 중 어디에 있는지는 HSR이 정합니다.
표처럼 attention이 target을 정하는 것은 집으러 가는 단계뿐입니다. 손을 닫는 동안에는 target을 고정하고, 들고 가는 동안에는 쥔 물체를 robot의 일부로 붙여 충돌 검사를 합니다. 놓은 뒤 HSR이 home이라고 판단하면 그 물체를 다시 장애물로 돌려놓습니다.
핵심 원칙은 HSR이 지울 대상을 추가할 수는 없고 막을 수만 있다는 것입니다. HSR이 틀려도 장애물이 사라지는 방향으로는 틀리지 않도록 일부러 권한을 좁혔습니다.""")


# ---------------------------------------------------------------- slide 3
s = new_slide()
header(s, "HSR 구현 — VLA를 재학습하지 않고 hidden state에서 subtask를 읽는다",
       "π0.5의 subtask text 생성은 우리 checkpoint에서 쓸 수 없어서, frozen KV cache 위에 linear probe 하나를 얹었다.")
_, (px, py, pw, ph) = image_fit(s, FIG / "subtask/subtask-inference-paths.png", 0.5, 1.98, 6.75, 4.85, name="Figure inference paths")
caption(s, px, py + ph + 0.05, pw, "π0.5 논문 경로 (①) 대 우리 16D checkpoint 실제 경로 (②), subtask를 꺼내는 세 길 (③)", size=10)
bullets(s, 7.55, 1.98, 5.3, 2.95, [
    (0, [("시도 1: ", {"bold": True}), ("zero-shot text decode → 사용 불가", {})]),
    (1, "들고 가는 36 frame 중 30 frame에서 영어 대신 FAST action token 출력"),
    (1, "우리 16D는 subtask text로 학습된 적이 없음"),
    (0, [("채택: ", {"bold": True, "color": ACCENT}), ("HSR = KV cache (layer 4, V) [text 평균 | image 평균] 512차원 → logistic regression", {})]),
    (1, "재학습 없음, attention 계산 경로에 추가 forward 없음 (같은 prefix pass의 KV cache 재사용), probe 추론 0.092\u00a0ms"),
    (0, "출력이 N회 연속 같으면 확정 label (debounce)"),
], size=13, space_after=4)
caption(s, 7.55, 4.5, 5.3, "각주: feature 9종 비교에서 val 정확도 0.974–0.981로 차이가 작았고, layer 4는 layer 17과 동률 (부록 A5)",
        size=10, h=0.45, name="Footnote")
_, (px, py, pw, ph) = image_fit(s, CROPS / "subtask-b-scene-top-row-compact.png", 7.55, 5.02, 5.3, 1.98, align="right", name="Figure HSR scene")
footer(s)
notes(s, """HSR은 Hidden-state Sub-task Recognition의 약자입니다. 원래 π0.5 논문은 subtask 문장을 먼저 생성하고 그 문장을 조건으로 action을 만듭니다. 그런데 우리가 fine-tuning한 16D checkpoint는 subtask text로 학습된 적이 없어서, 실제로 decode해 보니 들고 가는 구간 36 frame 중 30 frame에서 영어가 아니라 action token이 나왔습니다.
그래서 재학습 없이 모델 내부 표현에서 읽는 방법을 택했습니다. attention을 계산할 때 이미 만들어지는 KV cache에서 512차원 feature를 꺼내고, 그 위에 logistic regression 하나만 학습했습니다. attention 경로에 forward를 더하지 않고, probe 추론은 0.1 ms 미만입니다. layer는 9개 후보를 비교했는데 차이가 작아서, 특정 layer가 뚜렷이 우월하다고 주장하지는 않습니다. 오른쪽 그림처럼 실제 장면에서 pick, place, home 확률이 거의 1로 나옵니다.""")


# ---------------------------------------------------------------- slide 4
s = new_slide()
header(s, "HSR 검증 — 잘 읽히지만, 무엇을 읽는지에는 한계가 있다",
       "phase는 높은 정확도로 읽히지만 instruction · 실제 쥠 상태에는 둔감하다 → 권한을 좁게 준 것이 맞았다.")
table(s, 0.5, 1.98, 12.33, [4.6, 7.73], [
    ["항목", "값"],
    ["HSR 정확도: val / closed loop", "0.981 / 0.959"],
    ["gripper 값을 가렸을 때", "0.980 (state가 아니라 image에서 읽음)"],
    ["팔 관절만 쓴 control", "0.697"],
    ["instruction의 target 명사를 바꿨을 때 (T48a)", "반응 없음: place → pick 0.040 (변형 전 오류율과 같음)"],
    ["쥔 물체를 image에서 지웠을 때 (T48b, pilot)", "정답 pick 1 / 13 → 쥠 상태보다 팔 자세 순서를 읽을 가능성"],
    ["원인이 fine-tuning인가 (T50)", "아님 (base 모델도 같은 기준 미달)"],
], size=12, row_h=0.37, accent_cells={(4, 1), (5, 1)}, name="HSR table")
_, (px, py, pw, ph) = image_fit(s, T48B_HELDOUT, 0.5, 4.72, 7.2, 2.24, align="left", name="Figure T48b")
caption(s, px + pw + 0.3, py, 12.83 - (px + pw + 0.3),
        "T48b counterfactual edits, held-out primary panel (E1 empty = 쥔 물체 지움)", size=10.5, h=0.5)
bullets(s, px + pw + 0.3, py + 0.6, 12.83 - (px + pw + 0.3), 1.2, [
    (0, [("주의: ", {"bold": True, "color": ACCENT}),
         ("dataset phase는 scripted라 frame 번호만으로 1.000 → closed loop 0.959가 더 정직한 수치", {})]),
], size=12.5, space_after=0, name="Caveat")
footer(s)
notes(s, """HSR 자체의 검증입니다. 정확도는 validation 0.98, 실제 closed loop 기록에서 0.96이고, gripper 값을 가려도 그대로라 image에서 읽고 있다는 것까지 확인했습니다.
하지만 입력을 일부러 바꿔 보는 counterfactual 실험에서 한계가 나왔습니다. instruction의 물체 이름을 바꿔도 반응하지 않았고, 쥔 물체를 image에서 지워도 대부분 여전히 place라고 답했습니다. "물체를 쥐었다"보다는 "팔이 이 자세에 있다"를 읽고 있을 가능성이 있습니다.
그래서 HSR에 attach나 target 선택 권한을 주지 않은 설계가 맞았다고 봅니다. 쥐었는지는 물리 신호인 grasp latch가 판단하고, HSR은 작업 단계를 알려 주는 보조 신호로만 씁니다.""")


# ---------------------------------------------------------------- slide 5
s = new_slide()
header(s, "Closed-loop 검증 — HSR gate를 켜면 놓은 뒤의 잘못된 target이 사라진다",
       "같은 episode·seed 48쌍에서 놓은 뒤 target 교체 21 run → 0 run. 성공률은 변화 없음.")
_, (px, py, pw, ph) = image_fit(s, FIG / "subtask-d/subtask-d-scene-1800_s18001.png", 0.5, 1.98, 4.9, 4.62, align="left", name="Figure gate scene")
caption(s, px, py + ph + 0.05, 4.9, "실제 씬 + ESDF 단면 (위 gate off / 아래 gate on)", size=10)
caption(s, 5.75, 1.98, 7.1, "16D, 24 episode × 2 seed = 48쌍", size=12, color=INK, bold=True, name="Table caption")
table(s, 5.75, 2.36, 7.08, [3.4, 1.84, 1.84], [
    ["항목", "gate off (latch만)", "gate on (latch + HSR)"],
    ["놓은 뒤 target 교체가 있었던 run", "21", "0"],
    ["놓은 뒤 carving이 켜진 chunk", "965 / 1,051", "57 / 1,041"],
    ["해제 후 정지(HOLD) chunk: 첫 구현 → 최종", "—", "984 → 28"],
    ["마지막 팔 자세의 준비 자세 거리 p95 (rad)", "0.424", "0.164"],
    ["success / grasp / place", "29 / 35 / 29", "28 / 34 / 28"],
], size=12.5, row_h=0.5, accent_cells={(1, 2), (2, 2)}, name="Gate table")
# stat callout: 21 -> 0
for i, (big, small, col) in enumerate([("21 → 0", "놓은 뒤 target 교체가 있었던 run", ACCENT),
                                       ("29 / 28", "success (gate off / gate on)", MUTED)]):
    bx = 5.75 + i * 3.6
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(bx), Inches(6.02), Inches(3.45), Inches(0.9))
    b.adjustments[0] = 0.12
    b.fill.solid(); b.fill.fore_color.rgb = TINT if i == 0 else ROW_B
    b.line.fill.background(); b.shadow.inherit = False
    b.name = f"Callout {i}"
    tf = b.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    add_runs(p, big, 24, color=col, bold=True)
    p = tf.add_paragraph()
    p.alignment = PP_ALIGN.LEFT
    add_runs(p, small, 10.5, color=MUTED)
footer(s)
notes(s, """실제 closed loop에서 HSR gate를 끈 경우와 켠 경우를 같은 episode, 같은 seed 48쌍으로 비교했습니다. 위쪽 gate off에서는 사과를 놓은 뒤에도 crate 밖 탁자 쪽 영역이 계속 지워져 있습니다. target이 다른 과일로 넘어갔기 때문입니다. 이런 교체가 25 run 중 21 run에서 일어났고, gate를 켜면 0이 됩니다.
처음 구현에서는 target을 비우자 시스템이 인증되지 않은 상태로 보고 robot을 멈추는 문제가 있었습니다. 두 번 수정해서 해제 후 정지 chunk를 984에서 28로 줄였고, robot이 준비 자세로 돌아오게 했습니다.
다만 성공률은 29에서 28로 차이가 없습니다. 이 결과는 성공률 향상이 아니라 "놓은 뒤 잘못된 carving 제거"로 말씀드리는 것이 정확합니다.""")


# ---------------------------------------------------------------- slide 6
s = new_slide()
header(s, "16D 정량 — phase를 나누지 않으면 무엇을 고르는가 (T59)",
       "16D에서 phase 미분할 grounding은 들고 가는 동안 target을 0.59밖에 유지하지 못하고, 놓은 뒤에는 이미 놓인 물체를 다시 고른다.")
image_fit(s, FIG / "ppt-1013/t59-ppt-phase-bars.png", 0.5, 1.98, 6.05, 3.41, name="Figure T59 bars")
image_fit(s, FIG / "ppt-1013/t59-ppt-scene-ep1809.png", 6.78, 1.98, 6.05, 3.41, name="Figure T59 scene")
caption(s, 0.5, 5.44, 6.05, "T53 16D 16 episode, offline, 같은 frame에서 비교 (target 비율, episode 평균 [95% CI])",
        size=10.5, color=INK, bold=True, h=0.42, name="Table caption")
table(s, 0.5, 5.8, 6.05, [1.75, 2.3, 2.0], [
    ["단계", "phase 미분할", "phase 분할 (serving)"],
    ["쥐기 전 (477 frame)", "0.82 [0.69, 0.93]", "0.89 [0.75, 1.00]"],
    ["들고 가는 중 (106 frame)", "0.59 [0.39, 0.76], 다른 물체 0.35", "0.86 [0.63, 1.00]"],
], size=10.5, row_h=0.36, accent_cells={(2, 2)}, name="T59 table")
bullets(s, 6.78, 5.47, 6.05, 1.55, [
    (0, "놓고 해제된 119 frame (HSR home)에서 phase 미분할이 고른 것: 이미 놓인 물체 81 · 다른 물체 7 · table 5 · crate 1 · 없음 25"),
    (0, "들고 가는 동안 attention 자체는 16D에서도 crate에 몰림 (14D와 같음). 다만 crate는 크기 규칙(admissibility)에서 빠져 선택되지 않고 (held에서 crate 선택 0 / 106), 남은 약한 attention 중 table 위 다른 과일이 선택됨 → 다음 슬라이드"),
], size=11.5, space_after=5)
footer(s)
notes(s, """앞의 예시가 한 episode였다면, 이 슬라이드는 16D 모델 rollout 16개 전체입니다. 새로 rollout하지 않고 기존 기록에서 frame마다 두 방식을 같은 시점에 비교했습니다.
쥐기 전에는 두 방식이 비슷합니다. 차이는 들고 가는 동안 생깁니다. phase를 나누지 않으면 target 유지 비율이 0.59로 떨어지고 35%는 옆의 다른 과일을 고릅니다. 현재 시스템은 같은 frame에서 0.86을 유지합니다.
놓은 뒤가 더 분명합니다. 현재 시스템이 target을 해제한 119 frame에서, phase를 나누지 않았다면 81 frame에서 이미 놓은 물체를 다시 target으로 잡았을 것입니다. 오른쪽 가운데 장면이 바로 그 순간입니다.
왜 이런 일이 생기는지는 다음 슬라이드에서 말씀드리겠습니다.""")


# ---------------------------------------------------------------- slide 6-2
s = new_slide()
header(s, "원인 분석 — 들고 가는 동안 policy attention은 crate를 본다",
       "실패의 시작은 attention 단계다. 들고 가는 동안 L8H2 attention은 target이 아니라 crate에 몰리고, crate는 크기 규칙에서 빠지므로 남은 약한 attention끼리 경쟁하다 다른 과일이 선택된다.")
_, (px, py, pw, ph) = image_fit(s, T59_DIAG, 0.5, 1.95, 10.7, 3.12, align="left", name="Figure T59 diag")
cap = textbox(s, px + pw + 0.22, py + 0.1, 12.83 - (px + pw + 0.22), ph - 0.1, margin=0, name="Caption")
for j, line in enumerate(["T59 diag", "U = phase 미분할", "(a) 2D attention 질량", "(b) 3D seed 구성",
                          "(c) 처음 실패한 단계"]):
    q = cap.text_frame.paragraphs[0] if j == 0 else cap.text_frame.add_paragraph()
    add_runs(q, line, 10.5, color=INK if j < 2 else MUTED, bold=j == 0)
    q.space_after = Pt(6)
caption(s, 0.5, 5.2, 4.5, "T53 16D 16 episode, offline", size=10.5, color=INK, bold=True, h=0.28, name="Table caption")
table(s, 0.5, 5.52, 4.5, [2.3, 1.0, 1.2], [
    ["지표", "쥐기 전", "들고 가는 중"],
    ["2D attention 중 target 몫", "0.106", "0.021"],
    ["2D attention 중 crate 몫", "0.057", "0.473"],
    ["3D seed 중 crate 비율", "0.42", "0.92"],
], size=10.5, row_h=0.32, accent_cells={(2, 2), (3, 2)}, name="T59 diag table")
bullets(s, 5.25, 5.17, 7.58, 1.8, [
    (0, "들고 가는 중 phase 미분할이 틀린 37 frame의 처음 실패 단계: target이 cloud에서 사라짐 0 · target에 seed 없음 17 · 다른 cluster에 병합 2 · 후보는 있으나 다른 과일이 attention 순위에서 이김 18"),
    (0, "가림 · self-filter 때문이 아님 (106 frame 모두 target 점이 관측됨)"),
    (0, [("놓은 뒤: ", {"bold": True}), ("phase 미분할은 frame마다 처음부터 고르므로 \"이미 놓았다\"는 정보가 없음. crate 안 과일은 잡을 수 있는 크기라 후보로 남고, serving이 그것을 다시 고르지 않은 것은 HSR gate 때문 (81 / 81 frame이 gate로 막힘)", {})]),
    (0, [("확인 안 된 것: ", {"bold": True}), ("policy가 왜 들고 가는 동안 crate를 보는지 (가설: 목적지를 본다), 2D에서 거의 없는 다른 과일 attention이 3D seed를 받는 경로 (가설: 16×16 patch lifting 번짐)", {})]),
], size=11, space_after=3)
footer(s)
notes(s, """원인을 단계별로 분해했습니다. 들고 가는 동안 L8H2 attention의 절반 가까이가 crate에 있고, target에는 2%밖에 없습니다. 3D seed로 옮기면 92%가 crate입니다. 예전 14D에서 보았던 "들고 가는 동안 목적지를 본다"는 현상이 16D에서도 같다는 뜻입니다.
다만 지금은 crate가 크기 규칙에서 빠지기 때문에 target으로 선택되지는 않습니다. 대신 남은 약한 attention끼리 경쟁해서, table 위의 다른 과일이 이기거나 target에 seed가 하나도 남지 않습니다. 손에 가려서 target이 사라진 경우는 한 번도 없었습니다.
놓은 뒤의 재선택은 구조적인 이유입니다. 매 frame 새로 고르는 방식에는 "이미 놓았다"는 정보가 없고, crate 안의 과일은 잡을 수 있는 크기라 후보로 남습니다. 현재 시스템에서는 HSR gate가 81 frame 모두 이를 막았습니다.
policy가 왜 들고 가는 동안 crate를 보는지는 아직 측정하지 못했습니다. 목적지를 본다는 해석은 가설로 남겨 두겠습니다.""")


# ---------------------------------------------------------------- slide 7
s = new_slide()
header(s, "정리 · 한계 · 논의드릴 점", None)
cards = [
    ("정리", RGBColor(0x1D, 0x8A, 0x5E), [
        "attention은 pick 단계에서만 target을 정하고, 이후 단계는 grasp latch와 HSR이 물체의 역할을 바꾼다",
        "HSR: 재학습 없이, 추가 비용 0.1\u00a0ms 미만으로 phase를 0.96 정확도로 읽음",
        "효과: 놓은 뒤 target 교체 21 → 0 run, 들고 가는 동안 target 유지 0.59 → 0.86 (offline)",
    ], False),
    ("한계 (측정하지 않았거나 아직 나쁜 것)", ACCENT, [
        "성공률 향상 없음 (29 대 28)",
        "HSR은 instruction · 쥠 상태에 둔감 → 보조 구성 요소로 둠",
        "strong rule(grasp latch만)과 phase source 맞비교는 진행 중 (T55)",
        "16D 정량(T59)은 offline. phase 미분할 closed loop는 없음",
    ], False),
    ("논의드릴 점", RGBColor(0x3B, 0x5B, 0x92), [
        "HSR을 논문에서 \"보조 구성 요소\"로 두는 범위가 적절한지",
        "phase 미분할 arm을 closed loop로 확인할 필요가 있는지",
        "크기로 구분되지 않는 목적지(bowl · plate)에서 gate를 검증할지",
    ], True),
]
cw, gap = 3.93, 0.27
for i, (head, col, items, numbered) in enumerate(cards):
    x = 0.5 + i * (cw + gap)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.2), Inches(cw), Inches(5.0))
    card.adjustments[0] = 0.04
    card.fill.solid(); card.fill.fore_color.rgb = ROW_B
    card.line.fill.background(); card.shadow.inherit = False
    card.name = f"Card {head}"
    circ = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.25), Inches(1.42), Inches(0.5), Inches(0.5))
    circ.fill.solid(); circ.fill.fore_color.rgb = col
    circ.line.fill.background(); circ.shadow.inherit = False
    tf = circ.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = 0
    q = tf.paragraphs[0]; q.alignment = PP_ALIGN.CENTER
    add_runs(q, str(i + 1), 16, color=WHITE, bold=True)
    tb = textbox(s, x + 0.88, 1.38, cw - 1.0, 0.6, margin=0, anchor=MSO_ANCHOR.MIDDLE, name=f"Head {head}")
    hp = tb.text_frame.paragraphs[0]
    if i == 1:
        add_runs(hp, "한계", 17, color=col, bold=True)
        hp = tb.text_frame.add_paragraph()
        add_runs(hp, "(측정하지 않았거나 아직 나쁜 것)", 12, color=col, bold=False)
    else:
        add_runs(hp, head, 17, color=col, bold=True)
    tb = textbox(s, x + (0.2 if numbered else 0.25), 2.2, cw - (0.3 if numbered else 0.45), 3.9, margin=0, name=f"Body {head}")
    tf = tb.text_frame
    for j, it in enumerate(items):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        if numbered:
            add_runs(p, [(f"{j + 1}. ", {"bold": True, "color": col}), (it, {})], 15)
            no_bullet(p)
        else:
            add_runs(p, it, 15)
            set_bullet(p, 0, 15)
        p.space_after = Pt(14)
        p.line_spacing = 1.12
footer(s)
notes(s, """정리하면, attention은 무엇을 잡을지만 정하고, 쥔 뒤와 놓은 뒤에는 물리 신호와 HSR이 물체의 역할을 바꾸는 구조입니다. 그 결과 놓은 뒤의 잘못된 target 교체가 사라졌고, 16D 기록에서 들고 가는 동안 target 유지율이 0.59에서 0.86으로 높았습니다.
한계도 분명합니다. 성공률은 오르지 않았고, HSR은 instruction이나 실제 쥠 상태를 잘 구분하지 못합니다. 그래서 HSR은 핵심 기여가 아니라 보조 구성 요소로 두고 있고, grasp latch만 쓰는 방식과의 직접 비교는 현재 T55에서 진행 중입니다.
세 가지에 대해 교수님 의견을 듣고 싶습니다. HSR을 논문에서 어느 범위로 주장할지, phase 미분할을 closed loop로 확인할지, 크기로 구분되지 않는 목적지에서 검증이 필요한지입니다.""")


# ---------------------------------------------------------------- appendix
def appendix(title, fig, text_runs, script, fig_box=(0.5, 1.3, 8.4, 5.55), text_box=(9.2, 1.3, 3.63, 5.5)):
    s = new_slide()
    header(s, title, None)
    if fig is not None:
        image_fit(s, fig, *fig_box, name="Figure")
    tb = textbox(s, *text_box, margin=0.18, name="Body")
    sh = tb
    tb.fill.solid(); tb.fill.fore_color.rgb = ROW_B
    tf = tb.text_frame
    p = tf.paragraphs[0]
    add_runs(p, text_runs, 14)
    p.line_spacing = 1.2
    footer(s)
    notes(s, script)
    return s


appendix("부록 A1. 이전 모델(14D)에서의 문제", FIG / "ppt-1013/attention-latch-crop-f0-20.png",
         "들고 가는 frame 7–20 동안 grounding이 crate 선택 → latch 도입 계기 (14D 시기 기록)",
         "이전 14D 모델에서는 들고 가는 동안 attention이 목적지인 crate를 가리켰고, 그대로 target으로 쓰면 crate가 지워졌습니다. 이것이 grasp latch를 도입한 계기였습니다.",
         text_box=(9.2, 1.3, 3.63, 1.45))
appendix("부록 A2. 손을 닫는 중의 문제 (16D, T20)", FIG / "t20/t20-t14-topview.png",
         "사과가 손에 가려진 frame에서 1위가 crate · pear로 바뀜 (점수 0.015–0.047) → freeze 근거. 당시 self-filter가 사과 점을 지운 것도 함께 작용",
         "손을 닫는 동안에는 사과가 손가락에 가려 attention 1위가 crate나 배로 바뀌었습니다. 점수는 매우 낮았지만 이전 규칙은 target을 바꿨고, 그래서 손을 닫는 동안 target을 고정하는 freeze를 넣었습니다.",
         fig_box=(0.5, 1.3, 12.33, 4.3), text_box=(0.5, 5.8, 12.33, 0.95))

# A3: text only + stat callouts drawn from the same sentence
s = new_slide()
header(s, "부록 A3. 측정 정의의 변경 (투명성)", None)
tb = textbox(s, 0.5, 1.3, 12.33, 1.75, margin=0.25, anchor=MSO_ANCHOR.MIDDLE, name="Body")
tb.fill.solid(); tb.fill.fore_color.rgb = ROW_B
p = tb.text_frame.paragraphs[0]
add_runs(p, "serving 쪽 class의 원래 정의는 \"같은 frame의 GT 위치\". 쥔 물체는 attached로 처리되어 기록된 centroid가 마지막 관측 위치에 머물러 held target이 0.14로 왜곡됨 → 결과를 본 뒤 \"centroid를 관측한 frame의 GT\"(0.86)로 주 정의 변경 (사용자 판정), 두 값 모두 기록", 16)
p.line_spacing = 1.3
for i, (big, small, col) in enumerate([("0.14", "원래 정의: 같은 frame의 GT 위치\n(held target, 왜곡)", MUTED),
                                       ("0.86", "주 정의: centroid를 관측한 frame의 GT\n(held target)", ACCENT)]):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5 + i * 6.33), Inches(3.35), Inches(6.0), Inches(3.4))
    b.adjustments[0] = 0.08
    b.fill.solid(); b.fill.fore_color.rgb = TINT if i else ROW_B
    b.line.fill.background(); b.shadow.inherit = False
    b.name = f"Callout {big}"
    tf = b.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
    tf.margin_left = Inches(0.3)
    q = tf.paragraphs[0]
    q.alignment = PP_ALIGN.LEFT
    add_runs(q, big, 60, color=col, bold=True)
    for line in small.split("\n"):
        q = tf.add_paragraph()
        q.alignment = PP_ALIGN.LEFT
        add_runs(q, line, 16, color=INK)
footer(s)
notes(s, "측정 정의를 결과를 본 뒤 한 번 바꿨기 때문에 말씀드립니다. 원래 정의로는 쥔 물체의 위치가 갱신되지 않아 값이 왜곡되어서, 관측 시점 기준으로 바꿨고 두 값을 모두 남겼습니다.")

appendix("부록 A4. 놓은 뒤 구간(W2)의 쏠림", FIG / "ppt-1013/t59-ppt-phase-bars-W2.png",
         "placed 153 frame 중 88 frame은 준비 자세로 복귀하지 못한 2 episode에서 나옴 → 주 수치는 place 직후 종료(W1) 기준",
         "놓은 뒤 구간은 복귀하지 못한 두 episode가 절반 이상을 차지해서, 본문의 주 수치는 place 직후에서 자른 기준으로 드렸습니다.",
         fig_box=(0.5, 1.3, 8.75, 5.12), text_box=(9.5, 1.3, 3.33, 2.15))

# A5: feature comparison table (kv_L4 row highlighted) + rationale box
s = new_slide()
header(s, "부록 A5. HSR feature 선택 근거", None)
caption(s, 0.5, 1.3, 12.33, "9개 feature를 같은 logistic regression으로 비교 (선택 기준 = val 정확도 최대, 동률은 목록 순서)",
        size=14, color=INK, bold=True, h=0.36, name="Table caption")
A5_ROWS = [
    ["feature", "val", "closed loop (246 frame)"],
    ["prefix_last", "0.9765", "0.9309"],
    ["prefix_img", "0.9804", "0.9553"],
    ["kv_L0", "0.9800", "0.9593"],
    ["kv_L4 (채택)", "0.9808", "0.9593"],
    ["kv_L8", "0.9781", "0.9553"],
    ["kv_L12", "0.9796", "0.9634"],
    ["kv_L17", "0.9808", "0.9593"],
    ["ae_final", "0.9800", "0.9634"],
    ["attn_L8", "0.9742", "0.9553"],
]
KV4 = 4
shp = table(s, 0.5, 1.85, 6.4, [2.4, 1.6, 2.4], A5_ROWS, size=13, row_h=0.46,
            accent_cells={(KV4, c) for c in range(3)}, name="Feature table")
for c in range(3):
    cell = shp.table.cell(KV4, c)
    cell.fill.solid(); cell.fill.fore_color.rgb = TINT
tb = textbox(s, 7.3, 1.85, 5.53, 2.1, margin=0.18, name="Body")
tb.fill.solid(); tb.fill.fore_color.rgb = ROW_B
p = tb.text_frame.paragraphs[0]
add_runs(p, [("linear probe를 쓴 근거: ", {"bold": True}),
             ("action expert의 입력은 prefix KV · noise · time뿐인데 action이 phase마다 다르므로 phase 정보는 KV 안에 있어야 한다는 논리 (문헌 인용 없음). K, 나머지 layer, 다른 pooling, nonlinear probe는 비교하지 않음", {})], 14)
p.line_spacing = 1.2
footer(s)
notes(s, "layer 4는 layer 17과 val 정확도가 같았고 목록 순서로 정해졌습니다. closed loop에서는 layer 12나 action expert 마지막 hidden이 한 frame 더 맞혔습니다. 즉 특정 layer가 우월해서가 아니라, 대부분의 feature에서 phase가 선형으로 읽힌다는 것이 정확한 해석입니다.")

# ---------------------------------------------------------------- Q&A
s = new_slide()
header(s, "예상 질문", None)
qa = [
    ("Q. HSR 없이 grasp latch만으로 충분하지 않은가?",
     "손을 닫는 중 · 들고 가는 중은 latch로 해결. 하지만 latch는 \"작업이 끝났다\"를 몰라 놓은 뒤 target 교체(21 run)를 막지 못함 → HSR 도입 이유. 직접 비교는 T55 진행 중."),
    ("Q. 성공률이 안 오르는데 왜 필요한가?",
     "이 과제에서는 놓은 뒤 잘못 지워진 물체에 닿는 상황이 드묾. 효과는 놓은 뒤 장애물 정보를 올바르게 유지하는 것. 연속 작업 · 다른 목적지 영향은 미측정."),
    ("Q. 0.59 대 0.86은 유의한가?",
     "episode bootstrap 95% CI [0.39, 0.76] · [0.63, 1.00] 일부 겹침. held가 있는 episode 10개라 방향은 일관되나 강한 통계 주장은 하지 않음."),
]
y = 1.25
for i, (q, a) in enumerate(qa):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y), Inches(12.33), Inches(1.72))
    b.adjustments[0] = 0.07
    b.fill.solid(); b.fill.fore_color.rgb = ROW_B
    b.line.fill.background(); b.shadow.inherit = False
    b.name = f"QA {i + 1}"
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.3)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    add_runs(p, q, 17, color=ACCENT, bold=True)
    p.space_after = Pt(6)
    p = tf.add_paragraph()
    p.alignment = PP_ALIGN.LEFT
    add_runs(p, [("→ ", {"bold": True}), (a, {})], 15)
    p.line_spacing = 1.15
    y += 1.88
footer(s)
notes(s, "\n".join(f"{q}\n→ {a}" for q, a in qa))

assert _count[0] == TOTAL, _count[0]
for sl in prs.slides:
    for shp in sl.shapes:
        st = shp._element.find(qn("p:style"))
        if st is not None:
            shp._element.remove(st)
# Korean: break lines between words, not inside a word (eaLnBrk="0")
for sl in prs.slides:
    for el in sl._element.iter(qn("a:p")):
        el.get_or_add_pPr().set("eaLnBrk", "0") if hasattr(el, "get_or_add_pPr") else None
import sys
if OUT.exists() and "--rebuild" not in sys.argv:
    raise SystemExit(f"{OUT} already exists; refusing to overwrite (pass --rebuild to overwrite this script's own output)")
prs.save(OUT)
print("wrote", OUT)
