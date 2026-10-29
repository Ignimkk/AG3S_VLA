import sys
from PIL import Image, ImageDraw, ImageFont
src, out = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
s = 2174 / 2000
BG = (252, 252, 251)
plot = im.crop((0, int(62 * s), int(1003 * s), 521))   # 패널 (a) 의 frame 0-20, 제목 제외
d = ImageDraw.Draw(plot)
W, H = plot.size
# 잘린 legend 지우기 (원 위치: 패널 안 오른쪽 아래)
y0, y1 = int(377 * s) - int(62 * s) - 2, int(408 * s) - int(62 * s) + 2
px = plot.load()
for x in range(830, W):
    c = px[x, y0 - 4]
    for y in range(y0, y1):
        px[x, y] = c
# frame 19 (place 완료, t_step 152) 표시: frame 0 → x 159, frame 20 → x 1067
x19 = 159 + 19 * (1067 - 159) / 20
f = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 22, index=1)
fb = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 26, index=1)
for y in range(70, H - 75, 14):
    d.line((x19, y, x19, y + 7), fill=(60, 60, 60), width=3)
d.text((x19 - 8, 300), "place 완료\n(frame 19)", font=f, fill=(60, 60, 60), anchor="ra", align="right")
leg = im.crop((int(773 * s), int(379.5 * s), int(1318 * s), int(405 * s)))
top, bottom = 60, 95
canvas = Image.new("RGB", (W, top + H + bottom), BG)
canvas.paste(plot, (0, top))
cd = ImageDraw.Draw(canvas)
cd.text((20, 14), "grounding 이 매 frame 고른 물체와 attention 점수 — pick → carry → place (frame 0–20)", font=fb, fill=(30, 30, 30))
cd.text((W // 2, top + H + 4), "frame (run_0004, t_step = frame × 8, 14D 시기 기록)", font=f, fill=(60, 60, 60), anchor="ma")
canvas.paste(leg, ((W - leg.size[0]) // 2, top + H + 40))
canvas.save(out)
print(canvas.size)
