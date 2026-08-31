"""Отрисовка снимка экрана терминала в изображение."""
from __future__ import annotations

from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageFont

MENLO = "/System/Library/Fonts/Menlo.ttc"
EMOJI = "/System/Library/Fonts/Apple Color Emoji.ttc"
EMOJI_STRIKE = 20  # единственный размер, который Pillow берёт у Apple Color Emoji

BACKGROUND = (22, 24, 29)
FOREGROUND = (223, 226, 232)

# Базовая палитра ANSI. Цвета 256-палитры эмулятор отдаёт готовыми hex-строками.
NAMED: Dict[str, Tuple[int, int, int]] = {
    "black": (44, 47, 54),
    "red": (232, 92, 92),
    "green": (126, 196, 116),
    "brown": (214, 175, 90),
    "yellow": (214, 175, 90),
    "blue": (98, 152, 226),
    "magenta": (198, 120, 221),
    "cyan": (86, 182, 194),
    "white": (223, 226, 232),
    "brightblack": (110, 116, 128),
    "brightred": (255, 121, 121),
    "brightgreen": (150, 220, 140),
    "brightyellow": (240, 200, 110),
    "brightblue": (120, 175, 245),
    "brightmagenta": (218, 145, 240),
    "brightcyan": (110, 205, 215),
    "brightwhite": (255, 255, 255),
}


def resolve(color: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if not color or color == "default":
        return default
    if color in NAMED:
        return NAMED[color]
    if len(color) == 6:
        try:
            return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))
        except ValueError:
            pass
    return default


class FrameRenderer:
    def __init__(self, cols: int, rows: int, font_size: int = 18, padding: int = 14) -> None:
        self.cols, self.rows, self.padding = cols, rows, padding
        self.regular = ImageFont.truetype(MENLO, font_size, index=0)
        self.bold = ImageFont.truetype(MENLO, font_size, index=1)
        try:
            self.emoji_font = ImageFont.truetype(EMOJI, EMOJI_STRIKE)
        except OSError:
            self.emoji_font = None
        self.cell_w = int(round(self.regular.getlength("M")))
        ascent, descent = self.regular.getmetrics()
        self.cell_h = ascent + descent
        self.ascent = ascent
        # Размеры кадра делаем чётными: этого требует кодирование в yuv420p.
        self.caption_h = int(self.cell_h * 2.2)
        self.width = (self.cols * self.cell_w + 2 * padding + 1) // 2 * 2
        self.height = (self.rows * self.cell_h + 2 * padding
                       + self.caption_h + 1) // 2 * 2
        self._emoji_cache: Dict[str, Image.Image] = {}

    # --- символы, которых нет в моноширинном шрифте --------------------
    def _draw_braille(self, draw: ImageDraw.ImageDraw, ch: str,
                      x: int, y: int, color: Tuple[int, int, int]) -> None:
        """Брайль (спиннер) рисуется по битам кодовой точки: 2 колонки на 4 ряда."""
        bits = ord(ch) - 0x2800
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (0, 3), (1, 3)]
        radius = max(2, self.cell_w // 4)
        for bit, (col, row) in enumerate(positions):
            if bits >> bit & 1:
                cx = x + self.cell_w * (1 + 2 * col) // 4
                cy = y + self.cell_h * (1 + 2 * row) // 8
                draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)

    def _emoji_image(self, ch: str) -> Image.Image:
        if ch not in self._emoji_cache:
            box = EMOJI_STRIKE * 2
            tile = Image.new("RGBA", (box, box), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((0, 0), ch, font=self.emoji_font, embedded_color=True)
            target = (self.cell_w * 2, self.cell_h)
            size = min(target)
            self._emoji_cache[ch] = tile.crop(tile.getbbox() or (0, 0, box, box)).resize(
                (size, size), Image.LANCZOS)
        return self._emoji_cache[ch]

    # --- кадр целиком ---------------------------------------------------
    def render(self, snapshot, caption: str = "") -> Image.Image:
        image = Image.new("RGB", (self.width, self.height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        for row_index, row in enumerate(snapshot):
            y = self.padding + row_index * self.cell_h
            for col_index, (ch, fg, bg, bold, reverse) in enumerate(row):
                if not ch or ch == " ":
                    if bg == "default" and not reverse:
                        continue
                x = self.padding + col_index * self.cell_w
                color = resolve(fg, FOREGROUND)
                back = resolve(bg, BACKGROUND)
                if reverse:
                    color, back = back, color
                if back != BACKGROUND:
                    draw.rectangle([x, y, x + self.cell_w - 1, y + self.cell_h - 1], fill=back)
                if not ch or ch == " ":
                    continue
                point = ord(ch[0])
                if 0x2800 <= point <= 0x28FF:
                    self._draw_braille(draw, ch, x, y, color)
                elif point > 0x1F000 or ch in "✨⚠":
                    if self.emoji_font is not None:
                        tile = self._emoji_image(ch)
                        image.paste(tile, (x, y + (self.cell_h - tile.height) // 2), tile)
                else:
                    draw.text((x, y), ch, font=self.bold if bold else self.regular, fill=color)
        self._draw_caption(image, draw, caption)
        return image

    def _draw_caption(self, image: Image.Image, draw: ImageDraw.ImageDraw, caption: str) -> None:
        top = self.height - self.caption_h
        draw.rectangle([0, top, self.width, self.height], fill=(13, 14, 18))
        draw.line([0, top, self.width, top], fill=(48, 52, 62))
        if not caption:
            return
        accent_w = 4
        draw.rectangle([self.padding, top + self.cell_h // 2 - 1,
                        self.padding + accent_w, top + self.cell_h + 6], fill=(98, 152, 226))
        draw.text((self.padding + accent_w + 12, top + self.cell_h // 2 - 2), caption,
                  font=self.bold, fill=(205, 212, 224))

    def card(self, title: str, lines) -> Image.Image:
        """Заставка: заголовок и несколько строк по центру кадра."""
        image = Image.new("RGB", (self.width, self.height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        big = ImageFont.truetype(MENLO, 30, index=1)
        y = self.height // 2 - (len(lines) + 3) * self.cell_h // 2
        draw.text(((self.width - draw.textlength(title, font=big)) // 2, y), title,
                  font=big, fill=(255, 255, 255))
        y += int(self.cell_h * 2.4)
        for line in lines:
            draw.text(((self.width - draw.textlength(line, font=self.regular)) // 2, y),
                      line, font=self.regular, fill=(150, 156, 170))
            y += int(self.cell_h * 1.25)
        return image
