import re
from itertools import product, repeat

from jinja2 import Template

from src.utils.fn import convert_annotated_str_to_nltk_str

template = r"""\documentclass[preview]{standalone}
\usepackage{tikz}
\usepackage{tikz-qtree}
\usetikzlibrary{calc}
\begin{document}
\begin{tikzpicture}
\begin{scope}[local bounding box=scope1]
{{tree1}}
\end{scope}
\begin{scope}[shift={($(scope1.east)+(1cm,0)$)}]
{{tree2}}
\end{scope}
\begin{scope}[dashed]
{{alignment}}
\end{scope}
\end{tikzpicture}
\end{document}
"""


def make_latex_code(src, tgt, alignment_raw):
    alignment_raw = parse_alignment(alignment_raw)
    ch2 = ColorHelper2()
    alignment = [
        f"\draw[{ch2.get((s1, e1))}] (s{s1}to{e1})--(t{s2}to{e2});"
        for (s1, e1), (s2, e2) in alignment_raw
        if e1 - s1 > 1 and e2 - s2 > 1
    ]
    ch = ColorHelper()
    src = convert_annotated_str_to_nltk_str(src, prefix="s")
    src = add_label_to_qtree_code(src, attrs=ch)
    ch.copy_alignment(
        [(f"s{s1}to{e1}", f"t{s2}to{e2}") for (s1, e1), (s2, e2) in alignment_raw]
    )
    tgt = convert_annotated_str_to_nltk_str(tgt, prefix="t")
    tgt = add_label_to_qtree_code(tgt, attrs=ch)
    t = Template(template)
    return t.render(tree1=src, tree2=tgt, alignment="\n".join(alignment))


def parse_alignment(alignment):
    result = []
    pattern = re.compile(".*\((\d+), (\d+)\)")
    for line in alignment.split("\n"):
        line = line.strip().rstrip(" COPY")
        if len(line) == 0:
            continue
        s, t = line.split(" - ")
        s = re.match(pattern, s)
        t = re.match(pattern, t)
        result.append((tuple(map(int, s.groups())), tuple(map(int, t.groups()))))
    return result


def add_label_to_qtree_code(qtree_code, attrs=None):
    splitted = qtree_code.pformat_latex_qtree().split()
    for i, item in enumerate(splitted):
        if item.startswith("[."):
            attr = attrs.get(item[2:]) if attrs is not None else None
            attr = "" if attr is None else f"[{attr}]"
            splitted[i] = f"[.\\node({item[2:]}){attr}{{{item[2:]}}};"
    return " ".join(splitted)


class ColorHelper:
    color_list = ["green", "red", "blue", "yellow", "olive", "teal"]
    bg_color_list = ["lightgray", "white", "blue!20", "red!20", "orange!20"]

    def __init__(self) -> None:
        self.it = product(self.color_list, self.bg_color_list, [0, 40])
        self.mapping = {}

    def get(self, name):
        if name in self.mapping:
            return self.mapping[name]
        try:
            fc, bc, rate = next(self.it)
        except StopIteration:
            self.it = repeat(("black", "white", 0))
            fc, bc, rate = next(self.it)
        attr = f"black!{rate}!{fc},fill={bc}"
        self.mapping[name] = attr
        return attr

    def copy_alignment(self, alignment):
        for src, tgt in alignment:
            self.mapping[tgt] = self.mapping.get(src)


class ColorHelper2:
    color_list = ["green", "red", "blue", "orange"]

    def __init__(self) -> None:
        self.it = iter(self.color_list)
        self.mapping = {}

    def get(self, name):
        if name in self.mapping:
            return self.mapping[name]
        try:
            c = next(self.it)
        except StopIteration:
            self.it = iter(self.color_list)
            c = next(self.it)
        attr = c
        self.mapping[name] = attr
        return attr


if __name__ == "__main__":
    src = "((the luxury auto maker) (last year) (sold (1,214 cars) (in (the u.s.))))"
    tgt = "(((((1,214 cars) ((last year) were)) sold) by) (the (luxury (auto (maker (in (the u.s.)))))))"
    alignment = """
  cars (8, 9) - were (4, 5)
  sold (6, 7) - sold (5, 6) COPY
  the (0, 1) - by (6, 7)
  the (0, 1) - the (7, 8) COPY
  luxury (1, 2) - luxury (8, 9) COPY
  auto (2, 3) - auto (9, 10) COPY
  maker (3, 4) - maker (10, 11) COPY
  1,214 cars (7, 9) - 1,214 cars (0, 2) COPY
  last year (4, 6) - last year (2, 4) COPY
  last year (4, 6) - last year were (2, 5)
  in the u.s. (9, 12) - in the u.s. (11, 14) COPY
  maker last year sold 1,214 cars in the u.s. (3, 12) - maker in the u.s. (10, 14)
  1,214 cars (7, 9) - 1,214 cars last year were (0, 5)
  auto maker last year sold 1,214 cars in the u.s. (2, 12) - auto maker in the u.s. (9, 14)
  sold 1,214 cars in the u.s. (6, 12) - 1,214 cars last year were sold (0, 6)
  luxury auto maker last year sold 1,214 cars in the u.s. (1, 12) - luxury auto maker in the u.s. (8, 14)
  1,214 cars (7, 9) - 1,214 cars last year were sold by (0, 7)
  the luxury auto maker last year sold 1,214 cars in the u.s. (0, 12) - the luxury auto maker in the u.s. (7, 14)
  the luxury auto maker last year sold 1,214 cars in the u.s. (0, 12) - 1,214 cars last year were sold by the luxury auto maker in the u.s. (0, 14)
    """
    with open("tikz_fig.tex", "w") as f:
        f.write(make_latex_code(src, tgt, alignment))
