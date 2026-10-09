"""Chasing 37 mbb/g: what was tried on 2026-10-08/09 to reproduce Deep CFR on Flop Hold'em Poker.
Manim Community scenes; render each with `manim -qh --fps 30 fhp_story.py <Scene>` and concatenate."""
from manim import *
import numpy as np

config.background_color = "#0b0d12"
PAPER, OURS, K25, GOODC, BADC = YELLOW, BLUE, TEAL, GREEN, RED
DIM = GREY_B


class Base(Scene):
    _cap = None

    def say(self, text, wait=3.2, scale=0.74):
        t = Tex(text, tex_environment="center").scale(scale).to_edge(DOWN, buff=0.4)
        anims = [FadeIn(t, shift=UP * 0.15)]
        if self._cap is not None:
            anims.insert(0, FadeOut(self._cap, shift=UP * 0.15))
        self.play(*anims, run_time=0.6)
        self._cap = t
        self.wait(wait)

    def header(self, text):
        h = Tex(text).scale(0.9).to_edge(UP, buff=0.35).set_color(GREY_A)
        line = Line(LEFT * 6.5, RIGHT * 6.5, stroke_width=1, color=GREY_D).next_to(h, DOWN, buff=0.15)
        self.play(FadeIn(h, shift=DOWN * 0.2), Create(line), run_time=0.7)
        return VGroup(h, line)

    def finish(self):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.7)
        self._cap = None


def log_axes(xticks, yticks, xlabel, ylabel, x_len=8.0, y_len=4.3):
    """Axes that are linear in log10; ticks = [(value, label), ...]."""
    lx = [np.log10(v) for v, _ in xticks]
    ly = [np.log10(v) for v, _ in yticks]
    ax = Axes(x_range=[lx[0] - 0.06, lx[-1] + 0.06, 10], y_range=[ly[0] - 0.05, ly[-1] + 0.05, 10], x_length=x_len, y_length=y_len,
              tips=False, axis_config={"include_ticks": False, "stroke_width": 2, "color": GREY_B})
    deco = VGroup()
    for (v, lab), l in zip(xticks, lx):
        p = ax.c2p(l, ly[0] - 0.05)
        deco.add(Line(p + UP * 0.07, p + DOWN * 0.07, stroke_width=2, color=GREY_B))
        deco.add(Tex(lab).scale(0.5).set_color(GREY_A).next_to(p, DOWN, buff=0.16))
    for (v, lab), l in zip(yticks, ly):
        p = ax.c2p(lx[0] - 0.06, l)
        deco.add(Line(p, ax.c2p(lx[-1] + 0.06, l), stroke_width=1, color=GREY_E))
        deco.add(Tex(lab).scale(0.5).set_color(GREY_A).next_to(p, LEFT, buff=0.14))
    deco.add(Tex(xlabel).scale(0.55).set_color(GREY_A).next_to(ax, DOWN, buff=0.5))
    deco.add(Tex(ylabel).scale(0.55).set_color(GREY_A).rotate(PI / 2).next_to(ax, LEFT, buff=0.7))
    return ax, deco


def curve(ax, pts, color, width=5):
    points = [ax.c2p(np.log10(x), np.log10(y)) for x, y in pts]
    line = VMobject(color=color, stroke_width=width).set_points_as_corners(points)
    dots = VGroup(*[Dot(p, color=color, radius=0.065) for p in points])
    return line, dots


ITER_TICKS = [(50, "50"), (100, "100"), (200, "200"), (300, "300"), (450, "450")]
EXPL_TICKS = [(30, "30"), (50, "50"), (100, "100"), (200, "200"), (400, "400")]
PAPER_PTS = [(50, 154), (100, 70), (200, 51), (300, 40), (450, 40)]
OURS_PTS = [(50, 367), (100, 202), (200, 113), (300, 95), (450, 78)]
K25_PTS = [(50, 178), (100, 117), (200, 91), (300, 69), (450, 75)]


class S01Title(Base):
    def construct(self):
        title = Tex(r"Chasing 37 mbb/g").scale(1.7)
        sub = Tex(r"Reproducing Deep CFR on Flop Hold'em Poker").scale(0.95).set_color(GREY_A).next_to(title, DOWN, buff=0.4)
        sub2 = Tex(r"what we tried on October 8--9").scale(0.7).set_color(GREY_B).next_to(sub, DOWN, buff=0.3)
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(sub2), run_time=0.6)
        self.wait(1.5)
        self.play(VGroup(title, sub, sub2).animate.scale(0.6).to_edge(UP, buff=0.5), run_time=0.9)
        # what the number means: a strategy against a perfect opponent
        me = RoundedRectangle(width=3.2, height=1.3, corner_radius=0.2, color=OURS, stroke_width=3)
        me_t = Tex(r"our strategy").scale(0.75).move_to(me)
        opp = RoundedRectangle(width=3.2, height=1.3, corner_radius=0.2, color=BADC, stroke_width=3)
        opp_t = Tex(r"a perfect opponent").scale(0.75).move_to(opp)
        VGroup(me, me_t).move_to(LEFT * 3.4 + DOWN * 0.3)
        VGroup(opp, opp_t).move_to(RIGHT * 3.4 + DOWN * 0.3)
        arrow = Arrow(me.get_right(), opp.get_left(), buff=0.15, color=YELLOW, stroke_width=5)
        chips = Tex(r"chips lost per hand").scale(0.6).set_color(YELLOW).next_to(arrow, UP, buff=0.12)
        self.play(FadeIn(VGroup(me, me_t), shift=RIGHT * 0.3), FadeIn(VGroup(opp, opp_t), shift=LEFT * 0.3), run_time=0.8)
        self.play(GrowArrow(arrow), FadeIn(chips), run_time=0.8)
        self.say(r"\textbf{Exploitability}: what a perfect opponent wins against the strategy.\\Zero is a Nash equilibrium. The paper reports 37--40 mbb/g.", wait=4.5)
        self.finish()


class S02Gap(Base):
    def construct(self):
        self.header(r"The gap")
        ax, deco = log_axes(ITER_TICKS, EXPL_TICKS, r"CFR iteration", r"exploitability (mbb/g)")
        VGroup(ax, deco).move_to(UP * 0.25)
        self.play(Create(ax), FadeIn(deco), run_time=1.0)
        pl, pd = curve(ax, PAPER_PTS, PAPER)
        plab = Tex(r"the paper").scale(0.6).set_color(PAPER).next_to(pd[-1], RIGHT, buff=0.15)
        self.play(Create(pl), run_time=1.8)
        self.play(FadeIn(pd), FadeIn(plab), run_time=0.5)
        self.say(r"Deep CFR, 2019: neural networks learn poker regrets from sampled hands.\\On Flop Hold'em it reaches about 40 after 300 iterations.", wait=4.0)
        ol, od = curve(ax, OURS_PTS, OURS)
        olab = Tex(r"ours").scale(0.6).set_color(OURS).next_to(od[-1], RIGHT, buff=0.15)
        self.play(Create(ol), run_time=1.8)
        self.play(FadeIn(od), FadeIn(olab), run_time=0.5)
        self.say(r"Ours: the same game, the same network, the hyperparameters as printed.", wait=3.0)
        a, b = ax.c2p(np.log10(450), np.log10(40)), ax.c2p(np.log10(450), np.log10(78))
        br = BraceBetweenPoints(a, b, direction=LEFT, color=WHITE)
        x2 = Tex(r"$\approx 2\times$").scale(0.8).next_to(br, LEFT, buff=0.1)
        self.play(GrowFromCenter(br), FadeIn(x2), run_time=0.8)
        self.say(r"78 against 40. Twice as exploitable --- and it stays that way. Why?", wait=3.5)
        self.finish()


class S03Data(Base):
    def construct(self):
        self.header(r"Suspect 1: do we see less of the game?")
        base = DOWN * 1.35
        vals = [("the paper", 1.04, PAPER), ("ours", 0.41, OURS), ("DREAM (later paper,\\\\public code)", 0.41, GREY_B)]
        bars, labels, nums = VGroup(), VGroup(), VGroup()
        for i, (name, v, col) in enumerate(vals):
            bar = Rectangle(width=1.3, height=3.0 * v, fill_color=col, fill_opacity=0.9, stroke_width=0)
            bar.move_to(base + RIGHT * (i - 1) * 3.0, aligned_edge=DOWN)
            bars.add(bar)
            labels.add(Tex(name, tex_environment="center").scale(0.55).set_color(col).next_to(bar, DOWN, buff=0.15))
            nums.add(Tex(rf"{v:.2f} M").scale(0.65).next_to(bar, UP, buff=0.12))
        ytitle = Tex(r"game-tree nodes touched per iteration").scale(0.6).set_color(GREY_A).move_to(UP * 2.55)
        self.play(FadeIn(ytitle), run_time=0.5)
        self.play(GrowFromEdge(bars[0], DOWN), FadeIn(labels[0]), FadeIn(nums[0]), run_time=0.9)
        self.play(GrowFromEdge(bars[1], DOWN), FadeIn(labels[1]), FadeIn(nums[1]), run_time=0.9)
        self.say(r"The paper's ``10,000 traversals'' touch 2.5 times more nodes than ours.", wait=3.2)
        self.play(GrowFromEdge(bars[2], DOWN), FadeIn(labels[2]), FadeIn(nums[2]), run_time=0.9)
        self.say(r"A later paper with two of the same authors, and public code, counts like we do.\\So we give our run the paper's amount of data: 25,000 traversals.", wait=4.2)
        self.play(*[FadeOut(m) for m in (bars, labels, nums, ytitle)], run_time=0.6)
        ax, deco = log_axes(ITER_TICKS, EXPL_TICKS, r"CFR iteration", r"exploitability (mbb/g)")
        VGroup(ax, deco).move_to(UP * 0.25)
        pl, pd = curve(ax, PAPER_PTS, PAPER)
        ol, od = curve(ax, OURS_PTS, OURS)
        plab = Tex(r"the paper").scale(0.55).set_color(PAPER).next_to(pd[-1], RIGHT, buff=0.15)
        olab = Tex(r"10,000").scale(0.55).set_color(OURS).next_to(od[-1], UR, buff=0.1)
        self.play(FadeIn(ax), FadeIn(deco), FadeIn(pl), FadeIn(pd), FadeIn(ol), FadeIn(od), FadeIn(plab), FadeIn(olab), run_time=0.8)
        kl, kd = curve(ax, K25_PTS, K25)
        klab = Tex(r"25,000").scale(0.55).set_color(K25).next_to(kd[-1], DR, buff=0.1)
        self.play(Create(kl), run_time=2.0)
        self.play(FadeIn(kd), FadeIn(klab), run_time=0.5)
        self.play(Circumscribe(VGroup(kd[0], pd[0]), color=WHITE, buff=0.12), run_time=1.2)
        self.say(r"Early on it nearly matches: 178 against 154 at iteration 50.", wait=2.8)
        self.play(Circumscribe(VGroup(kd[-1], od[-1]), color=BADC, buff=0.12), run_time=1.2)
        self.say(r"But it ends where the small run ends: 75 and 78.\\More data buys a faster start, not a lower floor.", wait=4.2)
        self.finish()


class S04PaperSweep(Base):
    def construct(self):
        self.header(r"The paper against itself")
        steps = ["1,000", "2,000", "4,000", "8,000", "16,000", "32,000"]
        vals = [110, 80, 65, 43, 37, 35]
        base = DOWN * 1.35 + LEFT * 4.6
        h = lambda v: 3.3 * v / 110
        bars, labs, nums = VGroup(), VGroup(), VGroup()
        for i, (s, v) in enumerate(zip(steps, vals)):
            bar = Rectangle(width=1.0, height=h(v), fill_color=PAPER if i != 2 else ORANGE, fill_opacity=0.85, stroke_width=0)
            bar.move_to(base + RIGHT * i * 1.55, aligned_edge=DOWN)
            bars.add(bar)
            labs.add(Tex(s).scale(0.55).set_color(GREY_A).next_to(bar, DOWN, buff=0.15))
            nums.add(Tex(rf"\textbf{{{v}}}").scale(0.62).set_color(BLACK).move_to(bar.get_top() + DOWN * 0.25))
        xl = Tex(r"SGD steps per network fit (the paper's own sweep, final exploitability)").scale(0.55).set_color(GREY_A).move_to(DOWN * 2.15 + LEFT * 0.7)
        self.play(FadeIn(xl), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.15), run_time=2.0)
        self.play(FadeIn(labs), FadeIn(nums), run_time=0.6)
        self.say(r"The paper also varies how long each network is trained.", wait=2.5)
        box = SurroundingRectangle(VGroup(bars[2], labs[2], nums[2]), color=WHITE, buff=0.12)
        txt = Tex(r"the setting in its text").scale(0.55).next_to(box, UP, buff=0.5).align_to(box, LEFT).shift(RIGHT * 0.1)
        self.play(Create(box), FadeIn(txt), run_time=0.8)
        y40 = base[1] + h(40)
        dash = DashedLine([-6.2, y40, 0], [4.3, y40, 0], color=GOODC, stroke_width=3)
        dlab = Tex(r"its headline: 40").scale(0.55).set_color(GOODC).next_to(dash, RIGHT, buff=0.15)
        self.play(Create(dash), FadeIn(dlab), run_time=0.9)
        self.say(r"4,000 steps --- what the text says was used --- end at 65.\\The headline number sits with the 16,000 and 32,000-step runs.", wait=4.6)
        y75 = base[1] + h(76)
        ours = DashedLine([-6.2, y75, 0], [4.3, y75, 0], color=OURS, stroke_width=3)
        olab = Tex(r"ours at 4,000 steps: 75--78").scale(0.55).set_color(OURS).next_to(ours, UP, buff=0.1).shift(RIGHT * 3.4)
        self.play(Create(ours), FadeIn(olab), run_time=0.9)
        self.say(r"Our 4,000-step runs land next to the paper's own 4,000-step bar.\\So: does the training budget move \emph{our} floor too?", wait=4.6)
        self.finish()


class S05Branches(Base):
    def construct(self):
        self.header(r"Experiment: branch one run, change only the fit")
        ax, deco = log_axes([(100, "100"), (200, "200"), (325, "325"), (450, "450")], [(50, "50"), (70, "70"), (100, "100"), (130, "130")],
                            r"CFR iteration", r"exploitability (mbb/g)", x_len=8.5, y_len=4.2)
        VGroup(ax, deco).move_to(UP * 0.25 + LEFT * 0.6)
        self.play(Create(ax), FadeIn(deco), run_time=0.9)
        P = lambda x, y: ax.c2p(np.log10(x), np.log10(y))
        trunk = VMobject(color=K25, stroke_width=5).set_points_as_corners([P(100, 117), P(200, 91), P(300, 69), P(325, 70)])
        self.play(Create(trunk), run_time=1.6)
        fork = Dot(P(325, 70), color=WHITE, radius=0.08)
        self.play(FadeIn(fork, scale=2), run_time=0.4)
        self.say(r"Take the 25,000-traversal run at iteration 325 and continue it four ways.", wait=3.0)
        ends = [("1,000 steps", 90.3, BADC), ("4,000 steps", 74.8, K25), ("8,000 steps", 60.8, GREEN_B), ("16,000 steps", 58.1, GOODC)]
        for name, v, col in ends:
            br = Line(P(325, 70), P(450, v), color=col, stroke_width=5)
            d = Dot(P(450, v), color=col, radius=0.07)
            lab = Tex(rf"{name}: \textbf{{{v:.0f}}}").scale(0.58).set_color(col).next_to(d, RIGHT, buff=0.15)
            lab.shift(UP * (0.16 if v == 60.8 else -0.16 if v == 58.1 else 0))
            self.play(Create(br), run_time=0.9)
            self.play(FadeIn(d), FadeIn(lab), run_time=0.4)
        self.say(r"Fewer steps: clearly worse. More steps: better.\\The fit budget does set the floor.", wait=4.0)
        self.say(r"But the gain flattens near 58, and 16,000 steps from the very start\\are no better at iteration 300 (68 against 69). Not the whole story.", wait=5.0)
        self.finish()


class S06Where(Base):
    def construct(self):
        self.header(r"Where does the strategy lose?")
        self.say(r"Let the perfect opponent deviate at \emph{one} kind of decision only,\\and play our own strategy everywhere else.", wait=4.0)
        rows = [("small blind's first action", 15.1, OURS), ("first action on the flop", 12.9, K25), ("big blind facing a raise", 11.1, OURS),
                ("facing a flop bet", 10.5, K25), ("facing a bet after checking the flop", 9.5, K25), ("after a flop check", 7.6, K25),
                ("re-raised pots (each kind)", 1.5, GREY_B)]
        top = UP * 2.2
        group = VGroup()
        bars = []
        for i, (name, v, col) in enumerate(rows):
            y = top + DOWN * i * 0.62
            lab = Tex(name).scale(0.56).set_color(GREY_A)
            lab.move_to(y + LEFT * 1.2, aligned_edge=RIGHT)
            bar = Rectangle(width=0.36 * v, height=0.4, fill_color=col, fill_opacity=0.9, stroke_width=0)
            bar.move_to(y + LEFT * 0.95, aligned_edge=LEFT)
            num = Tex(f"{v:.1f}" if v > 2 else "0--3").scale(0.56).next_to(bar, RIGHT, buff=0.12)
            group.add(lab, bar, num)
            bars.append((lab, bar, num))
        legend = VGroup(Square(0.22, fill_color=OURS, fill_opacity=0.9, stroke_width=0), Tex("before the flop").scale(0.5),
                        Square(0.22, fill_color=K25, fill_opacity=0.9, stroke_width=0), Tex("on the flop").scale(0.5)).arrange(RIGHT, buff=0.2)
        legend.move_to(RIGHT * 3.9 + DOWN * 1.85)
        self.play(FadeIn(legend), run_time=0.4)
        for lab, bar, num in bars:
            self.play(FadeIn(lab, shift=RIGHT * 0.2), GrowFromEdge(bar, LEFT), FadeIn(num), run_time=0.55)
        unit = Tex("mbb/g lost").scale(0.5).set_color(GREY_B).next_to(bars[0][2], RIGHT, buff=0.2)
        self.play(FadeIn(unit), run_time=0.3)
        self.say(r"The losses are not in rare, exotic pots.", wait=2.5)
        self.play(Circumscribe(VGroup(*bars[0]), color=YELLOW, buff=0.08), run_time=1.2)
        self.say(r"They sit at the most common decisions --- the very first action of every hand\\loses the most. Exactly where the network has the \emph{most} data.", wait=5.0)
        self.finish()


class S07Precision(Base):
    def construct(self):
        self.header(r"Zoom in: the first decision of the hand")
        ax = Axes(x_range=[-18, 26, 5], y_range=[0, 0.215, 1], x_length=10.5, y_length=3.4, tips=False,
                  axis_config={"include_ticks": False, "stroke_width": 2, "color": GREY_B}, y_axis_config={"stroke_opacity": 0})
        ax.move_to(UP * 0.1)
        ticks = VGroup()
        for x in (-10, 0, 10, 20):
            p = ax.c2p(x, 0)
            ticks.add(Line(p + UP * 0.07, p + DOWN * 0.07, stroke_width=2, color=GREY_B), Tex(str(x)).scale(0.5).set_color(GREY_A).next_to(p, DOWN, buff=0.15))
        xl = Tex(r"regret of an action (chips)").scale(0.55).set_color(GREY_A).next_to(ax, DOWN, buff=0.45)
        self.play(Create(ax), FadeIn(ticks), FadeIn(xl), run_time=0.9)
        sig = ValueTracker(2.4)
        g = lambda mu: (lambda x: np.exp(-0.5 * ((x - mu) / sig.get_value()) ** 2) / (sig.get_value() * np.sqrt(2 * np.pi)))
        ca = always_redraw(lambda: ax.plot(g(0.0), x_range=[-18, 26, 0.1], color=OURS, stroke_width=4))
        cb = always_redraw(lambda: ax.plot(g(7.0), x_range=[-18, 26, 0.1], color=ORANGE, stroke_width=4))
        la = Tex("call").scale(0.6).set_color(OURS).move_to(ax.c2p(-4.8, 0.12))
        lb = Tex("raise").scale(0.6).set_color(ORANGE).move_to(ax.c2p(12.2, 0.12))
        self.play(Create(ca), Create(cb), FadeIn(la), FadeIn(lb), run_time=1.4)
        gap = DoubleArrow(ax.c2p(0, 0.185), ax.c2p(7, 0.185), buff=0, color=WHITE, stroke_width=3, tip_length=0.15)
        gl = Tex("7 chips").scale(0.5).next_to(gap, UP, buff=0.06)
        self.play(GrowFromCenter(gap), FadeIn(gl), run_time=0.7)
        self.say(r"Two good actions are typically 7 chips apart in regret.\\The samples pin each one down to about 2.4 chips: easy to tell apart.", wait=5.0)
        self.play(FadeOut(gap), FadeOut(gl), la.animate.move_to(ax.c2p(-7.5, 0.075)), lb.animate.move_to(ax.c2p(14.5, 0.075)), sig.animate.set_value(7.0), run_time=2.5)
        self.say(r"The network's answer is 6--9 chips off --- three times less precise than its data.\\Two retrainings on the same data disagree by as much.", wait=5.0)
        pct = Tex(r"different top action than the data supports: \textbf{16\,\%} of pre-flop situations").scale(0.62).set_color(YELLOW).move_to(UP * 2.4)
        self.play(FadeIn(pct, shift=DOWN * 0.2), run_time=0.7)
        self.say(r"So the strategy picks the wrong action where it matters most ---\\not for lack of data, but for lack of \emph{precision}.", wait=4.6)
        self.finish()


class S08RuledOut(Base):
    def construct(self):
        self.header(r"What it is not")
        items = [
            (r"$\times$", BADC, r"too little data", r"25,000 traversals: same floor (75 vs 78)"),
            (r"$\times$", BADC, r"memory too small", r"6 M vs 24 M samples: 97 vs 91"),
            (r"$\times$", BADC, r"suit symmetry", r"canonical suits fit no better"),
            (r"$\times$", BADC, r"random noise of one run", r"mixing two runs: 64 vs 68"),
            (r"$\times$", BADC, r"network details", r"no card table, 2$\times$ width: under 0.5\,\%"),
            (r"$\sim$", YELLOW, r"weight averaging", r"pre-flop part 30 $\to$ 20, total 75 $\to$ 73"),
            (r"$\checkmark$", GOODC, r"longer fits, late", r"75 $\to$ 61 $\to$ 58"),
        ]
        rows = VGroup()
        for mark, col, name, detail in items:
            m = Tex(mark).scale(0.85).set_color(col)
            n = Tex(name).scale(0.68)
            d = Tex(detail).scale(0.58).set_color(GREY_B)
            rows.add(VGroup(m, n, d))
        for i, r in enumerate(rows):
            y = UP * (2.3 - i * 0.68)
            r[0].move_to(y + LEFT * 5.2)
            r[1].move_to(y + LEFT * 4.7, aligned_edge=LEFT)
            r[2].move_to(y + RIGHT * 0.4, aligned_edge=LEFT)
        for i, r in enumerate(rows):
            self.play(FadeIn(r[1], shift=RIGHT * 0.2), run_time=0.35)
            self.play(FadeIn(r[0], scale=1.8), FadeIn(r[2]), run_time=0.45)
            self.wait(1.1 if i < 5 else 1.6)
        self.say(r"One by one, the usual suspects drop out.\\What is left is the precision of the regret network at the common decisions.", wait=5.0)
        self.finish()


class S09Others(Base):
    def construct(self):
        self.header(r"Has anyone else matched the paper?")
        xt = [(5e7, r"$5\cdot10^7$"), (1e8, r"$10^8$"), (3e8, r"$3\cdot10^8$"), (1e9, r"$10^9$")]
        yt = [(20, "20"), (30, "30"), (50, "50"), (100, "100"), (200, "200")]
        ax, deco = log_axes(xt, yt, r"game-tree nodes touched", r"exploitability (mbb/g)", x_len=8.6, y_len=4.2)
        VGroup(ax, deco).move_to(UP * 0.25)
        self.play(Create(ax), FadeIn(deco), run_time=0.9)
        P = lambda x, y: ax.c2p(np.log10(x), np.log10(y))
        pl, pd = curve(ax, [(5.2e7, 154), (1.04e8, 70), (2.1e8, 51), (3.1e8, 40), (4.7e8, 40)], PAPER)
        plab = Tex("the paper").scale(0.55).set_color(PAPER).next_to(pd[2], DL, buff=0.1)
        self.play(Create(pl), FadeIn(pd), FadeIn(plab), run_time=1.4)
        ol, od = curve(ax, [(8.2e7, 113), (1.23e8, 95), (1.85e8, 78)], OURS)
        olab = Tex("ours").scale(0.55).set_color(OURS).next_to(od[0], UP, buff=0.12)
        self.play(Create(ol), FadeIn(od), FadeIn(olab), run_time=1.2)
        liu1, liu2 = Dot(P(6e8, 47), color=PURPLE_A, radius=0.09), Dot(P(1.3e9, 73), color=PURPLE_A, radius=0.09)
        llab = Tex(r"independent\\re-implementation\\(Liu et al.): 47", tex_environment="center").scale(0.5).set_color(PURPLE_A).next_to(liu1, DOWN, buff=0.2).shift(RIGHT * 1.1)
        self.play(FadeIn(liu1, scale=2), FadeIn(llab), run_time=0.8)
        self.say(r"Another group re-implemented Deep CFR with the paper's settings:\\47 --- but only after twice as many nodes.", wait=4.5)
        up = Arrow(liu1.get_center(), liu2.get_center(), buff=0.1, color=PURPLE_A, stroke_width=4)
        l2 = Tex("then 73").scale(0.5).set_color(PURPLE_A).next_to(liu2, UP, buff=0.1)
        self.play(GrowArrow(up), FadeIn(liu2), FadeIn(l2), run_time=0.9)
        ext = DashedLine(P(1.85e8, 78), P(6e8, 43.3), color=OURS, stroke_width=4)
        q = Tex("?").scale(0.8).set_color(OURS).next_to(P(6e8, 43.3), DL, buff=0.08)
        self.play(Create(ext), FadeIn(q), run_time=1.2)
        self.say(r"Our run is still falling. If it keeps its pace it meets their number ---\\at twice the paper's budget. We found no public run with 37 at $3\cdot10^8$ nodes.", wait=5.5)
        self.finish()


class S10Now(Base):
    def construct(self):
        self.header(r"Running right now")
        def card(title, lines, col):
            box = RoundedRectangle(width=5.6, height=3.3, corner_radius=0.25, color=col, stroke_width=3)
            t = Tex(title).scale(0.72).set_color(col).move_to(box.get_top() + DOWN * 0.5)
            body = VGroup(*[Tex(l).scale(0.56).set_color(GREY_A) for l in lines]).arrange(DOWN, buff=0.22).next_to(t, DOWN, buff=0.35)
            return VGroup(box, t, body)
        a = card(r"Precision recipe", [r"25,000 traversals", r"8,000 steps per fit", r"+ weight averaging", r"from scratch"], GOODC).move_to(LEFT * 3.3 + UP * 0.35)
        b = card(r"The paper's settings, longer", [r"10,000 traversals, 4,000 steps", r"450 $\to$ 1,500 iterations", r"$6\cdot10^8$ nodes touched", r"40 M-sample memories"], OURS).move_to(RIGHT * 3.3 + UP * 0.35)
        self.play(FadeIn(a, shift=UP * 0.3), run_time=0.8)
        self.say(r"One run attacks the precision directly.", wait=2.6)
        self.play(FadeIn(b, shift=UP * 0.3), run_time=0.8)
        self.say(r"The other gives the printed recipe the budget at which\\the independent re-implementation reached 47.", wait=4.2)
        self.say(r"Where we stand: 58--78 against the paper's 37--40. What is missing is precision ---\\and the paper's own figures disagree on how much training it took.", wait=6.0)
        self.finish()
        end = Tex(r"to be continued\,\dots").scale(1.0).set_color(GREY_A)
        self.play(FadeIn(end), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(end), run_time=0.8)
