
import random


class Breaker:
    def __init__(self, cls, width):
        self.cls = cls
        self.width = width # in modules

class Rail:
    def __init__(self, max_modules=12):
        self.max_modules = max_modules
        self.components = []
        self._total_width = 0 

    def add_component(self, breaker):
        if (self._total_width + breaker.width) <= self.max_modules:
            self._total_width += breaker.width
            self.components.append(breaker)
            return True 
        else:
            return False
    def current_width(self):
        return self._total_width

class Panel:
    def __init__(self, rails_count=2, rail_height=200):
        self.rail_height = rail_height
        self.rails = [Rail() for _ in range (rails_count)]
    def get_rail_y_center(self, rail_index):
        half_height = self.rail_height / 2
        return (rail_index * self.rail_height) + half_height


class PanelFactory:
    """
    Generates procedurally randomized Panel objects following Spanish REBT grammar.

    Valid Spanish panel structure (per REBT):
        Rail 0:  [MAINBREAKER or OVERSURGE] → [RCD or RCD_SI] → [MCBs...]
        Rail 1+: [RCD or RCD_SI] → [MCBs...]

    Real-world panels frequently have MULTIPLE RCD-protected groups on a single rail,
    e.g.: [RCD] → [MCBs] → [RCD_SI] → [MCBs]
    The `add_rcd_group` behavior models this common pattern so the HMM learns that
    P(MCB → RCD) is non-trivially probable in field installations.

    Chaos behaviors model real-world non-compliant installs:
        - missing_mainbreaker : No IGA — installer relied on meter breaker (common)
        - missing_rcd         : No diferencial on a rail (rare, dangerous)
        - use_oversurge       : IGA replaced by IGA+DPS (modern REBT-recommended)
        - use_rcd_si          : Superinmunizado instead of standard RCD (growing trend)
        - wide_mcb            : 2-module MCBs mixed into layout
        - add_other           : Timer/contactor present on rail
        - add_rcd_group       : A second RCD group mid-rail (very common in real panels)

    Parameters
    ----------
    chaos_factor : float [0.0–1.0]
        Scales all chaos probabilities. 0.0 = perfectly valid panel.
        1.0 = full base probabilities apply to every decision point.
    rail_height : int
        Pixel height of each rail slot (passed through to Panel).
    """

    # Weighted rail counts reflecting Spanish residential reality:
    #   1 rail → small flat  |  2 rails → most households
    #   3 rails → larger/electrified home  |  4 rails → garage/business
    RAIL_COUNTS   = [1, 2, 3, 4]
    RAIL_WEIGHTS  = [0.15, 0.55, 0.20, 0.10]

    # Per-behavior base probabilities at chaos_factor=1.0
    CHAOS_PROBS = {
        "missing_mainbreaker": 0.20,  # common — meter has breaker, skipped
        "missing_rcd":         0.05,  # rare but dangerous
        "use_oversurge":       0.15,  # newer REBT-compliant installs
        "use_rcd_si":          0.30,  # growing recommendation
        "wide_mcb":            0.25,  # 2-module MCB mixed in
        "add_other":           0.10,  # timer/contactor present
        "add_rcd_group":       0.45,  # second RCD group mid-rail (very common in field)
    }

    # MAINBREAKER module widths: older panels larger, modern ones slimmer
    MAINBREAKER_WIDTHS  = [1, 2, 3, 4]
    MAINBREAKER_WEIGHTS = [0.10, 0.60, 0.20, 0.10]

    def __init__(self, chaos_factor=0.0, rail_height=200, boost_minority=False):
        self.chaos_factor = chaos_factor
        self.rail_height  = rail_height

        # When boost_minority is True, override probabilities to ensure
        # underrepresented classes (OVERSURGE, RCD_SI, OTHER) appear more
        # frequently in the generated dataset.
        if boost_minority:
            self.CHAOS_PROBS = dict(self.CHAOS_PROBS)  # instance copy
            self.CHAOS_PROBS["use_oversurge"] = 0.40   # was 0.15
            self.CHAOS_PROBS["use_rcd_si"]    = 0.50   # was 0.30
            self.CHAOS_PROBS["add_other"]     = 0.25   # was 0.10
            self.CHAOS_PROBS["add_rcd_group"] = 0.60   # was 0.45

    # ── Core generator ────────────────────────────────────────────────────────
    def generate(self):
        """
        Generate one randomized Panel.

        Returns
        -------
        Panel
            A Panel with rails filled according to REBT grammar ±chaos.
        """
        rail_count = random.choices(self.RAIL_COUNTS, weights=self.RAIL_WEIGHTS)[0]
        panel = Panel(rails_count=rail_count, rail_height=self.rail_height)

        for idx, rail in enumerate(panel.rails):
            if idx == 0:
                self._fill_main_rail(rail)
            else:
                self._fill_secondary_rail(rail)

        return panel

    # ── Rail fillers ──────────────────────────────────────────────────────────
    def _fill_main_rail(self, rail):
        """Rail 0: [MAINBREAKER/OVERSURGE?] → [RCD?] → MCBs (→ [RCD?] → MCBs)..."""
        if not self._chaos("missing_mainbreaker"):
            if self._chaos("use_oversurge"):
                w = random.choices([2, 3, 4], weights=[0.4, 0.4, 0.2])[0]
                rail.add_component(Breaker(cls="OVERSURGE", width=w))
            else:
                w = random.choices(
                    self.MAINBREAKER_WIDTHS, weights=self.MAINBREAKER_WEIGHTS
                )[0]
                rail.add_component(Breaker(cls="MAINBREAKER", width=w))

        has_initial_rcd = not self._chaos("missing_rcd")
        if has_initial_rcd:
            rcd = "RCD_SI" if self._chaos("use_rcd_si") else "RCD"
            rail.add_component(Breaker(cls=rcd, width=2))

        self._fill_mcbs(rail, has_initial_rcd=has_initial_rcd)

    def _fill_secondary_rail(self, rail):
        """Rail 1+: [RCD?] → MCBs (→ [RCD?] → MCBs)..."""
        has_initial_rcd = not self._chaos("missing_rcd")
        if has_initial_rcd:
            rcd = "RCD_SI" if self._chaos("use_rcd_si") else "RCD"
            rail.add_component(Breaker(cls=rcd, width=2))

        self._fill_mcbs(rail, has_initial_rcd=has_initial_rcd)

    def _fill_mcbs(self, rail, has_initial_rcd=True):
        """
        Fill remaining module space with MCBs, occasional OTHER devices, and
        optionally additional mid-rail RCD groups.

        A second RCD group (e.g. [MCBs] → [RCD_SI] → [MCBs]) is injected once
        per call when `add_rcd_group` chaos fires AND at least 2 MCBs have been
        placed since the last RCD, AND enough space remains (RCD=2 + 1 MCB = 3).
        This teaches the HMM that P(MCB → RCD) is non-trivially probable.
        """
        # Track MCBs placed since the last RCD so we don't inject an RCD immediately
        # after another one — a minimum gap of 2 MCBs is required.
        mcbs_since_last_rcd = 0 if has_initial_rcd else 2  # no initial RCD → already at gap
        extra_rcd_placed = False  # only one extra group per rail to keep panels realistic

        while True:
            remaining = rail.max_modules - rail.current_width()
            if remaining <= 0:
                break

            # Occasional OTHER device (timer/contactor)
            if self._chaos("add_other") and remaining >= 1:
                w = min(random.choice([1, 2]), remaining)
                if not rail.add_component(Breaker(cls="OTHER", width=w)):
                    break
                continue

            # Mid-rail RCD group injection:
            #   - Only once per rail (extra_rcd_placed guard)
            #   - Requires ≥2 MCBs since last RCD (realistic gap)
            #   - Requires room for RCD (2 modules) + at least 1 MCB (1 module)
            if (
                not extra_rcd_placed
                and mcbs_since_last_rcd >= 2
                and remaining >= 3
                and self._chaos("add_rcd_group")
            ):
                rcd = "RCD_SI" if self._chaos("use_rcd_si") else "RCD"
                if rail.add_component(Breaker(cls=rcd, width=2)):
                    mcbs_since_last_rcd = 0
                    extra_rcd_placed = True
                    continue

            # MCB — 1 or 2 modules
            w = 2 if (self._chaos("wide_mcb") and remaining >= 2) else 1
            if not rail.add_component(Breaker(cls="MCB", width=w)):
                break
            mcbs_since_last_rcd += 1

    # ── Chaos helper ──────────────────────────────────────────────────────────
    def _chaos(self, behavior):
        """
        Return True if this chaos behavior triggers this call.

        Effective probability = base_prob × chaos_factor.
        At chaos_factor=0.0, always returns False (valid panel).
        """
        prob = self.CHAOS_PROBS.get(behavior, 0.0) * self.chaos_factor
        return random.random() < prob