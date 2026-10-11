"""Layout-shell checks for the landing page (roadmap slice S2).

These tests are the definition of "done" for S2. Run them against the page
before changing any CSS: most fail, and each failure message says what to fix.

Markup contract (the only two hooks the tests need that the page does not have
yet -- add them as part of S2):

  data-testid="primary-action"  the control that opens the audit dialog
  data-testid="pro-tools"       the container holding analyzer / unifilar / blog
"""
import pytest

# Widths from the S0 baseline: small phone, common phone, large phone, tablet,
# small laptop, desktop.
WIDTHS = [320, 360, 390, 768, 1024, 1440]
PHONE_WIDTHS = [320, 360, 390]
TABLET_UP_WIDTHS = [768, 1024, 1440]
HEIGHT = 900
MIN_TAP_TARGET_PX = 44
INTERACTIVE = "a, button, select, summary, input:not([type=hidden])"


def open_landing(page, base_url, width, height=HEIGHT, lang=None):
    if lang:
        # The page reads its language from localStorage on load.
        page.add_init_script(f"localStorage.setItem('breakerLang', '{lang}')")
    page.set_viewport_size({"width": width, "height": height})
    page.goto(base_url + "/")
    page.wait_for_load_state("load")


def visible(locator):
    return [el for el in locator.all() if el.is_visible()]


def hero_font_px(page):
    return float(page.locator("h1").first.evaluate("el => parseFloat(getComputedStyle(el).fontSize)"))


def test_landing_page_loads(page, base_url):
    open_landing(page, base_url, 1024)
    assert "PanelSafe" in page.title()


@pytest.mark.parametrize("width", WIDTHS)
def test_page_does_not_scroll_horizontally(page, base_url, width):
    open_landing(page, base_url, width)

    overflow = page.evaluate(
        "document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )

    assert overflow <= 0, f"page is {overflow}px wider than the {width}px viewport"


@pytest.mark.parametrize("lang", ["es", "fr"])
@pytest.mark.parametrize("width", [320, 360])
def test_longer_languages_still_fit_on_small_phones(page, base_url, width, lang):
    open_landing(page, base_url, width, lang=lang)

    page_overflow = page.evaluate(
        "document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    nav_overflow = page.locator("nav.navbar").evaluate("el => el.scrollWidth - el.clientWidth")

    assert page_overflow <= 0, f"[{lang}] page overflows by {page_overflow}px at {width}px"
    assert nav_overflow <= 0, f"[{lang}] nav overflows by {nav_overflow}px at {width}px"


@pytest.mark.parametrize("width", WIDTHS)
def test_nav_fits_inside_the_viewport(page, base_url, width):
    open_landing(page, base_url, width)
    nav = page.locator("nav.navbar")

    own_overflow = nav.evaluate("el => el.scrollWidth - el.clientWidth")
    assert own_overflow <= 0, f"nav content is {own_overflow}px wider than the nav at {width}px"

    # Visible controls must also end inside the viewport, not just inside the nav box.
    off_screen = []
    for control in visible(nav.locator(INTERACTIVE)):
        box = control.bounding_box()
        if box and box["x"] + box["width"] > width + 0.5:
            off_screen.append(control.evaluate("el => (el.textContent || el.id || el.tagName).trim().slice(0, 30)"))
    assert not off_screen, f"controls extend past the {width}px viewport: {off_screen}"


@pytest.mark.parametrize("width", WIDTHS)
def test_interactive_controls_are_tappable(page, base_url, width):
    open_landing(page, base_url, width)

    too_small = []
    for control in visible(page.locator(INTERACTIVE)):
        box = control.bounding_box()
        if box and box["height"] < MIN_TAP_TARGET_PX:
            label = control.evaluate("el => (el.getAttribute('aria-label') || el.textContent || el.tagName).trim().slice(0, 30)")
            too_small.append(f"{label!r} {box['height']:.0f}px")

    assert not too_small, f"controls under {MIN_TAP_TARGET_PX}px tall at {width}px: {too_small}"


def test_hero_title_scales_with_width(page, base_url):
    sizes = {}
    for width in WIDTHS:
        open_landing(page, base_url, width)
        sizes[width] = hero_font_px(page)

    assert sizes[320] < sizes[1440], f"hero is a fixed size: {sizes}"
    ordered = [sizes[w] for w in WIDTHS]
    assert ordered == sorted(ordered), f"hero shrinks as the viewport grows: {sizes}"


@pytest.mark.parametrize("width,height", [(390, 844), (1440, 900)])
def test_consumer_audit_is_the_primary_action(page, base_url, width, height):
    open_landing(page, base_url, width, height=height)
    primary = page.locator('[data-testid="primary-action"]')
    tools = page.locator('[data-testid="pro-tools"]')

    assert primary.count() == 1, 'add data-testid="primary-action" to the audit control'
    assert tools.count() == 1, 'add data-testid="pro-tools" to the professional tools container'

    primary_box = primary.bounding_box()
    tools_box = tools.bounding_box()
    assert primary_box["y"] < tools_box["y"], "professional tools sit above the audit control"
    assert primary_box["y"] + primary_box["height"] <= height, (
        f"audit control is below the fold at {width}x{height}"
    )


@pytest.mark.parametrize("width", PHONE_WIDTHS)
def test_dialog_is_full_screen_on_phones(page, base_url, width):
    open_landing(page, base_url, width, height=844)
    page.evaluate("openUploadModal()")
    dialog = page.locator(".upload-modal-content")

    box = dialog.bounding_box()

    assert box["width"] == pytest.approx(width, abs=1), f"dialog is {box['width']:.0f}px wide"
    assert box["height"] == pytest.approx(844, abs=1), f"dialog is {box['height']:.0f}px tall"


@pytest.mark.parametrize("width", TABLET_UP_WIDTHS)
def test_dialog_is_a_centered_card_from_tablet_up(page, base_url, width):
    open_landing(page, base_url, width)
    page.evaluate("openUploadModal()")
    dialog = page.locator(".upload-modal-content")

    box = dialog.bounding_box()
    left_gap = box["x"]
    right_gap = width - (box["x"] + box["width"])

    assert box["width"] < width, "dialog still fills the whole viewport"
    assert left_gap == pytest.approx(right_gap, abs=1), f"off-centre: {left_gap:.0f}px vs {right_gap:.0f}px"
