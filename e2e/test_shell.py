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


@pytest.mark.parametrize("width", PHONE_WIDTHS)
def test_collapsed_menu_opens_and_its_links_are_tappable(page, base_url, width):
    open_landing(page, base_url, width, height=844)
    links = page.locator("nav.navbar .nav-link")

    assert not visible(links), "nav links should start collapsed on a phone"

    page.locator("nav.navbar summary").click()
    shown = visible(links)
    assert len(shown) == 5, f"expected 5 links in the open menu, saw {len(shown)}"
    for link in shown:
        box = link.bounding_box()
        assert box["height"] >= MIN_TAP_TARGET_PX, f"link is {box['height']:.0f}px tall"
        assert box["x"] >= 0 and box["x"] + box["width"] <= width, "link leaves the viewport"

    page.keyboard.press("Escape")
    assert not visible(links), "Escape should close the menu"


@pytest.mark.parametrize("width", TABLET_UP_WIDTHS)
def test_nav_links_are_always_shown_on_wide_screens(page, base_url, width):
    open_landing(page, base_url, width)

    assert len(visible(page.locator("nav.navbar .nav-link"))) == 5
    assert not page.locator("nav.navbar summary").is_visible(), "menu button on a wide screen"


def test_language_selector_in_the_nav_still_switches_the_nav(page, base_url):
    open_landing(page, base_url, 1024)

    page.select_option("#nav-lang", "fr")

    assert page.locator("nav.navbar .nav-link").first.inner_text() == "Accueil"


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


@pytest.mark.parametrize("width", [360, 1024])
def test_dialog_controls_stay_inside_the_form_column(page, base_url, width):
    """The form column scrolls (overflow-y), which clips anything wider than it."""
    open_landing(page, base_url, width, height=844)
    page.evaluate("openUploadModal()")
    # Open the optional sections too, so their controls are measured.
    page.evaluate(
        "() => { const t = document.getElementById('rcd-tested-checkbox');"
        " t.checked = true; t.dispatchEvent(new Event('change', {bubbles: true})); }"
    )

    sticking_out = page.evaluate(
        """() => {
          const col = document.querySelector('.upload-modal-content > .container');
          const edge = col.getBoundingClientRect();
          const bad = [];
          col.querySelectorAll('*').forEach(el => {
            const b = el.getBoundingClientRect();
            if (b.width === 0 || b.height === 0) return;
            const over = Math.max(b.right - edge.right, edge.left - b.left);
            if (over > 0.5) bad.push(`${el.tagName.toLowerCase()}#${el.id} +${Math.round(over)}px`);
          });
          return bad;
        }"""
    )

    assert not sticking_out, f"wider than the form column at {width}px: {sticking_out}"


@pytest.mark.parametrize("width", [390, 1024])
def test_dialog_close_button_works(page, base_url, width):
    """click() fails if another element covers the button, which is the point."""
    open_landing(page, base_url, width, height=844)
    page.evaluate("openUploadModal()")
    overlay = page.locator("#upload-modal")
    assert "open" in overlay.get_attribute("class")

    page.locator(".modal-close-btn").click(timeout=3000)

    assert "open" not in overlay.get_attribute("class")


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


# --- Responsive upload form checks (roadmap slice S3 / T21) ------------------


@pytest.mark.parametrize("width", [320, 390, 1440])
def test_form_interactive_controls_are_at_least_44x44(page, base_url, width):
    """Every interactive control in the upload form (using label area for
    checkboxes and radios) must be at least 44x44 px at 320, 390 and 1440."""
    open_landing(page, base_url, width, height=844)
    page.evaluate("openUploadModal()")
    # Expand optional RCD options and the privacy panel so all controls are visible.
    page.evaluate(
        "() => { const t = document.getElementById('rcd-tested-checkbox');"
        " t.checked = true; t.dispatchEvent(new Event('change', {bubbles: true})); }"
    )
    page.locator("#privacy-policy-link").click()
    assert page.locator("#privacy-policy-panel").is_visible()

    too_small = page.evaluate(
        """(minPx) => {
          const form = document.getElementById('uploadForm');
          const controls = form.querySelectorAll('a, button, select, input:not([type=hidden])');
          const bad = [];
          controls.forEach(el => {
            let target = el;
            if (el.tagName === 'INPUT' && (el.type === 'checkbox' || el.type === 'radio')) {
              const label = el.closest('label') || (el.id && form.querySelector(`label[for="${el.id}"]`));
              if (label) target = label;
            }
            const rect = target.getBoundingClientRect();
            if (rect.width === 0 && rect.height === 0) return;
            if (rect.width < minPx - 0.5 || rect.height < minPx - 0.5) {
              const name = (el.id || el.textContent || el.tagName).trim().slice(0, 30);
              bad.push(`${name}: ${Math.round(rect.width)}x${Math.round(rect.height)}px`);
            }
          });
          return bad;
        }""",
        MIN_TAP_TARGET_PX,
    )

    assert not too_small, f"form controls under {MIN_TAP_TARGET_PX}x{MIN_TAP_TARGET_PX}px at {width}px: {too_small}"


@pytest.mark.parametrize("lang", ["es", "fr"])
def test_open_dialog_has_no_horizontal_scroll_at_320(page, base_url, lang):
    """No horizontal scroll inside the open upload dialog at 320 px in ES and FR."""
    open_landing(page, base_url, 320, height=844, lang=lang)
    page.evaluate("openUploadModal()")
    page.evaluate(
        "() => { const t = document.getElementById('rcd-tested-checkbox');"
        " t.checked = true; t.dispatchEvent(new Event('change', {bubbles: true})); }"
    )
    page.locator("#privacy-policy-link").click()
    assert page.locator("#privacy-policy-panel").is_visible()

    overflows = page.evaluate(
        """() => {
          const col = document.querySelector('.upload-modal-content > .container');
          const modal = document.querySelector('.upload-modal-content');
          return {
            page: document.documentElement.scrollWidth - document.documentElement.clientWidth,
            modal: modal.scrollWidth - modal.clientWidth,
            col: col.scrollWidth - col.clientWidth,
          };
        }"""
    )

    assert overflows["page"] <= 0, f"[{lang}] page overflows by {overflows['page']}px at 320px"
    assert overflows["modal"] <= 0, f"[{lang}] modal overflows by {overflows['modal']}px at 320px"
    assert overflows["col"] <= 0, f"[{lang}] form column overflows by {overflows['col']}px at 320px"


def test_upload_requires_mandatory_consent_not_training_only(page, base_url):
    """Upload stays disabled until the required consent is ticked; training alone does not enable it."""
    open_landing(page, base_url, 390, height=844)
    page.evaluate("openUploadModal()")

    consent = page.locator("#consent-checkbox")
    training = page.locator("#training-consent-checkbox")
    submit = page.locator("#submit-btn")

    assert not consent.is_checked(), "required consent must start unticked"
    assert not training.is_checked(), "optional training consent must never be pre-ticked"
    assert submit.is_disabled(), "submit must start disabled"

    # Tick optional training consent alone via its clickable label.
    page.locator('label[for="training-consent-checkbox"]').click()
    assert training.is_checked()
    assert not consent.is_checked()
    assert submit.is_disabled(), "training consent alone must not enable Upload"

    # Tick required consent: now Upload becomes enabled.
    consent.check()
    assert submit.is_enabled(), "required consent should enable Upload"

    # Untick required consent while training stays ticked: Upload returns to disabled.
    consent.uncheck()
    assert training.is_checked()
    assert submit.is_disabled(), "unchecking required consent must disable Upload again"


def test_privacy_panel_opens_and_closes_at_320(page, base_url):
    """The privacy panel can be opened and closed at 320 px without overflowing or toggling consent."""
    open_landing(page, base_url, 320, height=844)
    page.evaluate("openUploadModal()")

    panel = page.locator("#privacy-policy-panel")
    link = page.locator("#privacy-policy-link")
    consent = page.locator("#consent-checkbox")

    assert not panel.is_visible()

    link.click(timeout=3000)
    assert panel.is_visible(), "privacy panel should open on link click at 320px"
    assert not consent.is_checked(), "opening privacy panel must not toggle the consent checkbox"
    box = panel.bounding_box()
    assert box["x"] >= 0 and box["x"] + box["width"] <= 320.5, f"privacy panel leaves 320px viewport: {box}"

    link.click(timeout=3000)
    assert not panel.is_visible(), "privacy panel should close on second link click at 320px"


def test_photo_buttons_stack_on_phones_and_row_from_640(page, base_url):
    """Photo buttons stack full-width on phones and sit in a row from 640 px."""
    # Phones: all three stacked full-width
    for width in [320, 390]:
        open_landing(page, base_url, width, height=844)
        page.evaluate("openUploadModal()")
        col_box = page.locator(".upload-modal-content > .container").bounding_box()
        vf = page.locator("#viewfinder-btn").bounding_box()
        cam = page.locator("#start-camera-btn").bounding_box()
        browse = page.locator("#browse-file-btn").bounding_box()

        assert vf["y"] + vf["height"] <= cam["y"] + 0.5, f"buttons not stacked at {width}px"
        assert cam["y"] + cam["height"] <= browse["y"] + 0.5, f"buttons not stacked at {width}px"
        for btn_box in (vf, cam, browse):
            assert btn_box["width"] == pytest.approx(col_box["width"], abs=2), (
                f"photo button not full-width at {width}px"
            )

    # 640px and desktop: secondary photo buttons sit side-by-side in a row
    for width in [640, 1440]:
        open_landing(page, base_url, width, height=900)
        page.evaluate("openUploadModal()")
        cam = page.locator("#start-camera-btn").bounding_box()
        browse = page.locator("#browse-file-btn").bounding_box()

        assert cam["y"] == pytest.approx(browse["y"], abs=2), f"secondary photo buttons not in a row at {width}px"
        assert cam["x"] + cam["width"] <= browse["x"] + 0.5, f"photo buttons overlap at {width}px"


@pytest.mark.parametrize("width", [320, 390])
def test_submit_button_stays_reachable_on_phones(page, base_url, width):
    """On phones, sticky footer keeps Submit reachable inside the viewport without scrolling."""
    open_landing(page, base_url, width, height=844)
    page.evaluate("openUploadModal()")

    box = page.locator("#submit-btn").bounding_box()
    assert box is not None
    assert box["y"] >= 0 and box["y"] + box["height"] <= 844.5, (
        f"submit button is outside the {width}x844 phone viewport: {box}"
    )


def test_upload_form_has_no_static_inline_styles(page, base_url):
    """Inline styles inside the upload form were moved into CSS classes."""
    open_landing(page, base_url, 390, height=844)
    page.evaluate("openUploadModal()")

    styled = page.evaluate(
        """() => Array.from(document.querySelectorAll('#uploadForm [style]'))
          .map(el => `${el.tagName.toLowerCase()}#${el.id}`)"""
    )

    assert not styled, f"unexpected inline style attributes in #uploadForm: {styled}"
