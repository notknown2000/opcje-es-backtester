"""Optional live integration check: python tests/browser_spx_monitor.py."""
import json
import os
from pathlib import Path

from PIL import Image, ImageStat
from playwright.sync_api import sync_playwright


def main():
    output = Path("results/spx_live_monitor")
    output.mkdir(parents=True, exist_ok=True)
    url = os.environ.get("SPX_MONITOR_URL", "http://spx-monitor:8502")
    checks = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        for name, viewport in [("desktop", {"width": 1440, "height": 1050}),
                               ("mobile", {"width": 390, "height": 844})]:
            page = browser.new_page(viewport=viewport, device_scale_factor=1)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url, wait_until="domcontentloaded")
            page.get_by_role("heading", name="SPX / Tail Monitor").wait_for(timeout=60000)
            page.locator('[data-testid="stDataFrame"]').first.wait_for(timeout=90000)
            assert page.locator('[data-testid="stException"]').count() == 0
            assert not errors, errors
            page.screenshot(path=str(output / f"{name}.png"), full_page=True)
            chart = page.locator('[data-testid="stPlotlyChart"]').first
            chart.screenshot(path=str(output / f"{name}-heatmap.png"))
            stats = ImageStat.Stat(Image.open(output / f"{name}-heatmap.png").convert("RGB"))
            assert max(stats.stddev) > 10, "Blank heatmap"
            pixels = Image.open(output / f"{name}-heatmap.png").convert("RGB")
            colored = sum(1 for r, g, b in pixels.getdata() if (r > g * 1.4 and r > b * 1.2)
                          or (g > r * 1.3 and g > b * 1.1))
            assert colored > pixels.width * pixels.height * .03, "No colored heatmap cells"
            assert "3-3.5" in chart.locator(".xaxislayer-above").text_content(), "Sigma axis parsed as dates"
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2"), "Page overflow"
            page.get_by_role("tab", name="Lancuch i IV").click()
            page.wait_for_timeout(1000)
            visible = page.locator('[data-testid="stPlotlyChart"]:visible')
            assert visible.count() == 2
            for i in range(2):
                assert visible.nth(i).locator("svg .scatterlayer .trace").count() >= 1, "Blank curve"
            page.screenshot(path=str(output / f"{name}-curves.png"), full_page=True)
            page.get_by_role("tab", name="Jakosc danych").click()
            assert page.get_by_text("Kontrakty w feedzie:", exact=False).is_visible()
            checks.append(dict(viewport=name, charts=3, nonblank=True, overflow=False, js_errors=errors))
            page.close()
        browser.close()
    (output / "browser_validation.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
