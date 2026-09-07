// Optional: NODE_PATH=/path/to/cached/node_modules node tests/frontend/workspace_race.browser.js
// Every request is fulfilled locally. No production server or refresh job is contacted.
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "../../web");
const origin = "http://workspace-race.test";
const files = new Map([
  ["/", ["index.html", "text/html"]],
  ["/app.js", ["app.js", "text/javascript"]],
  ["/styles.css", ["styles.css", "text/css"]],
  ["/favicon.svg", ["favicon.svg", "image/svg+xml"]],
  ["/core/api-client.js", ["core/api-client.js", "text/javascript"]],
  ["/core/formatters.js", ["core/formatters.js", "text/javascript"]],
]);
const dates = ["2026-09-01", "2026-09-02"];
const calendar = {
  latest_signal_date: "", days: dates.map((date) => ({
    date, status: "ready", label: "ready", is_open: true, disabled: false,
  })),
};

async function run() {
  const browser = await chromium.launch({ headless: true, ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
    ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}) });
  try {
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const context = await browser.newContext({ viewport });
      const page = await context.newPage();
      const pending = [];
      let calendarRequests = 0;
      const requestWaiters = [];
      const waitForPending = (count) => pending.length >= count ? Promise.resolve() : new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error(`Expected ${count} workspace requests; got ${pending.length}`)), 10000);
        requestWaiters.push({ count, resolve: () => { clearTimeout(timeout); resolve(); } });
      });
      const errors = [];
      const consoleErrors = [];
      page.on("pageerror", (error) => errors.push(error.message));
      page.on("console", (message) => {
        if (["error", "warning"].includes(message.type())) consoleErrors.push(message.text());
      });
      await page.route("**/*", async (route) => {
        const url = new URL(route.request().url());
        assert.equal(url.origin, origin, "test must never access a real service");
        const json = (data, status = 200) => route.fulfill({
          status, contentType: "application/json", headers: { "X-Quant-Generation": "test-generation-1" }, body: JSON.stringify(data),
        });
        if (url.pathname.startsWith("/api/long/")) {
          assert.equal(route.request().headers()["x-quant-generation"], "test-generation-1");
          pending.push({ url, json });
          for (const waiter of requestWaiters) {
            if (pending.length >= waiter.count) waiter.resolve();
          }
          return;
        }
        if (url.pathname === "/api/selector/calendar") {
          calendarRequests += 1;
          return json(calendar);
        }
        if (url.pathname === "/api/selector/refresh-latest/status") return json({ status: "idle" });
        const file = files.get(url.pathname);
        assert.ok(file, `unexpected request ${url}`);
        let body = readFileSync(path.join(root, file[0]), "utf8");
        if (url.pathname === "/app.js") body += "\nglobalThis.__race = { state };";
        return route.fulfill({ contentType: file[1], body });
      });
      await page.goto(`${origin}/#long`);
      await page.waitForFunction(() => globalThis.__race?.state.calendar);
      assert.equal(page.url(), `${origin}/#long`);
      assert.equal(await page.title(), "策略工作台");
      assert.ok((await page.locator("#longPage").innerText()).length > 100);
      assert.equal(await page.locator("vite-error-overlay, nextjs-portal").count(), 0);
      await waitForPending(1);
      assert.equal(pending.length, 1);

      // Fix the calendar fixture's month independently of the machine's current date.
      await page.evaluate(() => { globalThis.__race.state.calendarMonth = "2026-09"; });
      await page.locator("#calendarToggle").click();
      await page.locator(`[data-date="${dates[0]}"]`).click();
      await page.locator("#calendarToggle").click();
      await page.locator(`[data-date="${dates[1]}"]`).click();
      await waitForPending(3);
      const selected = pending.find((request) => request.url.searchParams.get("signal_date") === dates[1]);
      assert.ok(selected, "new date must dispatch its own request");
      await selected.json({ signal_date: dates[1], stocks: [], marker: "current" });
      await page.waitForFunction(() => globalThis.__race.state.longPayload?.marker === "current");
      const expected = await page.locator("#longPoolMeta").innerText();
      assert.ok(expected.includes(dates[1]));
      const oldDate = pending.find((request) => request.url.searchParams.get("signal_date") === dates[0]);
      await oldDate.json({ detail: "obsolete-date-error" }, 500);
      await pending[0].json({ signal_date: "2020-01-01", stocks: [], marker: "obsolete" });
      await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      assert.equal(await page.locator("#longPoolMeta").innerText(), expected);
      assert.ok(!(await page.locator(".refresh-status").innerText()).includes("obsolete-date-error"));

      await page.locator('.long-variant-button[data-long-variant="blood_chip"]:visible').first().click();
      await page.locator('.long-variant-button[data-long-variant="tea"]:visible').first().click();
      await waitForPending(5);
      const newest = pending.at(-1);
      const obsoleteVariant = pending.at(-2);
      await obsoleteVariant.json({ marker: "obsolete-blood-chip" });
      await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      assert.equal(await page.evaluate(() => globalThis.__race.state.longLoading), true);
      await newest.json({ signal_date: dates[1], stocks: [], marker: "newest-tea" });
      await page.waitForFunction(() => globalThis.__race.state.longPayload?.marker === "newest-tea");
      assert.equal(await page.locator("#longPoolMeta").innerText(), expected);
      // A persistent conflict with the same calendar pin must settle, not loop.
      const calendarsBeforeConflict = calendarRequests;
      await page.locator('.long-variant-button[data-long-variant="blood_chip"]:visible').first().click();
      await waitForPending(6);
      await pending[5].json({ detail: "persistent conflict" }, 409);
      await waitForPending(7);
      await page.waitForFunction(() => globalThis.__race.state.workspaceGenerationReloadPromise === null);
      await pending[6].json({ detail: "persistent conflict" }, 409);
      await page.waitForFunction(() => !globalThis.__race.state.longLoading
        && globalThis.__race.state.longError.includes("停止自动重载"));
      assert.equal(calendarRequests, calendarsBeforeConflict + 1);
      assert.equal(pending.length, 7);
      assert.equal(await page.evaluate(() => globalThis.__race.state.longPayload), null);
      assert.ok((await page.locator("#longPage").innerText()).includes("停止自动重载"));
      assert.deepEqual(errors, []);
      assert.equal(consoleErrors.length, 3);
      assert.match(consoleErrors[0], /500/); // Deliberately rejected obsolete HTTP response.
      assert.ok(consoleErrors.slice(1).every((message) => /409/.test(message)));
      const screenshot = `/tmp/quant-workspace-race-${viewport.width}.png`;
      await page.screenshot({ path: screenshot, fullPage: false });
      console.log(JSON.stringify({ viewport, url: page.url(), checks: "PASS", screenshot, expectedHttpErrors: 3 }));
      await context.close();
    }
  } finally {
    await browser.close();
  }
}

run().catch((error) => { console.error(error); process.exitCode = 1; });
