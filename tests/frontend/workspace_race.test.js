const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const root = path.resolve(__dirname, "../..");
const source = readFileSync(path.join(root, "web/app.js"), "utf8");
const tick = () => new Promise((resolve) => setImmediate(resolve));

function harness(page = "long", { generationTransport = false } = {}) {
  const requests = [];
  const elements = new Map();
  const timers = new Map();
  let timerId = 0;
  function element(selector) {
    if (!elements.has(selector)) {
      elements.set(selector, {
        value: "", dataset: {}, textContent: "", innerHTML: "", disabled: false,
        children: [], listeners: {},
        classList: { add() {}, remove() {}, toggle() {} },
        addEventListener(type, handler) { this.listeners[type] = handler; },
        querySelectorAll() { return this.children; },
        setAttribute() {}, removeAttribute() {},
      });
    }
    return elements.get(selector);
  }
  const variant = element("variant");
  variant.dataset.longVariant = "tea";
  const context = vm.createContext({
    URLSearchParams, Headers, AbortController, console,
    fetch: (url, options) => new Promise((resolve, reject) => {
      requests.push({ url, options, resolve, reject });
    }),
    document: {
      querySelector: element,
      querySelectorAll: (selector) => selector === ".long-variant-button" ? [variant] : [],
      addEventListener() {},
    },
    window: {
      location: { hash: `#${page}` }, addEventListener() {},
      setTimeout(fn) { timers.set(++timerId, fn); return timerId; },
      clearTimeout(id) { timers.delete(id); },
    },
    localStorage: { getItem() { return null; } },
    createApiClient: () => (url, options) => new Promise((resolve, reject) => {
      requests.push({ url, options, resolve, reject });
    }),
    errors: [], renders: [], notices: [],
  });
  // Evaluate the production functions and event registrations, not copied loaders.
  // Only imports and network-starting bootstrap are replaced by test adapters.
  const app = source.slice(0, source.indexOf("const initialDateInput ="))
    .replace(/^import[\s\S]*?from "\.\/core\/[^"\n]+";\n/gm, "");
  const formatters = readFileSync(path.join(root, "web/core/formatters.js"), "utf8")
    .replace(/\bexport /g, "");
  vm.runInContext(`${formatters}\n${app}`, context, { filename: "web/app.js" });
  const renderers = [
    "renderShortPage", "renderLongStockPool", "renderLongOverview", "renderLongStrategies",
    "renderChanModelPage", "renderConvertibleBondPage", "renderConvertibleBondAllotments",
    "renderBydPage", "renderSimilarPatternsPage", "renderOperationPlans", "renderDateStatus",
    "renderPageShell",
  ];
  vm.runInContext(`
    ${generationTransport ? "" : "fetchJson = createApiClient(API_BASE);"}
    ${renderers.map((name) => `${name} = () => renders.push("${name}");`).join("\n")}
    showError = (error) => errors.push(error.message);
    showWatchlistToast = (message) => notices.push(message);
    maybeShowBydTradeToast = (payload) => notices.push(payload.marker);
  `, context);
  const run = (code) => vm.runInContext(code, context);
  const state = run("state");
  state.activePage = page;
  state.signalDate = "2026-09-01";
  function snapshot() {
    return JSON.stringify({
      state: Object.fromEntries(Object.entries(state).map(([key, value]) => [
        key, value instanceof Promise ? "promise" : value,
      ])),
      renders: context.renders, errors: context.errors, notices: context.notices,
      dateInput: element("#signalDateInput").value,
    });
  }
  return { context, run, state, requests, element, timers, snapshot };
}

const pages = [
  { scope: "short", loader: "loadSelector", loading: "loading", payload: "payload",
    data: (marker) => ({ marker, signal_date: "2026-09-01", stocks: [{ symbol: marker }] }) },
  { scope: "long", loader: "loadLongStockPool", loading: "longLoading", payload: "longPayload",
    data: (marker) => ({ marker }) },
  { scope: "chan", loader: "loadChanModelStrategy", loading: "chanLoading", payload: "chanPayload",
    data: (marker) => ({ marker, signal_date: "2026-09-01", candidates: [{ symbol: marker }] }) },
  { scope: "cb", loader: "loadConvertibleBondPlan", loading: "cbLoading", payload: "cbPayload",
    data: (marker) => ({ marker, strategy_plans: [{ strategy: { key: "grid" }, candidates: [{ ts_code: marker }] }] }) },
  { scope: "cbAllotment", loader: "loadConvertibleBondAllotments", loading: "cbAllotmentLoading", payload: "cbAllotmentPayload",
    data: (marker) => ({ marker }) },
  { scope: "byd", loader: "loadBydMinuteStrategy", loading: "bydLoading", payload: "bydPayload",
    data: (marker) => ({ marker }) },
  { scope: "plans", loader: "loadOperationPlans", loading: "operationPlansLoading", payload: "operationPlans",
    data: (marker) => ({ plans: [marker] }) },
];

for (const spec of pages) {
  for (const failure of [false, true]) {
    test(`${spec.scope}: late ${failure ? "failure" : "success"} cannot replace a newer result or detail selection`, async () => {
      const h = harness(spec.scope);
      const old = h.run(`${spec.loader}().catch(showError)`);
      h.run('setWorkspaceSignalDate("2026-09-02")');
      const current = h.run(`${spec.loader}().catch(showError)`);
      h.requests[1].resolve(spec.data("current"));
      await current;
      const snapshot = h.snapshot();
      if (failure) h.requests[0].reject(new Error("obsolete failure"));
      else h.requests[0].resolve(spec.data("obsolete"));
      await old;
      assert.equal(h.snapshot(), snapshot);
      assert.equal(h.state[spec.loading], false);
      assert.deepEqual(h.context.errors, []);
    });
  }

  test(`${spec.scope}: obsolete finally leaves the newest same-query spinner running`, async () => {
    const h = harness(spec.scope);
    const old = h.run(`${spec.loader}().catch(showError)`);
    const first = h.state.workspaceRequests.get(spec.scope);
    const current = h.run(`${spec.loader}().catch(showError)`);
    const second = h.state.workspaceRequests.get(spec.scope);
    assert.ok(Object.isFrozen(first));
    assert.ok(second.token > first.token);
    assert.equal(first.queryKey, h.requests[0].url);
    assert.equal(first.queryKey, second.queryKey);
    const snapshot = h.snapshot();
    h.requests[0].reject(new Error("old"));
    await old;
    assert.equal(h.snapshot(), snapshot);
    assert.equal(h.state[spec.loading], true);
    h.requests[1].resolve(spec.data("current"));
    await current;
    assert.equal(h.state[spec.loading], false);
  });

  test(`${spec.scope}: date A-B-A without a replacement load still invalidates the old request`, async () => {
    const h = harness(spec.scope);
    const old = h.run(`${spec.loader}().catch(showError)`);
    h.run('setWorkspaceSignalDate("2026-09-02"); setWorkspaceSignalDate("2026-09-01")');
    const snapshot = h.snapshot();
    h.requests[0].resolve(spec.data("obsolete"));
    await old;
    assert.equal(h.snapshot(), snapshot);
    assert.equal(h.state[spec.loading], false);
  });

  test(`${spec.scope}: a current failure still propagates and clears loading`, async () => {
    const h = harness(spec.scope);
    const current = h.run(`${spec.loader}()`);
    h.requests[0].reject(new Error("current failure"));
    await assert.rejects(current, /current failure/);
    assert.equal(h.state[spec.loading], false);
    assert.ok(h.context.renders.length >= 2);
  });
}

test("long: actual variant controls reject an A-B-A response race", async () => {
  const h = harness();
  const old = h.run("loadLongStockPool().catch(showError)");
  const button = h.element("variant");
  button.dataset.longVariant = "blood_chip";
  button.listeners.click();
  button.dataset.longVariant = "tea";
  button.listeners.click();
  assert.match(h.requests[1].url, /\/long\/blood-chip\?/);
  const snapshot = h.snapshot();
  h.requests[0].resolve({ marker: "old tea" });
  await old;
  h.requests[1].reject(new Error("old blood chip"));
  await tick();
  assert.equal(h.snapshot(), snapshot);
  h.requests[2].resolve({ marker: "new tea" });
  await tick();
  assert.equal(h.state.longPayload.marker, "new tea");
});

test("selector: actual debounced filter clicks invalidate before a new fetch, including A-B-A", async () => {
  const h = harness("short");
  const old = h.run("loadSelector().catch(showError)");
  const filter = h.element("filter");
  filter.dataset.strategy = "B1";
  h.element("#strategyFilters").children = [filter];
  h.run("renderStrategyFilters()");
  filter.listeners.click();
  filter.listeners.click();
  assert.equal(h.requests.length, 1);
  assert.equal(h.timers.size, 1);
  const snapshot = h.snapshot();
  h.requests[0].reject(new Error("obsolete during debounce"));
  await old;
  assert.equal(h.snapshot(), snapshot);
  [...h.timers.values()][0]();
  assert.equal(h.requests.length, 2);
  h.requests[1].resolve(pages[0].data("latest"));
  await tick();
  assert.equal(h.state.payload.marker, "latest");
});

test("BYD: input changes invalidate without submitting a replacement request", async () => {
  const h = harness("byd");
  const old = h.run("loadBydMinuteStrategy().catch(showError)");
  const input = h.element("#bydSharesInput");
  input.value = "5000";
  input.listeners.input();
  input.value = "";
  input.listeners.input();
  const snapshot = h.snapshot();
  h.requests[0].resolve({ marker: "old holding toast" });
  await old;
  assert.equal(h.snapshot(), snapshot);
});

test("navigation invalidates a pending page even when the destination is cached", async () => {
  const h = harness();
  const old = h.run("loadLongStockPool().catch(showError)");
  h.state.cbPayload = { marker: "cached" };
  h.run('setActiveWorkspacePage("cb"); loadActivePageData()');
  assert.equal(h.requests.length, 1);
  const snapshot = h.snapshot();
  h.requests[0].reject(new Error("previous page"));
  await old;
  assert.equal(h.snapshot(), snapshot);
  h.run('setActiveWorkspacePage("long"); loadActivePageData()');
  assert.equal(h.requests.length, 2);
  h.requests[1].resolve({ marker: "reopened" });
  await tick();
  assert.equal(h.state.longPayload.marker, "reopened");
});

for (const spec of pages.slice(0, 3).filter((item) => item.scope !== "long")) {
  test(`${spec.scope}: a current latest response adopts its date and finishes loading`, async () => {
    const h = harness(spec.scope);
    h.state.signalDate = "";
    const current = h.run(`${spec.loader}()`);
    h.requests[0].resolve(spec.data("latest"));
    await current;
    assert.equal(h.state.signalDate, "2026-09-01");
    assert.equal(h.state[spec.loading], false);
    assert.equal(h.state[spec.payload].marker, "latest");
  });
}

test("calendar: obsolete metadata failure is silent; latest metadata cannot override a chosen date", async () => {
  const h = harness("chan");
  const old = h.run("loadCalendar().catch(showError)");
  const current = h.run("loadCalendar().catch(showError)");
  h.run('setWorkspaceSignalDate("2026-08-31")');
  h.requests[1].resolve({ latest_signal_date: "2026-09-02", days: [] });
  await current;
  assert.equal(h.state.signalDate, "2026-08-31");
  const snapshot = h.snapshot();
  h.requests[0].reject(new Error("obsolete calendar"));
  await old;
  assert.equal(h.snapshot(), snapshot);
});

test("calendar: resolving the initial date replaces an undated pending loader", async () => {
  const h = harness("short");
  h.state.signalDate = "";
  const calendar = h.run("loadCalendar()");
  const undated = h.run("loadSelector().catch(showError)");
  h.requests[0].resolve({ latest_signal_date: "2026-09-02", days: [] });
  await calendar;
  assert.equal(h.requests.length, 3);
  assert.match(h.requests[2].url, /signal_date=2026-09-02/);
  h.requests[2].resolve(pages[0].data("dated"));
  await tick();
  const snapshot = h.snapshot();
  h.requests[1].resolve(pages[0].data("undated"));
  await undated;
  assert.equal(h.snapshot(), snapshot);
});

const watchlist = { stocks: [{ symbol: "A", note: "saved" }, { symbol: "B" }] };
const analysis = { watchlist: watchlist.stocks, results: [{ target: { symbol: "A" } }, { target: { symbol: "B" } }] };

for (const phase of ["watchlist", "analysis"]) {
  for (const failure of [false, true]) {
    test(`similar: obsolete ${phase} ${failure ? "failure" : "success"} cannot mutate state or cleanup a newer load`, async () => {
      const h = harness("similar");
      const old = h.run("loadSimilarPatterns().catch(showError)");
      if (phase === "analysis") {
        h.requests[0].resolve(watchlist);
        await tick();
      }
      const stale = h.requests.at(-1);
      h.run('setWorkspaceSignalDate("2026-09-02")');
      const current = h.run("loadSimilarPatterns()");
      const snapshot = h.snapshot();
      const shared = h.state.similarRefreshPromise;
      if (failure) stale.reject(new Error("old similar"));
      else stale.resolve(phase === "analysis" ? analysis : watchlist);
      await old;
      assert.equal(h.snapshot(), snapshot);
      assert.equal(h.state.similarRefreshPromise, shared);
      h.requests.at(-1).resolve(watchlist);
      await tick();
      h.state.similarSelectedSymbol = "B";
      h.requests.at(-1).resolve(analysis);
      await current;
      assert.equal(h.state.similarLoading, false);
      assert.equal(h.state.similarRefreshPromise, null);
      assert.equal(h.state.similarSelectedSymbol, "B");
    });
  }
}

test("similar: an applied watchlist edit invalidates pending analysis and scores but preserves existing details", async () => {
  const h = harness("similar");
  h.state.similarPayload = analysis;
  const pending = h.run("loadSimilarPatterns().catch(showError)");
  h.requests[0].resolve(watchlist);
  await tick();
  const scores = h.run("refreshSimilarWatchlistScores()");
  h.run('applySimilarWatchlistPayload({stocks: [{symbol: "B", note: "new note"}]}, {analysisChanged: true})');
  assert.equal(h.state.similarPayload.results[0].target.symbol, "B");
  const snapshot = h.snapshot();
  h.requests[1].resolve(analysis);
  h.requests[2].reject(new Error("old scores"));
  await pending;
  await scores;
  assert.equal(h.snapshot(), snapshot);
  assert.equal(h.state.similarSelectedSymbol, "B");
});

test("similar: score single-flight cleanup cannot clear a replacement or replay obsolete pending work", async () => {
  const h = harness("similar");
  h.state.similarPayload = analysis;
  const old = h.run("refreshSimilarWatchlistScores()");
  const duplicate = h.run("refreshSimilarWatchlistScores()");
  assert.equal(h.requests.length, 1);
  h.run('setWorkspaceSignalDate("2026-09-02")');
  const current = h.run("refreshSimilarWatchlistScores()");
  const shared = h.state.similarScoreRefreshPromise;
  const snapshot = h.snapshot();
  h.requests[0].resolve(watchlist);
  await old;
  await duplicate;
  assert.equal(h.snapshot(), snapshot);
  assert.equal(h.state.similarScoreRefreshPromise, shared);
  assert.equal(h.requests.length, 2);
  h.requests[1].resolve(watchlist);
  await current;
  assert.equal(h.state.similarScoreRefreshPromise, null);
});

test("different workspace scopes do not supersede each other's request tokens", async () => {
  const h = harness();
  const long = h.run("loadLongStockPool()");
  const cb = h.run("loadConvertibleBondPlan()");
  h.requests[1].resolve(pages[3].data("cb"));
  h.requests[0].resolve({ marker: "long" });
  await long;
  await cb;
  assert.equal(h.state.longPayload.marker, "long");
  assert.equal(h.state.cbPayload.marker, "cb");
});

function respond(request, payload, generation = "generation-1", status = 200) {
  request.resolve({
    ok: status >= 200 && status < 300, status,
    headers: new Headers(generation ? { "X-Quant-Generation": generation } : {}),
    json: async () => payload,
  });
}

test("generation: concurrent workspace reads wait for one unpinned calendar and share its pin", async () => {
  const h = harness("long", { generationTransport: true });
  const calendar = h.run("loadCalendar()");
  const long = h.run("loadLongStockPool()");
  const cb = h.run("loadConvertibleBondPlan()");
  assert.equal(h.requests.length, 1);
  assert.match(h.requests[0].url, /selector\/calendar/);
  assert.equal(h.requests[0].options.headers.get("X-Quant-Generation"), null);
  respond(h.requests[0], { days: [] });
  await tick();
  assert.equal(h.requests.length, 3);
  for (const request of h.requests.slice(1)) {
    assert.equal(request.options.headers.get("X-Quant-Generation"), "generation-1");
  }
  respond(h.requests[1], { marker: "long" });
  respond(h.requests[2], pages[3].data("cb"));
  await calendar;
  await long;
  await cb;
  assert.equal(h.state.workspaceGeneration, "generation-1");
  assert.equal(h.state.longPayload.marker, "long");
  assert.equal(h.state.cbPayload.marker, "cb");
});

test("generation: legacy is a valid pin until 409 reboots against the first committed pointer", async () => {
  const h = harness("long", { generationTransport: true });
  const calendar = h.run("loadCalendar()");
  const initial = h.run("loadLongStockPool()");
  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].options.headers.get("X-Quant-Generation"), null);
  respond(h.requests[0], { days: [] }, "legacy");
  await tick();
  assert.equal(h.state.workspaceGeneration, "legacy");
  assert.equal(h.requests[1].options.headers.get("X-Quant-Generation"), "legacy");
  respond(h.requests[1], { marker: "pre-migration" }, "legacy");
  await calendar;
  await initial;
  assert.equal(h.state.longPayload.marker, "pre-migration");

  const lateLegacy = h.run("loadConvertibleBondPlan().catch(showError)");
  const expiredLegacy = h.run("loadLongStockPool().catch(showError)");
  assert.equal(h.requests[2].options.headers.get("X-Quant-Generation"), "legacy");
  assert.equal(h.requests[3].options.headers.get("X-Quant-Generation"), "legacy");
  respond(h.requests[3], { detail: "Legacy is unavailable after pointer creation" }, "", 409);
  await tick();
  assert.equal(h.state.workspaceGeneration, "");
  assert.equal(h.state.longPayload, null);
  assert.equal(h.requests.length, 5);
  assert.match(h.requests[4].url, /selector\/calendar/);
  assert.equal(h.requests[4].options.headers.get("X-Quant-Generation"), null);
  respond(h.requests[4], { days: [] }, "committed-baseline");
  await tick();
  await expiredLegacy;
  assert.equal(h.requests.length, 6);
  assert.equal(h.requests[5].options.headers.get("X-Quant-Generation"), "committed-baseline");
  assert.equal(h.state.longLoading, true);
  respond(h.requests[5], { marker: "committed" }, "committed-baseline");
  await tick();
  assert.equal(h.state.workspaceGeneration, "committed-baseline");
  assert.equal(h.state.longPayload.marker, "committed");
  const snapshot = h.snapshot();
  respond(h.requests[2], pages[3].data("late legacy"), "legacy");
  await lateLegacy;
  assert.equal(h.snapshot(), snapshot);
  assert.equal(h.state.cbPayload, null);
  assert.deepEqual(h.context.errors, []);
});

test("generation: POST mutations/refresh and status controls bypass both bootstrap and the pin", async () => {
  const h = harness("long", { generationTransport: true });
  h.state.workspaceGeneration = "generation-1";
  const post = h.run('fetchJson("/selector/refresh-latest", {method: "POST", body: "{}"})');
  const status = h.run('fetchJson("/selector/refresh-latest/status")');
  const watchlist = h.run('fetchJson("/similar-patterns/watchlist", {method: "POST", body: "{}"})');
  assert.equal(h.requests.length, 3);
  for (const request of h.requests) {
    assert.equal(request.options?.headers?.["X-Quant-Generation"], undefined);
    request.resolve({ status: "succeeded", generation: "generation-2" });
  }
  await post;
  await status;
  await watchlist;
  assert.equal(h.state.workspaceGeneration, "generation-1");
});

test("generation: pinned 409 coalesces recovery, clears cached views, and reloads the active page", async () => {
  const h = harness("long", { generationTransport: true });
  h.state.workspaceGeneration = "expired";
  h.state.payload = { marker: "cached old selector" };
  h.state.similarPayload = analysis;
  const old = h.run("loadLongStockPool().catch(showError)");
  const cb = h.run("loadConvertibleBondPlan().catch(showError)");
  respond(h.requests[0], { detail: "generation expired" }, "generation-2", 409);
  await tick();
  assert.equal(h.state.workspaceGeneration, "");
  assert.equal(h.state.payload, null);
  assert.equal(h.state.similarPayload, null);
  assert.equal(h.requests.length, 3);
  assert.equal(h.requests[2].options.headers.get("X-Quant-Generation"), null);
  respond(h.requests[1], { detail: "also expired" }, "generation-2", 409);
  await cb;
  assert.equal(h.requests.length, 3);
  respond(h.requests[2], { days: [] }, "generation-2");
  await tick();
  assert.equal(h.requests.length, 4);
  assert.equal(h.requests[3].options.headers.get("X-Quant-Generation"), "generation-2");
  await old;
  assert.equal(h.state.longLoading, true);
  respond(h.requests[3], { marker: "recovered" }, "generation-2");
  await tick();
  assert.equal(h.state.longPayload.marker, "recovered");
  assert.deepEqual(h.context.errors, []);
});

test("generation: recovery failure is visible instead of leaving silent stale content", async () => {
  const h = harness("long", { generationTransport: true });
  h.state.workspaceGeneration = "expired";
  const old = h.run("loadLongStockPool().catch(showError)");
  respond(h.requests[0], {}, "generation-2", 409);
  await tick();
  h.requests[1].reject(new Error("calendar recovery failed"));
  await old;
  assert.deepEqual(h.context.errors, ["calendar recovery failed"]);
  assert.equal(h.state.longPayload, null);
  assert.equal(h.state.longLoading, false);
  assert.equal(h.state.workspaceGeneration, "");
});

for (const failure of ["409", "mismatched success"]) {
  test(`generation: persistent ${failure} with an unchanged pin stops after one automatic reload`, async () => {
    const h = harness("long", { generationTransport: true });
    h.state.workspaceGeneration = "generation-1";
    const fail = (request) => failure === "409"
      ? respond(request, { detail: "persistent conflict" }, "", 409)
      : respond(request, { marker: "wrong generation" }, "generation-other");
    const initial = h.run("loadLongStockPool().catch(showError)");
    fail(h.requests[0]);
    await tick();
    respond(h.requests[1], { days: [] }, "generation-1");
    await tick();
    await initial;
    assert.equal(h.state.workspaceGenerationReloadPromise, null);
    fail(h.requests[2]);
    await tick();
    assert.equal(h.requests.length, 3, "one calendar reload and one page retry only");
    assert.equal(h.state.workspaceGenerationEpoch, 1);
    assert.equal(h.state.workspaceGeneration, "generation-1");
    assert.equal(h.state.longPayload, null);
    assert.equal(h.state.longLoading, false);
    assert.match(h.state.longError, /停止自动重载/);
    assert.ok(h.context.errors.some((message) => message.includes("停止自动重载")));

    const manual = h.run("loadLongStockPool().catch(showError)");
    fail(h.requests[3]);
    await tick();
    assert.equal(h.requests.length, 4, "same-selection reloads cannot replenish the recovery budget");
    await manual;
    assert.equal(h.state.longLoading, false);
  });
}

test("generation: a new committed generation gets one recovery attempt of its own", async () => {
  const h = harness("long", { generationTransport: true });
  h.state.workspaceGeneration = "generation-1";
  const initial = h.run("loadLongStockPool().catch(showError)");
  respond(h.requests[0], {}, "", 409);
  await tick();
  respond(h.requests[1], { days: [] }, "generation-2");
  await tick();
  await initial;
  respond(h.requests[2], {}, "", 409);
  await tick();
  assert.equal(h.requests.length, 4);
  assert.equal(h.requests[3].options.headers.get("X-Quant-Generation"), null);
  respond(h.requests[3], { days: [] }, "generation-2");
  await tick();
  respond(h.requests[4], {}, "", 409);
  await tick();
  assert.equal(h.requests.length, 5, "generation-2 cannot trigger a second recovery");
  assert.equal(h.state.workspaceGenerationEpoch, 2);
  assert.equal(h.state.longLoading, false);
  assert.match(h.state.longError, /停止自动重载/);
});

for (const selection of ["date", "variant"]) {
  test(`generation: changing ${selection} permits one recovery without erasing the previous budget`, async () => {
    const h = harness("long", { generationTransport: true });
    h.state.workspaceGeneration = "generation-1";
    const initial = h.run("loadLongStockPool().catch(showError)");
    respond(h.requests[0], {}, "", 409);
    await tick();
    respond(h.requests[1], { days: [] }, "generation-1");
    await tick();
    await initial;
    respond(h.requests[2], {}, "", 409);
    await tick();
    assert.equal(h.requests.length, 3);

    if (selection === "date") h.run('setWorkspaceSignalDate("2026-09-02")');
    else h.run('state.longVariant = "blood_chip"; invalidateWorkspaceRequest("long")');
    const changed = h.run("loadLongStockPool().catch(showError)");
    respond(h.requests[3], {}, "", 409);
    await tick();
    assert.equal(h.requests.length, 5);
    respond(h.requests[4], { days: [] }, "generation-1");
    await tick();
    await changed;
    respond(h.requests[5], { marker: "recovered selection" }, "generation-1");
    await tick();
    assert.equal(h.state.longPayload.marker, "recovered selection");

    if (selection === "date") h.run('setWorkspaceSignalDate("2026-09-01")');
    else h.run('state.longVariant = "tea"; invalidateWorkspaceRequest("long")');
    const previous = h.run("loadLongStockPool().catch(showError)");
    respond(h.requests[6], {}, "", 409);
    await tick();
    assert.equal(h.requests.length, 7, "returning to an exhausted selection must not restart recovery");
    await previous;
    assert.equal(h.state.longLoading, false);
  });
}

for (const race of ["changed date", "A-B-A date", "same selection"]) {
  test(`generation: an obsolete ${race} request's 409 cannot reset the active workspace`, async () => {
    const h = harness("long", { generationTransport: true });
    h.state.workspaceGeneration = "generation-1";
    const old = h.run("loadLongStockPool().catch(showError)");
    if (race !== "same selection") h.run('setWorkspaceSignalDate("2026-09-02")');
    if (race === "A-B-A date") h.run('setWorkspaceSignalDate("2026-09-01")');
    const current = h.run("loadLongStockPool()");
    const snapshot = h.snapshot();
    respond(h.requests[0], {}, "", 409);
    await tick();
    assert.equal(h.requests.length, 2);
    await old;
    assert.equal(h.snapshot(), snapshot);
    respond(h.requests[1], { marker: "current selection" }, "generation-1");
    await current;
    assert.equal(h.state.longPayload.marker, "current selection");
  });
}

test("generation: a mismatched success header is rejected and recovered like 409", async () => {
  const h = harness("long", { generationTransport: true });
  h.state.workspaceGeneration = "generation-1";
  const old = h.run("loadLongStockPool().catch(showError)");
  respond(h.requests[0], { marker: "mixed-generation" }, "generation-2");
  await tick();
  assert.equal(h.state.longPayload, null);
  assert.equal(h.requests[1].options.headers.get("X-Quant-Generation"), null);
  respond(h.requests[1], { days: [] }, "generation-2");
  await tick();
  respond(h.requests[2], { marker: "consistent" }, "generation-2");
  await old;
  await tick();
  assert.equal(h.state.longPayload.marker, "consistent");
});

test("generation: a reset during bootstrap cannot repin an obsolete initial response", async () => {
  const h = harness("long", { generationTransport: true });
  const old = h.run("loadLongStockPool().catch(showError)");
  h.run("resetWorkspaceGeneration()");
  const current = h.run("loadLongStockPool()");
  respond(h.requests[0], { days: [] }, "generation-1");
  await old;
  assert.equal(h.state.workspaceGeneration, "");
  assert.equal(h.state.longLoading, true);
  respond(h.requests[1], { days: [] }, "generation-2");
  await tick();
  assert.equal(h.requests.length, 3);
  respond(h.requests[2], { marker: "current" }, "generation-2");
  await current;
  assert.equal(h.state.workspaceGeneration, "generation-2");
  assert.deepEqual(h.context.errors, []);
});

test("generation: daily refresh completion clears the pin before calendar and view reloads", async () => {
  const h = harness("long", { generationTransport: true });
  h.state.workspaceGeneration = "generation-1";
  const stale = h.run("loadLongStockPool().catch(showError)");
  const refresh = h.run('reloadAfterRefresh({scope: "long", status: "succeeded"})');
  assert.equal(h.state.workspaceGeneration, "");
  assert.equal(h.requests[1].options.headers.get("X-Quant-Generation"), null);
  respond(h.requests[1], { days: [] }, "generation-2");
  await tick();
  assert.equal(h.requests.length, 3);
  assert.equal(h.requests[2].options.headers.get("X-Quant-Generation"), "generation-2");
  respond(h.requests[2], { marker: "published" }, "generation-2");
  await refresh;
  const snapshot = h.snapshot();
  respond(h.requests[0], { marker: "stale-before-publication" }, "generation-1");
  await stale;
  assert.equal(h.snapshot(), snapshot);
  assert.equal(h.state.longPayload.marker, "published");
});

test("generation: missing bootstrap header fails closed and a later load can retry", async () => {
  const h = harness("long", { generationTransport: true });
  const failed = h.run("loadLongStockPool()");
  respond(h.requests[0], { days: [] }, "");
  await assert.rejects(failed, /X-Quant-Generation/);
  assert.equal(h.state.workspaceGeneration, "");
  assert.equal(h.state.workspaceGenerationPromise, null);
  const current = h.run("loadLongStockPool()");
  respond(h.requests[1], { days: [] }, "generation-2");
  await tick();
  respond(h.requests[2], { marker: "retried" }, "generation-2");
  await current;
  assert.equal(h.state.longPayload.marker, "retried");
});
