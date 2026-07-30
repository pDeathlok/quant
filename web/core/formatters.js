export const fmtPct = (value, digits = 2) => {
  if (value == null || Number.isNaN(Number(value))) return "-";
  return `${Number(value).toFixed(digits)}%`;
};

export const fmtRate = (value, digits = 1) => {
  if (value == null || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(digits)}%`;
};

export const fmtNumber = (value, digits = 2) => {
  if (value == null || value === "" || Number.isNaN(Number(value))) return "-";
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "-";
};

export const fmtPrice = (value) => {
  if (value == null || value === "" || Number.isNaN(Number(value))) return "-";
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? numeric.toFixed(2) : "-";
};

const t1ReturnBucketOrder = (bucket) => {
  const text = String(bucket || "");
  const knownOrder = {
    "大跌<=-3%": 0,
    "小跌-3~-1%": 1,
    "震荡-1~1%": 2,
    "小涨1~3%": 3,
    "大涨>=3%": 4,
  };
  if (Object.prototype.hasOwnProperty.call(knownOrder, text)) return knownOrder[text];
  const numbers = text.match(/-?\d+(?:\.\d+)?/g)?.map(Number).filter(Number.isFinite) || [];
  if (!numbers.length) return Number.POSITIVE_INFINITY;
  return Math.min(...numbers);
};

const t1VolumeBucketOrder = (bucket) => {
  const text = String(bucket || "");
  const knownOrder = {
    "缩量<0.8": 0,
    "平量0.8~1.3": 1,
    "放量1.3~2": 2,
    "巨量>2": 3,
  };
  if (Object.prototype.hasOwnProperty.call(knownOrder, text)) return knownOrder[text];
  return text;
};

export const sortT1ScenarioRows = (rows) => [...(rows || [])].sort((a, b) => {
  const returnDiff = t1ReturnBucketOrder(a.t1_return_bucket) - t1ReturnBucketOrder(b.t1_return_bucket);
  if (returnDiff !== 0) return returnDiff;
  const volumeA = t1VolumeBucketOrder(a.t1_volume_bucket);
  const volumeB = t1VolumeBucketOrder(b.t1_volume_bucket);
  if (typeof volumeA === "number" && typeof volumeB === "number") return volumeA - volumeB;
  return String(volumeA).localeCompare(String(volumeB), "zh-Hans-CN");
});

export const fmtWeight = (value) => {
  if (value == null || value === "" || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
};

export const fmtMoney = (value) => {
  if (value == null || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 0 });
};

export const fmtRange = (range) => {
  if (!range) return "-";
  if (range.label) return range.label;
  const low = fmtPrice(range.low);
  const high = fmtPrice(range.high);
  return low === high ? low : `${low}-${high}`;
};

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
