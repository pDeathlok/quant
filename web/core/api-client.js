export function createApiClient(apiBase) {
  const normalizedBase = String(apiBase || "").replace(/\/+$/, "");

  return async function fetchJson(path, options = {}) {
    const { timeoutMs = 0, ...fetchOptions } = options;
    const controller = timeoutMs > 0 && !fetchOptions.signal ? new AbortController() : null;
    const timeoutId = controller ? window.setTimeout(() => controller.abort(), timeoutMs) : null;
    let response;
    try {
      response = await fetch(`${normalizedBase}${path}`, {
        cache: "no-store",
        ...fetchOptions,
        ...(controller ? { signal: controller.signal } : {}),
        headers: {
          ...(fetchOptions.body ? { "Content-Type": "application/json" } : {}),
          ...(fetchOptions.headers || {}),
        },
      });
    } catch (error) {
      if (controller?.signal.aborted) {
        throw new Error(`${path} 加载超时，请稍后重试`);
      }
      throw error;
    } finally {
      if (timeoutId !== null) window.clearTimeout(timeoutId);
    }
    if (!response.ok) {
      let detail = "";
      try {
        detail = (await response.json()).detail || "";
      } catch {
        detail = "";
      }
      throw new Error(`${path} 加载失败: ${response.status}${detail ? ` · ${detail}` : ""}`);
    }
    return response.json();
  };
}
