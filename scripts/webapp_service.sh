#!/bin/bash

set -euo pipefail

LABEL="com.didi.quant.webapp"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="$PROJECT_ROOT/config/launchd/$LABEL.plist.in"
AGENT_DIR="$HOME/Library/LaunchAgents"
PLIST="$AGENT_DIR/$LABEL.plist"
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/$LABEL"

find_python() {
    local candidate
    local candidates=(
        "${QUANT_PYTHON:-}"
        "$PROJECT_ROOT/.venv/bin/python"
        "$HOME/miniforge3/bin/python"
        "$HOME/miniconda3/bin/python"
        "$(command -v python3 || true)"
    )
    for candidate in "${candidates[@]}"; do
        if [[ -n "$candidate" && -x "$candidate" ]] &&
            "$candidate" -c 'import fastapi, pandas, uvicorn' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    echo "找不到已安装 fastapi、pandas、uvicorn 的 Python；可设置 QUANT_PYTHON=/absolute/path/python" >&2
    return 1
}

render_plist() {
    local python_path="$1"
    local output_path="$2"
    local escaped_root escaped_python
    escaped_root="${PROJECT_ROOT//&/\\&}"
    escaped_python="${python_path//&/\\&}"
    sed \
        -e "s|__PROJECT_ROOT__|$escaped_root|g" \
        -e "s|__PYTHON__|$escaped_python|g" \
        "$TEMPLATE" >"$output_path"
    plutil -lint "$output_path" >/dev/null
}

bootstrap_service() {
    local attempt
    for attempt in 1 2 3 4 5; do
        if ((attempt < 5)); then
            if launchctl bootstrap "$DOMAIN" "$PLIST" >/dev/null 2>&1; then
                return 0
            fi
        elif launchctl bootstrap "$DOMAIN" "$PLIST"; then
            return 0
        fi
        sleep 1
    done
    echo "无法加载 $LABEL；请检查 plist 和 launchd 日志" >&2
    return 1
}

wait_for_health() {
    local deadline response
    deadline=$((SECONDS + 60))
    while ((SECONDS < deadline)); do
        if response="$(curl --fail --silent --max-time 3 http://127.0.0.1:8088/api/health 2>/dev/null)"; then
            printf '%s\n' "$response"
            return 0
        fi
        sleep 1
    done
    echo "服务已交给 launchd，但 60 秒内健康检查未通过" >&2
    return 1
}

install_service() {
    local python_path temp_plist
    python_path="$(find_python)"
    mkdir -p "$PROJECT_ROOT/.run" "$AGENT_DIR"
    temp_plist="$(mktemp "${TMPDIR:-/tmp}/quant-webapp.plist.XXXXXX")"
    trap 'rm -f "$temp_plist"' EXIT
    render_plist "$python_path" "$temp_plist"
    launchctl bootout "$SERVICE" >/dev/null 2>&1 || true
    install -m 600 "$temp_plist" "$PLIST"
    bootstrap_service
    launchctl enable "$SERVICE"
    launchctl kickstart -k "$SERVICE"
    rm -f "$temp_plist"
    trap - EXIT
    echo "已安装并启动 $LABEL (Python: $python_path)"
    wait_for_health
}

start_service() {
    if [[ ! -f "$PLIST" ]]; then
        echo "服务尚未安装，请先运行: $0 install" >&2
        return 1
    fi
    if ! launchctl print "$SERVICE" >/dev/null 2>&1; then
        bootstrap_service
    fi
    launchctl enable "$SERVICE"
    launchctl kickstart "$SERVICE"
    echo "已启动 $LABEL"
    wait_for_health
}

stop_service() {
    launchctl bootout "$SERVICE" >/dev/null 2>&1 || true
    echo "已停止 $LABEL；plist 保留，可用 '$0 start' 再启动"
}

restart_service() {
    if launchctl print "$SERVICE" >/dev/null 2>&1; then
        launchctl kickstart -k "$SERVICE"
    else
        start_service
        return
    fi
    echo "已重启 $LABEL"
    wait_for_health
}

status_service() {
    launchctl print "$SERVICE" 2>/dev/null | sed -n '1,45p' || true
    echo
    wait_for_health
}

uninstall_service() {
    launchctl bootout "$SERVICE" >/dev/null 2>&1 || true
    if [[ -f "$PLIST" ]]; then
        mv "$PLIST" "$HOME/.Trash/$LABEL.plist.$(date +%Y%m%d%H%M%S)"
    fi
    echo "已卸载 $LABEL；plist 已移到废纸篓"
}

case "${1:-status}" in
    install) install_service ;;
    start) start_service ;;
    stop) stop_service ;;
    restart) restart_service ;;
    status) status_service ;;
    logs) tail -n 100 -F "$PROJECT_ROOT/.run/webapp.log" ;;
    uninstall) uninstall_service ;;
    *) echo "用法: $0 {install|start|stop|restart|status|logs|uninstall}" >&2; exit 2 ;;
esac
