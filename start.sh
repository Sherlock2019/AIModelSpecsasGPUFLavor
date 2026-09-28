#!/usr/bin/env bash
# LLM GPU Calculator launcher.
#
#   ./start.sh                  set up (first run) and start in the background on 0.0.0.0:$PORT
#   ./start.sh run              same, in the foreground (used by the systemd service)
#   ./start.sh stop | restart | status | logs
#   ./start.sh build            force re-install dependencies and rebuild the web UI
#   ./start.sh install-service  run at boot with systemd (survives reboots / SSH logout)
#   ./start.sh uninstall-service
#
# Settings come from the environment or ./.env (created on first run):
#   PORT=8080          port to serve on (open it in the EC2 security group)
#   HOST=0.0.0.0       bind address
#   ADMIN_TOKEN=...    required for catalog/settings changes; generated if absent
#   DATABASE_URL=...   optional PostgreSQL URL (default: SQLite in backend/var/)
#
# Works on Amazon Linux 2023 and Ubuntu 22.04+ (x86_64 and arm64/Graviton). Missing Python is
# installed with the OS package manager; Node.js (only needed to build the UI) is downloaded to
# ./.tools and verified against the official SHA-256 checksums - no system-wide install.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
RUN_DIR="$BACKEND/var"
PID_FILE="$RUN_DIR/gpucalc.pid"
LOG_FILE="$RUN_DIR/gpucalc.log"
ENV_FILE="$ROOT/.env"
TOOLS="$ROOT/.tools"
VENV="$BACKEND/.venv"
NODE_MAJOR=22
SERVICE=gpucalc

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

bold=$'\e[1m'; dim=$'\e[2m'; green=$'\e[32m'; yellow=$'\e[33m'; red=$'\e[31m'; reset=$'\e[0m'
[[ -t 1 ]] || { bold=; dim=; green=; yellow=; red=; reset=; }
info() { echo "${bold}==>${reset} $*"; }
ok()   { echo "${green}✓${reset} $*"; }
warn() { echo "${yellow}!${reset} $*" >&2; }
die()  { echo "${red}✗ $*${reset}" >&2; exit 1; }

as_root() {
  if [[ $EUID -eq 0 ]]; then "$@"
  elif command -v sudo >/dev/null; then sudo "$@"
  else die "Need root to run: $* (install sudo or run as root)"; fi
}

# ---------------------------------------------------------------------------- config

ensure_env() {
  mkdir -p "$RUN_DIR"
  if [[ ! -f "$ENV_FILE" ]]; then
    printf '# LLM GPU Calculator settings (read by start.sh)\nPORT=%s\nHOST=%s\n' "$PORT" "$HOST" > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
  fi
  if [[ -z "${ADMIN_TOKEN+x}" ]]; then
    ADMIN_TOKEN="$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    printf 'ADMIN_TOKEN=%s\n' "$ADMIN_TOKEN" >> "$ENV_FILE"
    ok "Generated an admin token (saved in .env)"
  fi
  export ADMIN_TOKEN PORT HOST
  [[ -n "$ADMIN_TOKEN" ]] || warn "ADMIN_TOKEN is empty: anyone who can reach the site can edit the catalog and settings."
}

# ---------------------------------------------------------------------------- python

python_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }

find_python() {
  local c
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null && python_ok "$c"; then command -v "$c"; return 0; fi
  done
  return 1
}

install_python() {
  info "Installing Python 3.10+ with the system package manager"
  if command -v dnf >/dev/null; then
    as_root dnf install -y python3.11 python3.11-pip
  elif command -v apt-get >/dev/null; then
    # python3-venv provides pip inside the virtual environment; python3-pip would pull in a compiler toolchain.
    as_root apt-get update -y && as_root apt-get install -y python3 python3-venv
  else
    die "Python 3.10+ not found and no dnf/apt-get available. Install Python 3.10+ and re-run."
  fi
}

ensure_venv() {
  if [[ ! -x "$VENV/bin/python" ]]; then
    local py
    py="$(find_python)" || { install_python; py="$(find_python)" || die "Python 3.10+ still not found"; }
    info "Creating Python virtual environment ($("$py" --version))"
    if ! "$py" -m venv "$VENV" 2>/dev/null; then
      # Debian/Ubuntu ship venv separately
      command -v apt-get >/dev/null || die "Could not create a virtual environment with $py"
      local ver; ver="$("$py" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
      as_root apt-get update -y
      as_root apt-get install -y "python${ver}-venv" || as_root apt-get install -y python3-venv
      rm -rf "$VENV"
      "$py" -m venv "$VENV"
    fi
  fi
  local stamp="$VENV/.deps-installed"
  if [[ ! -f "$stamp" || "$BACKEND/requirements.txt" -nt "$stamp" ]]; then
    info "Installing Python dependencies"
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q -r "$BACKEND/requirements.txt"
    touch "$stamp"
    ok "Python dependencies ready"
  fi
}

# ---------------------------------------------------------------------------- node + web UI

node_ok() { command -v node >/dev/null && [[ "$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)" -ge 20 ]]; }

ensure_node() {
  [[ -x "$TOOLS/node/bin/node" ]] && export PATH="$TOOLS/node/bin:$PATH"
  node_ok && return 0
  local arch
  case "$(uname -m)" in
    x86_64) arch=x64 ;;
    aarch64 | arm64) arch=arm64 ;;
    *) die "Unsupported CPU architecture $(uname -m) for the bundled Node.js download" ;;
  esac
  command -v curl >/dev/null || die "curl is required"
  info "Downloading Node.js ${NODE_MAJOR}.x (linux-$arch) to .tools/ (build-time only)"
  local base="https://nodejs.org/dist/latest-v${NODE_MAJOR}.x" sums file sum
  sums="$(curl -fsSL "$base/SHASUMS256.txt")" || die "Could not reach nodejs.org"
  file="$(awk -v s="linux-$arch.tar.xz" '$2 ~ s"$" {print $2; exit}' <<<"$sums")"
  sum="$(awk -v f="$file" '$2 == f {print $1; exit}' <<<"$sums")"
  [[ -n "$file" && -n "$sum" ]] || die "Could not find a Node.js build for linux-$arch"
  mkdir -p "$TOOLS"
  curl -fsSL -o "$TOOLS/$file" "$base/$file"
  echo "$sum  $TOOLS/$file" | sha256sum -c --quiet - || die "Node.js checksum mismatch - download discarded"
  rm -rf "$TOOLS/node" && mkdir -p "$TOOLS/node"
  tar -xJf "$TOOLS/$file" -C "$TOOLS/node" --strip-components=1
  rm -f "$TOOLS/$file"
  export PATH="$TOOLS/node/bin:$PATH"
  node_ok || die "Node.js install failed"
  ok "Node.js $(node --version) ready"
}

ui_stale() {
  local index="$FRONTEND/dist/index.html"
  [[ ! -f "$index" ]] && return 0
  [[ -n "$(find "$FRONTEND/src" "$FRONTEND/index.html" "$FRONTEND/package.json" "$FRONTEND/vite.config.ts" -newer "$index" -print -quit)" ]]
}

build_ui() {
  ensure_node
  cd "$FRONTEND"
  if [[ ! -d node_modules || package-lock.json -nt node_modules/.package-lock.json ]]; then
    info "Installing web UI dependencies"
    npm ci --no-audit --no-fund --loglevel=error
  fi
  info "Building the web UI"
  # Vite only; type checking is a development step and needs extra memory on small instances.
  NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=1024}" npx vite build --logLevel warn
  cd "$ROOT"
  ok "Web UI built"
}

prepare() {
  ensure_env
  ensure_venv
  if ui_stale; then build_ui; fi
}

# ---------------------------------------------------------------------------- process control

running_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid; pid="$(cat "$PID_FILE")"
  kill -0 "$pid" 2>/dev/null && { echo "$pid"; return 0; }
  rm -f "$PID_FILE"
  return 1
}

check_port() {
  if (( PORT < 1024 )) && [[ $EUID -ne 0 ]]; then
    die "Port $PORT needs root. Use PORT=8080 (default) or run 'sudo ./start.sh'."
  fi
  if command -v ss >/dev/null && ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q .; then
    die "Port $PORT is already in use. Set another one, e.g. PORT=8090 ./start.sh"
  fi
}

uvicorn_cmd() {
  UVICORN=("$VENV/bin/uvicorn" app.main:app --host "$HOST" --port "$PORT" --workers 1 --proxy-headers "--forwarded-allow-ips=*")
}

wait_healthy() {
  local i
  for i in $(seq 1 40); do
    curl -fs -m 2 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  return 1
}

public_ip() {
  local token ip=""
  token="$(curl -fs -m 2 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true)"
  if [[ -n "$token" ]]; then
    ip="$(curl -fs -m 2 -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true)"
  fi
  [[ -z "$ip" ]] && ip="$(curl -fs -m 3 https://checkip.amazonaws.com 2>/dev/null | tr -d '[:space:]' || true)"
  echo "$ip"
}

print_urls() {
  local ip; ip="$(public_ip)"
  echo
  echo "${bold}LLM GPU Calculator is running${reset}"
  echo "  Local:   http://localhost:$PORT"
  if [[ -n "$ip" ]]; then
    echo "  Public:  ${bold}http://$ip:$PORT${reset}"
  else
    echo "  Public:  http://<your-public-ip>:$PORT"
  fi
  echo "  API docs: /docs    Logs: ./start.sh logs"
  echo
  echo "${dim}If the public URL does not load, allow inbound TCP $PORT in the instance's security group"
  echo "(EC2 console → Instances → Security → Security groups → Edit inbound rules), ideally from your IP only.${reset}"
  if [[ -n "${ADMIN_TOKEN:-}" ]]; then
    echo "${dim}To edit catalogs or settings, paste the admin token into Settings → Admin access:${reset}"
    echo "  ADMIN_TOKEN=$ADMIN_TOKEN   ${dim}(stored in .env)${reset}"
  fi
}

start() {
  if pid="$(running_pid)"; then
    ok "Already running (pid $pid)"
    print_urls
    return 0
  fi
  prepare
  check_port
  info "Starting on $HOST:$PORT"
  cd "$BACKEND"
  uvicorn_cmd
  setsid nohup "${UVICORN[@]}" >>"$LOG_FILE" 2>&1 </dev/null &
  echo $! >"$PID_FILE"
  cd "$ROOT"
  if wait_healthy; then
    ok "Healthy"
    print_urls
  else
    warn "Did not become healthy. Last log lines:"
    tail -n 30 "$LOG_FILE" >&2 || true
    stop >/dev/null 2>&1 || true
    exit 1
  fi
}

run_foreground() {
  prepare
  check_port
  cd "$BACKEND"
  uvicorn_cmd
  exec "${UVICORN[@]}"
}

stop() {
  local pid
  if ! pid="$(running_pid)"; then
    ok "Not running"
    return 0
  fi
  info "Stopping (pid $pid)"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  ok "Stopped"
}

status() {
  if pid="$(running_pid)"; then
    ok "Running (pid $pid) on port $PORT"
    curl -fs -m 2 "http://127.0.0.1:$PORT/healthz" >/dev/null && ok "Health check OK" || warn "Health check failed"
  elif command -v systemctl >/dev/null && systemctl is-active --quiet "$SERVICE" 2>/dev/null; then
    ok "Running as systemd service '$SERVICE' (journalctl -u $SERVICE -f for logs)"
  else
    echo "Not running"
    return 1
  fi
}

# ---------------------------------------------------------------------------- systemd

install_service() {
  command -v systemctl >/dev/null || die "systemd not available"
  prepare
  stop >/dev/null 2>&1 || true
  local unit="/etc/systemd/system/$SERVICE.service"
  info "Installing systemd service $SERVICE (runs as $(id -un))"
  as_root tee "$unit" >/dev/null <<EOF
[Unit]
Description=LLM GPU Calculator
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(id -un)
WorkingDirectory=$ROOT
EnvironmentFile=-$ENV_FILE
ExecStart=$ROOT/start.sh run
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
  as_root systemctl daemon-reload
  as_root systemctl enable --now "$SERVICE"
  wait_healthy || { as_root journalctl -u "$SERVICE" -n 30 --no-pager; die "Service did not become healthy"; }
  ok "Service enabled - starts automatically at boot"
  print_urls
}

uninstall_service() {
  command -v systemctl >/dev/null || die "systemd not available"
  as_root systemctl disable --now "$SERVICE" 2>/dev/null || true
  as_root rm -f "/etc/systemd/system/$SERVICE.service"
  as_root systemctl daemon-reload
  ok "Service removed"
}

# ---------------------------------------------------------------------------- main

cmd="${1:-start}"
case "$cmd" in
  start) start ;;
  run) run_foreground ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs) touch "$LOG_FILE" 2>/dev/null || true; exec tail -n 100 -f "$LOG_FILE" ;;
  build) ensure_env; rm -f "$VENV/.deps-installed"; ensure_venv; build_ui ;;
  install-service) install_service ;;
  uninstall-service) uninstall_service ;;
  -h | --help | help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "Unknown command '$cmd'. Try ./start.sh help" ;;
esac
