#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"

LIMA_INSTANCE="${LIMA_INSTANCE:-ubuntu-22.04-x86_64}"
LIMA_HOST="${LIMA_HOST:-lima-ubuntu-22-04-x86-64}"
HOST_PORT="${HOST_PORT:-5555}"
GUEST_TTY_LINK="${GUEST_TTY_LINK:-/tmp/ttyUSB0}"
GUEST_DEV_LINK="${GUEST_DEV_LINK:-/dev/ttyUSB0}"
BAUD="${BAUD:-115200}"
STATE_DIR="${STATE_DIR:-/tmp/beingbeyond-lima-usb-forward}"
HOST_LOG="${STATE_DIR}/host-socat.log"
HOST_PID_FILE="${STATE_DIR}/host-socat.pid"
GUEST_UNIT="${GUEST_UNIT:-beingbeyond-usb-forward}"
SSH_OPTS=(-o ControlMaster=no -o ControlPath=none)

mkdir -p "${STATE_DIR}"

detect_usb_device() {
  if [[ -n "${USB_SERIAL_DEVICE:-}" ]]; then
    printf '%s\n' "${USB_SERIAL_DEVICE}"
    return 0
  fi

  local devices=()
  while IFS= read -r line; do
    [[ -n "${line}" ]] && devices+=("${line}")
  done < <(find /dev -maxdepth 1 \( -name 'cu.usbserial*' -o -name 'tty.usbserial*' \) | sort)

  if [[ "${#devices[@]}" -eq 0 ]]; then
    echo "No host USB serial device found under /dev/cu.usbserial* or /dev/tty.usbserial*" >&2
    exit 1
  fi

  printf '%s\n' "${devices[0]}"
}

host_process_running() {
  if [[ -f "${HOST_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${HOST_PID_FILE}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

ensure_guest_ready() {
  ssh "${SSH_OPTS[@]}" -o ConnectTimeout=10 "${LIMA_HOST}" 'true' >/dev/null
}

start_host_forwarder() {
  local usb_device="$1"

  stty -f "${usb_device}" "${BAUD}" raw -echo

  nohup socat -d -d \
    "TCP-LISTEN:${HOST_PORT},reuseaddr,fork" \
    "FILE:${usb_device},raw,echo=0" \
    >"${HOST_LOG}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${HOST_PID_FILE}"

  for _ in {1..20}; do
    if lsof -nP -iTCP:"${HOST_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.2
  done

  echo "Host forwarder failed to listen on port ${HOST_PORT}" >&2
  [[ -f "${HOST_LOG}" ]] && cat "${HOST_LOG}" >&2
  exit 1
}

start_guest_forwarder() {
  ssh "${SSH_OPTS[@]}" "${LIMA_HOST}" \
    env HOST_PORT="${HOST_PORT}" GUEST_TTY_LINK="${GUEST_TTY_LINK}" GUEST_DEV_LINK="${GUEST_DEV_LINK}" GUEST_UNIT="${GUEST_UNIT}" \
    'bash -s' <<'EOF'
set -euo pipefail
sudo systemctl stop "${GUEST_UNIT}" >/dev/null 2>&1 || true
sudo systemctl reset-failed "${GUEST_UNIT}" >/dev/null 2>&1 || true
for _ in $(seq 1 20); do
  if ! systemctl status "${GUEST_UNIT}" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done
HOST_IP="$(ip route | sed -n 's/^default via \([^ ]*\).*/\1/p' | head -n 1)"
rm -f "${GUEST_TTY_LINK}"
sudo systemd-run \
  --unit="${GUEST_UNIT}" \
  --property=Type=simple \
  /bin/bash -lc "exec socat -d -d PTY,link='${GUEST_TTY_LINK}',rawer,echo=0,wait-slave TCP:${HOST_IP}:${HOST_PORT}"
sleep 1
sudo ln -sf "${GUEST_TTY_LINK}" "${GUEST_DEV_LINK}"
if [[ -e "${GUEST_TTY_LINK}" ]]; then
  sudo chmod 666 "$(readlink -f "${GUEST_TTY_LINK}")"
fi
EOF
}

stop_host_forwarder() {
  if host_process_running; then
    local pid
    pid="$(cat "${HOST_PID_FILE}")"
    kill "${pid}" 2>/dev/null || true
  fi
  pkill -f "socat -d -d TCP-LISTEN:${HOST_PORT},reuseaddr,fork" >/dev/null 2>&1 || true
  rm -f "${HOST_PID_FILE}"
}

stop_guest_forwarder() {
  ssh "${SSH_OPTS[@]}" "${LIMA_HOST}" \
    "bash -lc \"sudo systemctl stop '${GUEST_UNIT}' >/dev/null 2>&1 || true; pkill -f 'socat -d -d PTY,link=${GUEST_TTY_LINK}' >/dev/null 2>&1 || true; sudo rm -f '${GUEST_DEV_LINK}'; rm -f '${GUEST_TTY_LINK}'\"" >/dev/null 2>&1 || true
}

show_status() {
  local usb_device="${1:-}"

  echo "LIMA_HOST=${LIMA_HOST}"
  echo "HOST_PORT=${HOST_PORT}"
  [[ -n "${usb_device}" ]] && echo "USB_SERIAL_DEVICE=${usb_device}"

  if host_process_running; then
    echo "Host forwarder: running (pid $(cat "${HOST_PID_FILE}"))"
  else
    echo "Host forwarder: stopped"
  fi

  if ssh "${SSH_OPTS[@]}" "${LIMA_HOST}" "bash -lc 'ls -l \"${GUEST_DEV_LINK}\" \"${GUEST_TTY_LINK}\" 2>/dev/null || true'" ; then
    :
  fi
}

main() {
  require_cmd ssh
  require_cmd socat

  case "${ACTION}" in
    start)
      local usb_device
      usb_device="$(detect_usb_device)"
      ensure_guest_ready
      stop_guest_forwarder
      stop_host_forwarder
      start_host_forwarder "${usb_device}"
      start_guest_forwarder
      echo "Forwarding ${usb_device} -> ${LIMA_HOST}:${GUEST_DEV_LINK}"
      ;;
    stop)
      stop_guest_forwarder
      stop_host_forwarder
      echo "USB forwarder stopped"
      ;;
    restart)
      "$0" stop
      "$0" start
      ;;
    status)
      show_status "$(detect_usb_device || true)"
      ;;
    *)
      echo "Usage: $0 {start|stop|restart|status}" >&2
      exit 2
      ;;
  esac
}

main "$@"
