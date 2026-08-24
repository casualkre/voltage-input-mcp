#!/usr/bin/env bash
# One command from a fresh clone to a working install.
#
#   ./install.sh
#
# Does everything that does not need root, prints the exact command for anything that
# does, and hands off to `voltage setup` for models and client registration.
# Safe to re-run: every step checks before acting.

set -uo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

b() { printf '\033[1m%s\033[0m\n' "$*"; }
ok() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
no() { printf '  \033[31m✗\033[0m %s\n' "$*"; }
cmd() { printf '      \033[36m%s\033[0m\n' "$*"; }

SUDO=()

printf '\033[36m'
cat <<'ART'
 ██╗   ██╗ ██████╗ ██╗  ████████╗ █████╗  ██████╗ ███████╗
 ██║   ██║██╔═══██╗██║  ╚══██╔══╝██╔══██╗██╔════╝ ██╔════╝
 ██║   ██║██║   ██║██║     ██║   ███████║██║  ███╗█████╗
 ╚██╗ ██╔╝██║   ██║██║     ██║   ██╔══██║██║   ██║██╔══╝
  ╚████╔╝ ╚██████╔╝███████╗██║   ██║  ██║╚██████╔╝███████╗
   ╚═══╝   ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
ART
printf '\033[0m\n'

# ------------------------------------------------------------------ 1. python
b "1/5  Python"
if ! command -v python3 >/dev/null 2>&1; then
  no "python3 not found -- install Python 3.11 or newer, then re-run"
  exit 1
fi
PYV=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
if python3 -c 'import sys;raise SystemExit(0 if sys.version_info>=(3,11) else 1)'; then
  ok "python $PYV"
else
  no "python $PYV is too old; 3.11+ required"; exit 1
fi

# --------------------------------------------------------------- 2. system deps
b "2/5  System packages"
MISSING=()
py_has() { python3 -c "import $1" >/dev/null 2>&1; }
if [[ "$(uname)" == "Linux" ]]; then
  py_has gi   || MISSING+=(python-gobject)
  command -v wl-copy >/dev/null 2>&1 || command -v xclip >/dev/null 2>&1 || MISSING+=(wl-clipboard)
  # tesseract AND its language data: the binary alone silently fails every read, which a
  # number probe would report as a real value of 0.
  command -v tesseract >/dev/null 2>&1 && tesseract --list-langs 2>&1 | grep -qw eng \
    || MISSING+=(tesseract tesseract-data-eng)
  python3 -c "import gi;gi.require_version('Gst','1.0');from gi.repository import Gst" 2>/dev/null \
    || MISSING+=(gst-plugins-base gst-plugin-pipewire)
  if [[ ${#MISSING[@]} -eq 0 ]]; then
    ok "all present"
  else
    no "missing: ${MISSING[*]}"
    if command -v pacman >/dev/null 2>&1;      then SUDO+=("pacman -S --needed ${MISSING[*]}")
    elif command -v apt >/dev/null 2>&1;       then SUDO+=("apt install -y python3-gi gstreamer1.0-pipewire wl-clipboard tesseract-ocr tesseract-ocr-eng")
    elif command -v dnf >/dev/null 2>&1;       then SUDO+=("dnf install -y python3-gobject gstreamer1-plugins-base wl-clipboard tesseract tesseract-langpack-eng")
    fi
  fi
else
  ok "no system packages needed on $(uname)"
fi

# ------------------------------------------------------------------ 3. uinput
b "3/5  Input device"
if [[ "$(uname)" != "Linux" ]]; then
  ok "SendInput (no setup needed)"
elif [[ ! -e /dev/uinput ]]; then
  no "/dev/uinput missing -- kernel module not loaded"
  SUDO+=("modprobe uinput" "sh -c 'echo uinput > /etc/modules-load.d/uinput.conf'")
elif [[ -w /dev/uinput ]] && python3 - <<'PY' 2>/dev/null
import os,sys
try: os.close(os.open("/dev/uinput", os.O_WRONLY|os.O_NONBLOCK))
except OSError: sys.exit(1)
PY
then
  ok "/dev/uinput writable"
else
  no "/dev/uinput not usable by $USER"
  SUDO+=("usermod -aG input $USER   # then log out and back in")
fi

# -------------------------------------------------------------------- 4. venv
b "4/5  Python environment"
VENV_ARGS=(--system-site-packages)   # `gi` is a distro package; pip cannot provide it
[[ "$(uname)" != "Linux" ]] && VENV_ARGS=()
if [[ ! -d .venv ]]; then
  python3 -m venv "${VENV_ARGS[@]}" .venv || { no "venv creation failed"; exit 1; }
  ok "created .venv"
else
  ok ".venv exists"
fi
BIN=".venv/bin"; [[ -d ".venv/Scripts" ]] && BIN=".venv/Scripts"
"$BIN/pip" install -q --upgrade pip >/dev/null 2>&1
if "$BIN/pip" install -q -e . 2>&1 | tail -3; then ok "package installed"; else no "pip install failed"; exit 1; fi

# -------------------------------------------------------------------- 5. PATH
b "5/5  PATH"
mkdir -p "$HOME/.local/bin"
for n in voltage voltage-input-mcp; do
  [[ -f "$ROOT/$BIN/$n" ]] && ln -sf "$ROOT/$BIN/$n" "$HOME/.local/bin/$n"
done
if command -v voltage >/dev/null 2>&1; then
  ok "voltage is on your PATH"
else
  no "~/.local/bin is not on your PATH"
  case "${SHELL##*/}" in
    fish) cmd "fish_add_path ~/.local/bin" ;;
    zsh)  cmd 'echo '"'"'export PATH="$HOME/.local/bin:$PATH"'"'"' >> ~/.zshrc && exec zsh' ;;
    *)    cmd 'echo '"'"'export PATH="$HOME/.local/bin:$PATH"'"'"' >> ~/.bashrc && exec bash' ;;
  esac
fi

echo
if [[ ${#SUDO[@]} -gt 0 ]]; then
  b "Run these (they need root), then continue:"
  echo
  for c in "${SUDO[@]}"; do cmd "sudo $c"; done
  echo
fi

b "Next -- models and connecting to your AI client:"
echo
cmd "voltage setup"
echo
echo "  That downloads the models and registers the server. ~10-25 minutes,"
echo "  mostly download time. Then:"
echo
cmd "voltage"
echo
