#!/usr/bin/env bash
# Set up VoltageInputMcp. Safe to re-run.
#
# Does the parts that need no privilege first and reports exactly what needs sudo, so you
# can read the commands before granting it.

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m %s\n' "$*"; }
fail() { printf '  \033[31mfail\033[0m %s\n' "$*"; }

NEED_SUDO=()

bold "VoltageInputMcp setup"
echo

# ---------------------------------------------------------------------------- session
bold "1. Session"
echo "   type=${XDG_SESSION_TYPE:-unknown}  desktop=${XDG_CURRENT_DESKTOP:-unknown}"
case "${XDG_SESSION_TYPE:-}" in
  wayland) ok "Wayland: input goes through /dev/uinput, below the compositor" ;;
  x11)     ok "X11: uinput works, and the x11 capture backend is available" ;;
  *)       warn "unrecognised session type; capture backend detection may need help" ;;
esac
echo

# ------------------------------------------------------------------------------ uinput
bold "2. Input device (/dev/uinput)"
if [[ ! -e /dev/uinput ]]; then
  fail "/dev/uinput does not exist -- the kernel module is not loaded"
  NEED_SUDO+=("modprobe uinput")
  NEED_SUDO+=("echo uinput > /etc/modules-load.d/uinput.conf   # load at boot")
elif [[ -w /dev/uinput ]]; then
  ok "writable -- no further permission work needed"
else
  fail "exists but is not writable by $USER"
  echo "     Two ways to fix it. The udev rule is preferred: it grants an ACL to"
  echo "     whoever is logged in at the seat, rather than making 'input' group"
  echo "     membership a permanent capability for your account."
  NEED_SUDO+=(
    "tee /etc/udev/rules.d/70-voltage-uinput.rules <<< 'KERNEL==\"uinput\", SUBSYSTEM==\"misc\", TAG+=\"uaccess\", OPTIONS+=\"static_node=uinput\"'"
    "udevadm control --reload-rules && udevadm trigger --name-match=uinput"
    "# or, the blunter option:  usermod -aG input $USER   (then log out and back in)"
  )
fi
echo

# --------------------------------------------------------------------------- packages
bold "3. System packages"
MISSING=()
have() { command -v "$1" >/dev/null 2>&1; }
py_has() { python3 -c "import $1" >/dev/null 2>&1; }

py_has gi        || MISSING+=(python-gobject)
py_has numpy     || MISSING+=(python-numpy)
py_has PIL       || MISSING+=(python-pillow)
have wl-copy     || MISSING+=(wl-clipboard)
py_has evdev     || MISSING+=(python-evdev)
have cmake       || MISSING+=(cmake)

if python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst" 2>/dev/null; then
  ok "GStreamer introspection present (portal/PipeWire capture available)"
else
  MISSING+=(gst-plugins-base gst-plugin-pipewire)
fi

if [[ ${#MISSING[@]} -eq 0 ]]; then
  ok "all system packages present"
else
  warn "missing: ${MISSING[*]}"
  if have pacman; then
    NEED_SUDO+=("pacman -S --needed ${MISSING[*]}")
  else
    echo "     install these with your distribution's package manager"
  fi
fi
echo "   note: python-gobject must come from the distro, not pip -- it has to match"
echo "         the system GLib/GStreamer ABI."
echo

# -------------------------------------------------------------------------------- venv
bold "4. Python environment"
if [[ ! -d .venv ]]; then
  # --system-site-packages is required: `gi` is a distro package and cannot be pip-installed.
  python3 -m venv --system-site-packages .venv || { fail "could not create venv"; exit 1; }
  ok "created .venv (with system site-packages, so 'gi' is importable)"
else
  ok ".venv exists"
fi
./.venv/bin/pip install -q --upgrade pip >/dev/null 2>&1
if ./.venv/bin/pip install -q -e . 2>&1 | tail -3; then
  ok "package installed in editable mode"
else
  fail "pip install failed -- see output above"
fi
echo

# ------------------------------------------------------------------------------ models
bold "5. Models"
if command -v llama-server >/dev/null 2>&1; then
  ok "llama-server on PATH (fast path: GBNF grammars, prompt caching)"
elif [[ -x "$ROOT/vendor/llama.cpp/build/bin/llama-server" ]]; then
  ok "llama-server built in vendor/ -- scripts/serve.sh will find it"
else
  warn "llama-server not found"
  echo "     fast path:      ./scripts/build-llama.sh      (needs cmake + CUDA)"
  echo "     zero-build path: set engine = \"ollama\" in voltage.toml, then"
  echo "                      ollama pull qwen2.5vl:3b && ollama pull qwen3:1.7b"
fi
command -v ollama >/dev/null 2>&1 && ok "ollama present (fallback engine available)"
echo

# ------------------------------------------------------------------------------- sudo
if [[ ${#NEED_SUDO[@]} -gt 0 ]]; then
  bold "Commands that need root -- read them, then run them:"
  echo
  for cmd in "${NEED_SUDO[@]}"; do
    if [[ "$cmd" == \#* ]]; then echo "    $cmd"; else echo "    sudo $cmd"; fi
  done
  echo
fi

bold "Next"
echo "    ./.venv/bin/voltage doctor        # what is still missing"
echo "    ./.venv/bin/voltage profiles      # which model profile fits this GPU"
echo "    ./scripts/fetch-models.sh lean    # download weights"
echo "    ./scripts/serve.sh lean           # start both model servers"
echo
echo "  Then register with Claude Code:"
echo "    claude mcp add voltage-input -- $ROOT/.venv/bin/voltage-input-mcp"
