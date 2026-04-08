#!/bin/bash
set -e

# =============================================================================
# mem-check.sh — Check system memory readiness for llama-turbo benchmarks
#
# Reports total/available RAM, top memory consumers, and checks whether
# the system has enough free memory for each benchmark tier.
#
# Tiers:
#   smoke    — stories15M (~15MB model), needs 1GB free
#   baseline — 7B Q4_0 (~3.5GB model), needs 8GB free
#   stress   — Mixtral-8x7B at 8K ctx (~24GB + KV), needs 24GB free
#
# Usage: ./scripts/mem-check.sh [--tier smoke|baseline|stress]
# =============================================================================

TIER="all"

while [[ $# -gt 0 ]]; do
  case $1 in
    --tier) TIER="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

OS="$(uname -s)"

# --- Gather memory info ---
if [ "$OS" = "Darwin" ]; then
  PAGE_SIZE=$(pagesize)
  FREE_PAGES=$(vm_stat | awk '/Pages free/ {gsub(/\./,""); print $3}')
  INACTIVE_PAGES=$(vm_stat | awk '/Pages inactive/ {gsub(/\./,""); print $3}')
  SPECULATIVE_PAGES=$(vm_stat | awk '/Pages speculative/ {gsub(/\./,""); print $3}')
  PURGEABLE_PAGES=$(vm_stat | awk '/Pages purgeable/ {gsub(/\./,""); print $3}')
  AVAILABLE_MB=$(( (FREE_PAGES + INACTIVE_PAGES) * PAGE_SIZE / 1024 / 1024 ))
  TOTAL_MB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 ))
  USED_MB=$(( TOTAL_MB - AVAILABLE_MB ))
elif [ "$OS" = "Linux" ]; then
  AVAILABLE_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  TOTAL_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
  USED_MB=$(( TOTAL_MB - AVAILABLE_MB ))
else
  echo "Unsupported platform: $OS"
  exit 1
fi

# --- Display summary ---
echo "============================================"
echo " Memory Report — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
echo "  Platform:   $OS / $(uname -m)"
echo "  Total RAM:  ${TOTAL_MB}MB ($(( TOTAL_MB / 1024 ))GB)"
echo "  Used:       ${USED_MB}MB ($(( USED_MB * 100 / TOTAL_MB ))%)"
echo "  Available:  ${AVAILABLE_MB}MB ($(( AVAILABLE_MB / 1024 ))GB)"
echo ""

# --- Top memory consumers ---
echo "  Top 10 memory consumers:"
echo "  -------------------------"
if [ "$OS" = "Darwin" ]; then
  ps -eo rss,comm | sort -k1 -rn | head -10 | while read rss comm; do
    mb=$(( rss / 1024 ))
    printf "    %6dMB  %s\n" "$mb" "$comm"
  done
else
  ps -eo rss,comm --sort=-rss | head -11 | tail -10 | while read rss comm; do
    mb=$(( rss / 1024 ))
    printf "    %6dMB  %s\n" "$mb" "$comm"
  done
fi
echo ""

# --- Benchmark tier readiness (bash 3 compatible — no associative arrays) ---
check_tier() {
  local name=$1
  local req=$2
  local model=$3

  if [ "$AVAILABLE_MB" -ge "$req" ]; then
    printf "  %-40s  READY  (%dMB >= %dMB)\n" "$name" "$AVAILABLE_MB" "$req"
    return 0
  else
    local deficit=$(( req - AVAILABLE_MB ))
    printf "  %-40s  NOT READY  (need %dMB more)\n" "$name" "$deficit"
    echo "    Model: $model"
    echo "    Hint: close heavy apps to free memory"
    return 1
  fi
}

echo "  Benchmark tier readiness:"
echo "  -------------------------"
FAIL=0
if [ "$TIER" = "all" ] || [ "$TIER" = "smoke" ]; then
  check_tier "Smoke (stories15M)" 1024 "stories15M-q4_0.gguf (~15MB)" || FAIL=1
fi
if [ "$TIER" = "all" ] || [ "$TIER" = "baseline" ]; then
  check_tier "Baseline (7B Q4_0)" 8192 "Llama-2-7B-Q4_0 (~3.5GB + KV cache)" || FAIL=1
fi
if [ "$TIER" = "all" ] || [ "$TIER" = "stress" ]; then
  check_tier "Stress (Mixtral-8x7B @ 8K ctx)" 24576 "Mixtral-8x7B-Q4_0 (~24GB + 8GB KV)" || FAIL=1
fi
echo ""

# --- Suggestions if memory is tight ---
if [ "$AVAILABLE_MB" -lt 8192 ]; then
  echo "  Suggestions to free memory:"
  if [ "$OS" = "Darwin" ]; then
    ps -eo rss,comm | sort -k1 -rn | while read rss comm; do
      mb=$(( rss / 1024 ))
      if [ "$mb" -gt 200 ]; then
        case "$comm" in
          *qemu*|*emulator*) echo "    - Android emulator (~${mb}MB) — quit if not needed" ;;
          *Docker*|*OrbStack*) echo "    - Docker/OrbStack (~${mb}MB) — stop containers or quit" ;;
          *Windsurf*|*Code*|*Cursor*) echo "    - IDE (~${mb}MB) — close extra windows/tabs" ;;
          *Chrome*|*Safari*|*Firefox*) echo "    - Browser (~${mb}MB) — close tabs" ;;
          *Slack*|*Teams*|*Discord*) echo "    - Chat app (~${mb}MB) — quit if not needed" ;;
        esac
      fi
    done | sort -u
  fi
  echo ""
fi

# --- Exit code: fail if a specific tier was requested and not ready ---
if [ "$TIER" != "all" ] && [ "$FAIL" -ne 0 ]; then
  echo "FAIL: Not enough memory for $TIER tier"
  exit 1
fi
