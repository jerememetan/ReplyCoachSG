#!/bin/bash
# User-local Linux x86_64 installation; no root access or system service.
set -euo pipefail
install_dir="${OLLAMA_INSTALL_DIR:-$HOME/.local/share/replycoach-ollama}"
mkdir -p "$install_dir"
exec 9>"$install_dir/install.lock"
flock -n 9
if [ -x "$install_dir/bin/ollama" ]; then
  echo "Ollama is already installed at $install_dir"
  exit 0
fi
curl --fail --location --retry 3 --connect-timeout 15 \
  https://ollama.com/download/ollama-linux-amd64.tar.zst \
  --output "$install_dir/ollama-linux-amd64.tar.zst"
sha256sum "$install_dir/ollama-linux-amd64.tar.zst" > "$install_dir/download.sha256"
tar --zstd -xf "$install_dir/ollama-linux-amd64.tar.zst" -C "$install_dir"
test -x "$install_dir/bin/ollama"
echo "Installed Ollama at $install_dir"
