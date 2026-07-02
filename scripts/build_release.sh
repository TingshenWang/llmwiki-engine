#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="$(awk -F '"' '/^version = / { print $2; exit }' pyproject.toml)"
if [[ -z "$VERSION" ]]; then
  echo "无法从 pyproject.toml 读取版本号。"
  exit 1
fi

PACKAGE_NAME="llmwiki-engine"
BUILD_DIR="$ROOT_DIR/dist/release-build"
PACKAGE_DIR="$BUILD_DIR/packages"
RELEASE_ROOT="$ROOT_DIR/dist/releases"
BUNDLE_NAME="${PACKAGE_NAME}-${VERSION}-macos-linux"
BUNDLE_DIR="$RELEASE_ROOT/$BUNDLE_NAME"
ARCHIVE_PATH="$RELEASE_ROOT/${BUNDLE_NAME}.tar.gz"

rm -rf "$BUILD_DIR" "$BUNDLE_DIR" "$ARCHIVE_PATH"
mkdir -p "$PACKAGE_DIR" "$BUNDLE_DIR/packages"

uv build --out-dir "$PACKAGE_DIR"

cp "$PACKAGE_DIR"/* "$BUNDLE_DIR/packages/"
cp "$ROOT_DIR/README.md" "$BUNDLE_DIR/README.md"
cp "$ROOT_DIR/README.zh-CN.md" "$BUNDLE_DIR/README.zh-CN.md"

cat > "$BUNDLE_DIR/install.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${1:-$HOME/.local/share/llmwiki-engine}"
BIN_DIR="${LLMWIKI_BIN_DIR:-$HOME/.local/bin}"
PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit("llmwiki-engine 需要 Python 3.11 或更高版本。")
PY

mkdir -p "$INSTALL_DIR" "$BIN_DIR"
"$PYTHON_BIN" -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/python" -m pip install --upgrade pip

WHEEL_PATH="$(find "$SCRIPT_DIR/packages" -maxdepth 1 -name 'llmwiki_engine-*.whl' | sort | tail -n 1)"
if [[ -z "$WHEEL_PATH" ]]; then
  echo "未找到 llmwiki_engine wheel。"
  exit 1
fi

"$INSTALL_DIR/.venv/bin/python" -m pip install "${WHEEL_PATH}[embeddings]"
ln -sf "$INSTALL_DIR/.venv/bin/llmwiki" "$BIN_DIR/llmwiki"

cat <<EOF
安装完成。

CLI: $BIN_DIR/llmwiki

如果 shell 找不到 llmwiki，请把下面这行加入 ~/.zshrc 或 ~/.bashrc：
  export PATH="$BIN_DIR:\$PATH"

首次使用 Qwen embedding 时，sentence-transformers 可能需要从 Hugging Face 下载模型；
如果本机已有缓存，会直接复用。
EOF
SH

chmod +x "$BUNDLE_DIR/install.sh"

cat > "$BUNDLE_DIR/README_RELEASE.md" <<EOF
# llmwiki-engine ${VERSION} release

这个包面向 macOS 和 Linux，包含：

- \`packages/\`：universal Python wheel 和 sdist。
- \`install.sh\`：创建本地 venv，并安装 \`llmwiki-engine[embeddings]\`。
- \`README.md\`：英文项目使用说明。
- \`README.zh-CN.md\`：中文项目使用说明。

## 安装

\`\`\`bash
tar -xzf ${BUNDLE_NAME}.tar.gz
cd ${BUNDLE_NAME}
./install.sh
llmwiki --help
\`\`\`

自定义安装目录：

\`\`\`bash
./install.sh /path/to/install/llmwiki-engine
\`\`\`

## Provider 配置

在 \`~/.llmwiki/config.yaml\` 配置真实 provider，例如：

\`\`\`yaml
providers:
  default:
    spec: openai_compatible:deepseek-v4-flash
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key: 你的 DeepSeek API Key
    max_tokens: 262144
    timeout_seconds: 300
\`\`\`

把真实 key 写在你自己的 \`~/.llmwiki/config.yaml\` 或 vault 的 \`.llmwiki/config.yaml\`；
不要把带 key 的配置文件提交到 Git。

## 基本使用

\`\`\`bash
llmwiki init /path/to/vault
llmwiki ingest run /path/to/vault example.md
llmwiki ingest status /path/to/vault --verify
\`\`\`
EOF

(
  cd "$BUNDLE_DIR"
  shasum -a 256 packages/* install.sh README.md README.zh-CN.md README_RELEASE.md > SHA256SUMS
)

tar -czf "$ARCHIVE_PATH" -C "$RELEASE_ROOT" "$BUNDLE_NAME"
shasum -a 256 "$ARCHIVE_PATH" > "${ARCHIVE_PATH}.sha256"

echo "Release bundle: $ARCHIVE_PATH"
echo "Checksum: ${ARCHIVE_PATH}.sha256"
