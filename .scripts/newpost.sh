#!/bin/bash
# newpost - 创建新博客文章

set -e

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTENT_DIR="$PROJECT_ROOT"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查参数
if [ -z "$1" ]; then
  echo -e "${YELLOW}用法: ./.scripts/newpost.sh <post-name>${NC}"
  echo "示例: ./.scripts/newpost.sh reinforcement-learning"
  exit 1
fi

POST_NAME="$1"
POST_DIR="$CONTENT_DIR/$POST_NAME"

# 检查目录是否已存在
if [ -d "$POST_DIR" ]; then
  echo -e "${YELLOW}⚠️  目录已存在: $POST_DIR${NC}"
  read -p "是否继续? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# 创建目录
mkdir -p "$POST_DIR/figs"
echo -e "${GREEN}✓ 创建目录: $POST_DIR${NC}"
echo -e "${GREEN}✓ 创建图片目录: $POST_DIR/figs${NC}"

# 获取当前日期
CURRENT_DATE=$(date "+%d %B %Y")

# 读取模板并替换变量
TITLE_FORMATED=$(echo "$POST_NAME" | sed 's/-/ /g' | sed 's/\b\(.\)/\u\1/g')
sed -e "s/{{TITLE}}/$TITLE_FORMATED/g" \
    -e "s/{{DATE}}/$CURRENT_DATE/g" \
    "$SCRIPT_DIR/post_template.md" > "$POST_DIR/index.md"

echo -e "${GREEN}✓ 创建文章: $POST_DIR/index.md${NC}"

# 打印成功信息
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 新文章创建成功！${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  文章路径: ${BLUE}$POST_DIR/index.md${NC}"
echo -e "  图片目录: ${BLUE}$POST_DIR/figs${NC}"
echo ""
echo -e "${YELLOW}下一步:${NC}"
echo -e "  1. 编辑文章: ${BLUE}code $POST_DIR/index.md${NC}"
echo -e "  2. 添加图片到: ${BLUE}$POST_DIR/figs${NC}"
echo ""

code $POST_DIR/index.md