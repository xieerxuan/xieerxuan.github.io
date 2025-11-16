#!/bin/bash

# 构建站点
bundle exec jekyll build

# 切换到 main 分支
git checkout main

# 清空当前文件（除了 .git）
rm -rf !(".git"|"_site")
cp -r _site/* .
rm -rf _site

# 确保有 .nojekyll 文件
touch .nojekyll

# 提交并推送
git add -A
git commit -m "部署更新: $(date +%Y-%m-%d_%H:%M:%S)"
git push origin main

# 返回 source 分支
git checkout source