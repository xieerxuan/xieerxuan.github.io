#!/bin/bash

# 构建站点
bundle exec jekyll build

# 切换到 main 分支
git checkout main

#!/bin/bash

# 删除除 .git、当前目录和 _site 外的所有文件/文件夹
find . -maxdepth 1 ! -name '.git' ! -name '.' ! -name '_site' -exec rm -rf {} +

# 将 _site 内容移动到根目录
cp -r _site/* .

# 删除空的 _site 文件夹
rm -rf _site

# 创建 .nojekyll 文件（用于 GitHub Pages 绕过 Jekyll 处理）
touch .nojekyll

echo "部署清理完成！"