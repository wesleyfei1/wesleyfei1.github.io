source "https://rubygems.org"

# 确保使用 Jekyll 3.7+ 版本
gem "jekyll", "~> 3.7"

# Minimal Mistakes 主题依赖
gem "minimal-mistakes-jekyll"

# 解决 Ruby 3.4+ 版本的兼容性问题
gem "base64" 
gem "bigdecimal" 
gem "tzinfo" # <--- 【请添加这一行】
# 【新增】解决 Markdown 解析器缺失问题
gem "kramdown-parser-gfm" # <--- 【请添加这一行】

group :jekyll_plugins do
  gem "jekyll-feed"
  gem "jekyll-sitemap"
  gem "jekyll-include-cache"
end