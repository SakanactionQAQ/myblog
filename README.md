# myblog

个人学习笔记与科研 idea 博客（Astro + GitHub Pages）。

## 在线发布

站点为静态部署，在线发帖通过管理后台把 Markdown 提交到本仓库，再由 GitHub Actions 自动构建上线。

1. 打开 [https://SakanactionQAQ.github.io/myblog/admin/](https://SakanactionQAQ.github.io/myblog/admin/)
2. 使用 GitHub Fine-grained Token 登录（仅本仓库，权限 `Contents: Read and write`）
3. 填写标题、分类、正文后点击发布
4. 等待 Actions 部署完成（约 1–2 分钟）即可在前台看到新文章

创建 Token：<https://github.com/settings/personal-access-tokens>

首次启用前请确认仓库已开启 Actions，并把本仓库的 GitHub Pages 来源设为 `gh-pages` 分支。

## 本地命令

| Command | Action |
| :------ | :----- |
| `npm install` | 安装依赖 |
| `npm run dev` | 本地开发 |
| `npm run build` | 构建到 `./dist/` |
| `npm run preview` | 预览构建结果 |
| `npm run deploy` | 手动部署到 gh-pages（一般不必，Actions 会自动部署） |
