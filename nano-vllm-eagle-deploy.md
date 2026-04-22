# 在 nano-vLLM 上部署 EAGLE v1

这篇文章的正式排版版已经改成 **GitHub Pages + HTML/CSS**：

- 本仓库页面入口：[index.html](./index.html)
- 启用 GitHub Pages 后访问：

```text
https://qiqi723.github.io/nano-VLLMEAGLEV1/
```

## 为什么不用 Markdown 直接做左右两栏？

GitHub 的 Markdown 渲染会过滤 `<style>` 等自定义 CSS。也就是说，本地 VSCode 能看到的两栏 CSS 布局，上传到 GitHub 的 `.md` 页面后可能不会生效。

所以现在的做法是：

- `nano-vllm-eagle-deploy.md`：作为 GitHub 仓库里的说明入口。
- `index.html`：作为真正的博客页面，使用完整 HTML/CSS 实现左右源码对照布局。

## 本文内容

HTML 页面包含：

- EAGLE v1 speculative decoding 工作流程图
- vanilla nano-vLLM decode 与 EAGLE decode 对比图
- 原始 nano-vLLM 与接入 EAGLE 后的左右代码对照
- `Config`、`Sampler`、`Qwen3Model`、`ModelRunner`、`Scheduler`、`BlockManager` 等关键修改点
- `bench_eagle.py` 的运行方式和对比输出

## 如何启用 GitHub Pages

在 GitHub 仓库页面进入：

```text
Settings -> Pages -> Build and deployment
```

选择：

```text
Source: Deploy from a branch
Branch: main
Folder: / (root)
```

保存后，等待 GitHub Pages 部署完成，然后访问：

```text
https://qiqi723.github.io/nano-VLLMEAGLEV1/
```
