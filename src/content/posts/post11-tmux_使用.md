---
title: 'tmux 使用'
pubdate: 2026-08-10
category: '科研工具'
---

tmux 是一个终端复用器，在一个窗口里开多个会话，保证 ssh 断了之后进程还可以继续。

```
session（会话）
	└── window（窗口，类似标签页）
		└── pane（窗格，窗口内分屏）
```

## 会话

| 操作         | 命令                                   |
| ---------- | ------------------------------------ |
| 新建会话       | `tmux` 或 `tmux new -s 名字`            |
| 列出会话       | `tmux ls`                            |
| 接入会话       | `tmux attach -t 名字` 或 `tmux a -t 名字` |
| 脱离会话（进程不停） | `Ctrl+b` 然后 `d`                      |
| 杀掉会话       | `tmux kill-session -t 名字`            |
| 重命名当前会话    | `Ctrl+b` 然后 `$`                      |
## 窗口

| 操作        | 按键（先 `Ctrl+b`） |
| --------- | -------------- |
| 新建窗口      | `c`            |
| 下一个 / 上一个 | `n` / `p`      |
| 按编号切换     | `0`–`9`        |
| 重命名窗口     | `,`            |
| 关闭窗口      | `&`            |
| 窗口列表      | `w`            |

## 窗格分屏

|操作|按键（先 `Ctrl+b`）|
|---|---|
|左右分屏|`%`|
|上下分屏|`"`|
|切换窗格|`方向键` 或 `o`|
|关闭当前窗格|`x`|
|放大/还原当前窗格|`z`|
|调整大小|`Ctrl+b` 后按住 `Ctrl` + 方向键|
|窗格变成独立窗口|`!`|
## 复制与滚动

| 操作          | 按键                  |
| ----------- | ------------------- |
| 进入复制/滚动模式   | `Ctrl+b` 然后 `[`     |
| 移动光标        | 方向键 / `PgUp` `PgDn` |
| 开始选择（vi 模式） | `Space`             |
| 复制          | `Enter`             |
| 粘贴          | `Ctrl+b` 然后 `]`     |
| 退出复制模式      | `q`                 |

```
tmux new -s dev # 新建并命名
tmux new -s dev -d # 后台新建，不进入
tmux rename-session -t old new
tmux kill-server # 关掉所有 tmux
tmux list-keys # 查看所有快捷键
tmux source-file ~/.tmux.conf # 重载配置
```

在 tmux 里也可以：`Ctrl+b` 然后 `:` 进入命令行，例如：

```
:new-window
:split-window -h
:kill-session
```

## 最小工作流
****
```
# 1. 开会话

tmux new -s proj

# 2. 分屏：Ctrl+b %（左右）、Ctrl+b "（上下）
# 3. 切窗格：Ctrl+b ←/→/↑/↓
# 4. 新窗口：Ctrl+b c ，切换：Ctrl+b n/p

# 5. 离开但保持运行
# Ctrl+b d

# 6. 回来
tmux a -t proj
```

## 实用配置（`~/.tmux.conf`）

```
# 把前缀改成 Ctrl+a（可选）
# unbind C-b
# set -g prefix C-a
# bind C-a send-prefix

set -g mouse on # 鼠标点选窗格、拖分界、滚轮
set -g history-limit 10000
set -g base-index 1 # 窗口从 1 编号
setw -g pane-base-index 1
setw -g mode-keys vi

# 更直观的分屏键
bind | split-window -h
bind - split-window -v
```
改完后：`tmux source-file ~/.tmux.conf`，或重开会话。
