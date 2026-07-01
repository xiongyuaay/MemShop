# Retail THOR Demo To-do

## 单货架范围修改

- [x] 确认默认货架：`FloorPlan1` / `shelf_FloorPlan1_010` / `CounterTop|-00.08|+01.15|00.00`。
- [x] 添加单货架过滤测试，要求商品目录只保留配置货架上的商品。
- [x] 添加目录构建脚本测试，要求 `03_build_catalog.py` 支持构建阶段过滤到单货架。
- [x] 添加 `configs/single_shelf.yaml`，集中记录默认货架信息。
- [x] 添加 `retail_thor/single_shelf.py`，提供配置加载、校验和商品目录过滤。
- [x] 修改 `scripts/03_build_catalog.py`，默认应用单货架过滤，并提供 `--disable-single-shelf` 调试入口。
- [x] 修改商品目录生成逻辑，给商品补充 `source_receptacle_id`，便于追踪到真实 AI2-THOR receptacle。
- [x] 修改任务模板，去掉默认单货架中不存在的饮料任务，改为早餐商品和容器商品任务。
- [x] 更新 README，说明当前默认不是多货架场景，而是一个货架和该货架上的商品。
- [x] 运行新增测试，确认单货架过滤行为通过。
- [x] 重新生成 `data/product_catalog.json`，确认只包含默认货架商品。
- [x] 重新生成单货架 demo episodes，确认 episode 的 `product_catalog` 和 `shelf_graph` 只包含一个货架。
- [x] 运行数据验证脚本，确认生成结果符合 schema 和路径要求。
- [x] 运行完整测试集，确认修改没有破坏已有功能。

## 机器人货架前走动试运行

- [x] 启动 AI2-THOR `FloorPlan1`，加载默认单货架 `CounterTop|-00.08|+01.15|00.00`。
- [x] 自动探测货架周围可达点，并选择可见商品最多的起始侧。
- [x] 将 agent 放到货架前并朝向货架。
- [x] 执行 `MoveRight` / `MoveLeft` 离散移动，让 agent 在货架前横向走动。
- [x] 保存每一步 RGB 截图和标注截图到 `outputs/walk_shelf_demo/`。
- [x] 生成 `outputs/walk_shelf_demo/walk_shelf_demo.gif` 作为走动序列预览。
- [x] 生成 `outputs/walk_shelf_demo/walk_report.json`，记录每一步动作成功状态、agent 位姿和可见商品。
- [x] 检查位姿变化和截图，确认 agent 实际移动且每一步可见货架商品。

## VLM 导航 agent

- [x] 实现导航动作集合：`MoveAhead`、`RotateLeft`、`RotateRight`、`LookUp`、`LookDown`、`Done`。
- [x] 实现导航动作执行器，只允许上述导航动作进入 AI2-THOR step。
- [x] 实现 OpenAI VLM navigation brain，通过当前 RGB 观测、任务指令、历史动作和场景上下文生成下一步导航动作。
- [x] 使用结构化 JSON schema 限制 VLM 输出，只能返回允许的导航动作。
- [x] 添加 `scripts/08_vlm_navigate.py`，用于真实调用 OpenAI VLM 控制 AI2-THOR agent 导航。
- [x] 更新 `requirements.txt` 和 README，记录 `openai` 依赖和运行命令。
