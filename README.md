# SkillRL 可移植性实验

基于 SkillRL 框架的可移植性实验，使用 API 调用评估不同 LLM 的技能迁移能力。

## 实验目标

评估 LLM 是否能通过技能注入实现跨模型的可移植性，主要对比：
- 有技能 vs 无技能的性能差异
- 不同模型（如 LLaMA、GPT-4o）的迁移效果

## 快速开始

### 环境要求

- Python 3.8+
- 安装依赖：`pip install -r requirements.txt`

### 运行实验

```bash
# 基本运行
python portability_exp.py \
  --provider openkey \
  --model qwen3-8b \
  --condition_name test_run \
  --env_num 10 \
  --max_steps 20

# 启用技能注入
python portability_exp.py \
  --provider openkey \
  --model qwen3-8b \
  --condition_name test_with_skills \
  --use_skills \
  --env_num 10
```

#### 参数说明
- `--provider`: API 提供商 (openkey, poxie)
- `--model`: 模型名称
- `--use_skills`: 启用技能注入
- `--env_num`: 并行环境数 (默认 134)
- `--max_steps`: 每个 episode 最大步数 (默认 50)
- `--test_times`: 重复试验次数 (默认 3)
- `--save_trajectories`: 保存完整轨迹用于负迁移分析
- `--summarize_only`: 仅汇总已有结果，不运行实验

### 文件结构
```
.
├── portability_exp.py          # 主实验脚本
├── legacy/                     # 旧版本及修复前备份
├── patches/                    # 依赖修改补丁
├── agent_system/               # 环境系统依赖（forked SkillRL）
├── memory_data/                # 技能数据 (claude_style_skills.json)
└── README.md                   # 本文件
```
### 依赖修改
```bash
git apply patches/envs.patch
git apply patches/__init__.patch
```

## 结果管理
- 结果保存在 results/portability/
- 日志保存在 logs/portability/
- 这些目录已通过 .gitignore 排除上传

## 许可证
基于 Apache License 2.0，详见上游 SkillRL 项目。


