"""
SkillRL Portability Experiment - API版本
=========================================
基于 examples/prompt_agent/gpt4o_alfworld.py 最小改动。
无需GPU，通过API调用任意模型。

实验条件：
  C3: LLaMA via API  + skill  → 核心迁移实验
  C4: LLaMA via API  + 无skill → baseline
  C5: GPT-4o         + skill  → 强模型参考

典型用法：
  # C3: LLaMA-3.1-8B via Together AI + skill
  python portability_exp.py \\
      --provider together \\
      --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \\
      --condition_name C3_llama_skill \\
      --use_skills

  # C4: 同模型，无skill
  python portability_exp.py \\
      --provider together \\
      --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \\
      --condition_name C4_llama_noskill

  # 汇总结果
  python portability_exp.py --summarize_only
"""

# 标准库
import os
import sys
import json
import time
import logging
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# 计算项目根目录
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LOGS_DIR = ROOT / "logs" / "portability"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "portability"

# 第三方库
import numpy as np
from openai import OpenAI
from omegaconf import OmegaConf

# 项目内部模块
from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld import (
    alfworld_projection,
    build_alfworld_envs,
)

# ──────────────────────────────────────────────────────────
# Provider配置
# ──────────────────────────────────────────────────────────

PROVIDER_CONFIGS = {
    "openkey":      {"base_url": "https://openkey.cloud/v1",
                    "api_key_env": "OPENKEY_API_KEY"},
    "poxie":        {"base_url": "https://api.poixe.com/v1",
                    "api_key_env": "POXIE_API_KEY"}
}

# ──────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────

class Agent:
    """LLM代理：通过API调用模型生成动作。
    
    支持多个Provider（如Together AI、OpenAI等），通过environment variable配置API密钥。
    支持处理thinking模式（如Qwen3），自动关闭thinking以避免action提取干扰。
    """
    
    def __init__(self, model_name: str, provider: str = "openai",
                 temperature: float = 0.0):
        """初始化Agent。
        
        Args:
            model_name: 模型名称（如'gpt-4o'、'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'）
            provider: API提供商（如'openai'、'together'）
            temperature: 采样温度，控制输出随机性（0=确定性）
        
        Raises:
            ValueError: 如果provider不在PROVIDER_CONFIGS中
            EnvironmentError: 如果API密钥环境变量未设置
        """
        if provider not in PROVIDER_CONFIGS:
            raise ValueError(f"Unknown provider '{provider}'. "
                             f"Options: {list(PROVIDER_CONFIGS.keys())}")
        cfg = PROVIDER_CONFIGS[provider]
        api_key = os.environ.get(cfg["api_key_env"])
        if not api_key:
            raise EnvironmentError(
                f"Please set env var: {cfg['api_key_env']}")
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
        logging.info(f"Agent: {model_name} | provider={provider}")

    def get_action_from_model(self, obs: str) -> tuple:
        """从观察状态生成动作。
        
        使用指数退避策略重试（最多5次），处理临时API故障。
        对于Qwen3系列模型，自动关闭thinking模式以减少token消耗。
        
        Args:
            obs: 环境观察状态文本
        
        Returns:
            tuple: (extracted_action, raw_response)
                - extracted_action: 解析后的动作字符串
                - raw_response: 模型原始输出
        """
        for attempt in range(5):
            try:
                # Qwen3系列需要关掉thinking模式，否则<think>块会干扰action提取
                extra = {}
                if "qwen3" in self.model_name.lower():
                    extra["extra_body"] = {"enable_thinking": False}

                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": obs}],
                    temperature=self.temperature,
                    max_tokens=512,  # 避免thinking内容截断action部分
                    **extra,
                )
                raw = resp.choices[0].message.content or ""
                action = self._extract_action(raw)
                return action, raw
            except Exception as e:
                wait = 2 ** attempt
                logging.warning(f"API error ({attempt+1}/5): {e} — retry in {wait}s")
                time.sleep(wait)
        logging.error("All retries failed.")
        return "", ""

    @staticmethod
    def _extract_action(raw: str) -> str:
        """从模型输出中提取动作，处理所有常见格式。
        
        支持的输出格式：
          1. <think>...</think><action>动作</action>  （thinking模式完整输出）
          2. </think>\n<action>动作</action>          （thinking被max_tokens截断）
          3. <action>动作</action>                    （只有action标签）
          4. Action: 动作                             （带前缀的纯文本）
          5. 动作                                     （纯文本无标签）
        
        该方法按优先级依次尝试提取，确保在各种模型输出格式下都能正确解析。
        
        Args:
            raw: 模型的原始输出字符串
        
        Returns:
            str: 解析后的动作字符串
        """
        text = raw.strip()

        # 情况1&2：清理所有thinking残留（不管<think>有没有开头）
        if "</think>" in text:
            text = text[text.rfind("</think>") + len("</think>"):].strip()

        # 情况1&2&3：提取<action>标签内容
        if "<action>" in text:
            start = text.find("<action>") + len("<action>")
            end = text.find("</action>", start)
            if end != -1:
                # 标签完整，直接返回内容
                return text[start:end].strip()
            else:
                # </action>被截断了，取<action>后面的内容
                return text[start:].strip().split("\n")[0].strip()

        # 情况4：去掉 "Action:" 前缀
        for prefix in ["action:", "action :", "next action:"]:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
                break

        # 情况5：去掉引号，只取第一行
        return text.strip("\"'`").split("\n")[0].strip()

# ──────────────────────────────────────────────────────────
# SkillBank
# ──────────────────────────────────────────────────────────

class SkillBank:
    """技能库：根据任务类型检索相关的Agent经验与指导原则。
    
    将skills组织为两类：
    - general_skills: 所有任务都适用的通用技能
    - task_specific_skills: 特定任务类型的技能
    """
    
    # 任务类型关键词映射：用于匹配任务描述并检索相应技能
    TASK_KEYWORDS = {
        "pick_and_place":                 ["put", "place"],
        "pick_two_obj_and_place":         ["two", "both"],
        "look_at_obj_in_light":           ["examine", "look at", "light"],
        "pick_heat_then_place_in_recep":  ["heat", "warm", "microwave"],
        "pick_cool_then_place_in_recep":  ["cool", "fridge", "refrigerator"],
        "pick_clean_then_place_in_recep": ["clean", "wash"],
    }

    def __init__(self, path: str):
        """从JSON文件加载技能库。
        
        Args:
            path: JSON技能文件路径
        """
        with open(path) as f:
            raw = json.load(f)
        self.general = raw.get("general_skills", raw.get("general", []))
        if "task_specific_skills" in raw:
            self.task_specific = raw["task_specific_skills"]
        else:
            self.task_specific = {t: raw[t] for t in self.TASK_KEYWORDS if t in raw}
        logging.info(f"SkillBank: {len(self.general)} general + "
                     f"{sum(len(v) for v in self.task_specific.values())} task-specific")

    def retrieve(self, task_desc: str) -> str:
        """根据任务描述检索相关技能。
        
        匹配流程：
        1. 在任务描述中查找关键词，确定任务类型
        2. 获取所有通用技能
        3. 如果匹配到任务类型，添加该类型的特定技能
        4. 格式化为Markdown文本，作为Agent的context注入
        
        Args:
            task_desc: 任务描述文本
        
        Returns:
            str: 格式化的技能文本（Markdown），如无匹配返回空字符串
        """
        # 根据关键词匹配任务类型
        task_type = next(
            (t for t, kws in self.TASK_KEYWORDS.items()
             if any(k in task_desc.lower() for k in kws)),
            None
        )
        
        # 收集通用技能 + 任务特定技能
        skills = list(self.general)
        if task_type and task_type in self.task_specific:
            skills += self.task_specific[task_type]
        
        if not skills:
            return ""
        
        # 格式化为Markdown文本
        lines = ["## Relevant Experience\n"]
        for s in skills:
            if "title"         in s: lines.append(f"### {s['title']}")
            if "principle"     in s: lines.append(f"Principle: {s['principle']}")
            if "when_to_apply" in s: lines.append(f"When to apply: {s['when_to_apply']}")
            lines.append("")
        
        return "\n".join(lines)

# ──────────────────────────────────────────────────────────
# 环境构建
# ──────────────────────────────────────────────────────────

def build_env(env_num: int) -> AlfWorldEnvironmentManager:
    """构建AlfWorld环境管理器。
    
    Args:
        env_num: 并行环境数量
    
    Returns:
        AlfWorldEnvironmentManager: 配置好的环境管理器实例
    """
    # 加载AlfWorld配置文件
    alf_config_path = str(
        ROOT / "agent_system" / "environments" /
        "env_package" / "alfworld" / "configs" / "config_tw.yaml"
    )
    
    # 创建并行环境
    envs = build_alfworld_envs(
        alf_config_path, seed=1, env_num=env_num, group_n=1,
        is_train=False,
        env_kwargs={"eval_dataset": "eval_in_distribution"},
        resources_per_worker={"num_cpus": 0.05, "num_gpus": 0.0},
    )
    
    # 环境配置：关闭内部memory系统，skill由agent层注入
    config = OmegaConf.create({
        "env": {
            "history_length": 5,
            "use_skills_only_memory": False,  # skill由agent层注入
            "use_retrieval_memory": False,
        }
    })
    
    return AlfWorldEnvironmentManager(envs, alfworld_projection, config)

# ──────────────────────────────────────────────────────────
# 主循环
# ──────────────────────────────────────────────────────────

TASKS = [
    "pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light",
    "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

def run(args) -> dict:
    """运行迁移性实验的主循环。
    
    实验流程：
    1. 初始化Agent、SkillBank、环境
    2. 运行多个trial，每个trial包含多个episode
    3. 每个episode中：
       - 注入skill（如果启用）到观察状态
       - 并发调用Agent生成动作
       - 记录轨迹用于后续分析
    4. 汇总结果并保存
    
    Args:
        args: 命令行参数对象
    
    Returns:
        dict: 实验结果（成功率统计、任务级别指标等）
    """
    # 初始化组件
    agent      = Agent(args.model, args.provider)
    skill_bank = SkillBank(args.skills_json) if args.use_skills else None
    env_manager = build_env(args.env_num)
 
    all_sr, task_history = [], defaultdict(list)
    # 所有episode的完整轨迹，用于后续分析
    all_trajectories = []
 
    for trial in range(args.test_times):
        logging.info(f"\n===== Trial {trial+1}/{args.test_times} =====")
        t0 = time.time()
 
        # 重置环境
        obs, infos  = env_manager.reset(kwargs={})
        dones       = [False] * args.env_num
        success     = np.zeros(args.env_num, dtype=bool)
        task_s, task_t = defaultdict(int), defaultdict(int)
        
        # 初始化轨迹记录（用于分析每个episode的决策过程）
        trajectories = []
        for i in range(args.env_num):
            task_desc = (env_manager.tasks[i]
                         if hasattr(env_manager, "tasks") and i < len(env_manager.tasks)
                         else "")
            task_type = next((t for t in TASKS if t in
                              infos[i].get("extra.gamefile", "")), "other") if infos else "other"
            trajectories.append({
                "env_idx":    i,
                "task_desc":  task_desc,
                "task_type":  task_type,
                "gamefile":   infos[i].get("extra.gamefile", "") if infos else "",
                "use_skills": args.use_skills,
                "condition":  args.condition_name,
                "steps":      [],   # 每一步的详细记录
                "success":    False,
                "total_steps": 0,
            })

        DEBUG_STEPS = 3
        DEBUG_ENV_IDX = 0

        for step in range(args.max_steps):
            logging.info(f"  step={step:3d} done={sum(dones)}/{args.env_num} "
                         f"sr={success.mean():.3f}")
 
            # 构建每个环境的obs文本（含skill注入），同时记录注入的skill
            obs_texts = []
            injected_skills = [""] * args.env_num
            raw_obs_texts   = [""] * args.env_num
            for i in range(args.env_num):
                if dones[i]:
                    obs_texts.append(None); continue
                # 这里直接使用原始obs文本，后续分析用于对比skill注入前后的差异
                # obs_text = obs["text"][i]
                raw_obs = obs["text"][i]
                raw_obs_texts[i] = raw_obs
                skill_text = ""
                if skill_bank is not None:
                    task_desc = (env_manager.tasks[i]
                                 if hasattr(env_manager, "tasks") and i < len(env_manager.tasks)
                                 else raw_obs)
                    # s = skill_bank.retrieve(task_desc)
                    skill_text = skill_bank.retrieve(task_desc)
                    # 如果有技能文本，就放在obs前面，分两段；否则直接用原始obs
                    if skill_text:
                        obs_text = skill_text + "\n\n" + raw_obs
                    else:
                        obs_text = raw_obs
                else:
                    obs_text = raw_obs
                # 记录注入的技能文本（如果有的话），用于后续分析
                injected_skills[i] = skill_text
                obs_texts.append(obs_text)

            # ===== DEBUG: 打印 obs =====
            if step < DEBUG_STEPS:
                debug_obs = obs_texts[DEBUG_ENV_IDX]
                print("\n" + "="*40)
                print(f"[DEBUG] Step {step} | Env {DEBUG_ENV_IDX} OBS:")
                print(debug_obs)  # 打印全部，不截断
                print("="*40)

            # ===== 阶段2：并发调用API生成动作 =====
            from concurrent.futures import ThreadPoolExecutor, as_completed
            actions = ["None"] * args.env_num
            raw_outputs = [""] * args.env_num  # 接收 ra
            active = [(i, obs_texts[i]) for i in range(args.env_num) if not dones[i]]
            with ThreadPoolExecutor(max_workers=min(len(active), 20)) as executor:
                future_to_idx = {
                    executor.submit(agent.get_action_from_model, txt): idx
                    for idx, txt in active
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        # actions[idx] = future.result()
                        action, raw = future.result()
                        actions[idx] = action
                        raw_outputs[idx] = raw
                    except Exception as e:
                        logging.warning(f"  env {idx} API failed: {e}")
                        actions[idx] = "look"  # fallback

            # ===== DEBUG: 打印模型输出 =====
            if step < DEBUG_STEPS:
                print(f"\n[DEBUG] Step {step} | Env {DEBUG_ENV_IDX} MODEL OUTPUT:")
                print("RAW:")
                print(raw_outputs[DEBUG_ENV_IDX][:1000])
                print("\nPARSED ACTION:")
                print(actions[DEBUG_ENV_IDX])
            
            # ===== 阶段3：环境步进与轨迹记录 =====
            obs, rewards, new_dones, infos = env_manager.step(actions)
 
            for i in range(args.env_num):
                if dones[i]:
                    continue

                # 记录这一步的详细信息
                reward_value = 0.0 if rewards is None else rewards[i]
                step_record = {
                    "step":           step,
                    "raw_obs":        raw_obs_texts[i],
                    "injected_skill": injected_skills[i],
                    "raw_output":     raw_outputs[i],
                    "action":         actions[i],
                    "reward":         float(reward_value),
                    "done":           bool(new_dones[i]),
                    # 关键字段：action是否在admissible列表里
                    "action_valid":   bool(infos[i].get("is_action_valid",
                                          actions[i] in str(infos[i].get("admissible_commands", "")))),
                }
                trajectories[i]["steps"].append(step_record)

                # 如果episode结束，更新统计信息
                if not new_dones[i]:
                    continue
                
                dones[i] = True
                won = bool(infos[i].get("won", False))
                success[i] = won
                trajectories[i]["success"]     = won
                trajectories[i]["total_steps"] = step + 1
                
                # 从gamefile确定准确的任务类型
                gf = infos[i].get("extra.gamefile", "")
                trajectories[i]["gamefile"]  = gf
                trajectories[i]["task_type"] = next((t for t in TASKS if t in gf), "other")
                
                # 更新任务统计
                matched = trajectories[i]["task_type"]
                task_t[matched] += 1
                if won:
                    task_s[matched] += 1
 
            if all(dones):
                break
 
        # ===== 统计单个trial的结果 =====
        sr = float(success.mean())
        all_sr.append(sr)
        all_trajectories.extend(trajectories)
        logging.info(f"Trial {trial+1} SR={sr:.4f}  ({time.time()-t0:.0f}s)")
        
        # 按任务类型输出成功率
        for t in TASKS:
            if task_t.get(t, 0) > 0:
                r = task_s[t] / task_t[t]
                task_history[t].append(r)
                logging.info(f"  {t:<40}: {r:.4f} ({task_s[t]}/{task_t[t]})")
 
    # ===== 汇总实验结果 =====
    result = {
        "condition":       args.condition_name,
        "model":           args.model,
        "provider":        args.provider,
        "use_skills":      args.use_skills,
        "overall_sr_mean": float(np.mean(all_sr)),
        "overall_sr_std":  float(np.std(all_sr)),
        "per_run_sr":      all_sr,
        "per_task_mean":   {t: float(np.mean(v)) for t, v in task_history.items()},
    }
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    out = Path(args.output_dir) / f"{args.condition_name}.json"
    out.write_text(json.dumps(result, indent=2))

    # 保存完整轨迹（可选，用于负迁移分析）
    if args.save_trajectories:
        traj_out = Path(args.output_dir) / f"{args.condition_name}_trajectories.json"
        traj_out.write_text(json.dumps(all_trajectories, indent=2, ensure_ascii=False))
        logging.info(f"Trajectories saved → {traj_out} "
                     f"({len(all_trajectories)} episodes)")

    logging.info(f"[{args.condition_name}] SR={result['overall_sr_mean']:.4f} → {out}")
    return result
 
# ──────────────────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────────────────
 
def summarize(output_dir: str) -> None:
    """汇总并显示实验结果。
    
    从输出目录读取所有JSON结果文件，生成对比表格。
    
    Args:
        output_dir: 结果输出目录
    """
    # 加载所有结果文件
    results = {}
    for fp in Path(output_dir).glob("*.json"):
        d = json.loads(fp.read_text())
        results[d["condition"]] = d
    
    if not results:
        print("No results yet.")
        return
 
    # ===== 输出总体结果 =====
    print("\n" + "="*65)
    print("PORTABILITY RESULTS")
    print("="*65)
    for k, d in sorted(results.items()):
        flag = "✓ skill" if d["use_skills"] else "✗ no skill"
        print(f"  {k:<38} {d['overall_sr_mean']:>6.1%}  "
              f"[{d['model'][:22]} | {flag}]")
 
    # ===== 计算关键指标差异 =====
    def sr(k): return results[k]["overall_sr_mean"] if k in results else None
    print("\nKEY GAPS:")
    # 定义关键对比：技能收益、模型迁移性
    checks = [
        ("C3_llama_skill", "C4_llama_noskill", "Skill gain on LLaMA       (C3-C4)"),
        ("C3_llama_skill", "C5_gpt4o_skill",   "Portability gap LLaMA-GPT4o (C3-C5)"),
    ]
    for a, b, label in checks:
        a_sr = sr(a)
        b_sr = sr(b)
        if a_sr is not None and b_sr is not None:
            print(f"  {label}: {a_sr-b_sr:+.1%}")
 
    # ===== 输出任务级别对比 =====
    print("\nPER-TASK:")
    conds = sorted(results.keys())
    print(f"  {'Task':<40}" + "".join(f"{c[:12]:>13}" for c in conds))
    for t in TASKS:
        row = f"  {t:<40}"
        for c in conds:
            v = results[c]["per_task_mean"].get(t, float("nan"))
            row += f"  {v:>10.1%}"
        print(row)
    print("="*65)
 

# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ===== 日志配置 =====
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(
                str(LOGS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    
    # ===== 命令行参数配置 =====
    p = argparse.ArgumentParser(description="SkillRL迁移性实验 - API版本")
    
    # 模型配置
    p.add_argument("--provider",       default="together",
                   choices=list(PROVIDER_CONFIGS.keys()),
                   help="API提供商")
    p.add_argument("--model",          default="",
                   help="模型名称")
    p.add_argument("--condition_name", default="",
                   help="实验条件名称（如C3_llama_skill）")
    
    # Skill配置
    p.add_argument("--use_skills",     action="store_true",
                   help="是否注入skills")
    p.add_argument("--skills_json",
                   default=str(ROOT / "memory_data" / "alfworld" / "claude_style_skills.json"),
                   help="skills JSON文件路径")
    
    # 环境配置
    p.add_argument("--env_num",        type=int, default=134,
                   help="并行环境数")
    p.add_argument("--max_steps",      type=int, default=50,
                   help="每个episode最大步数")
    p.add_argument("--test_times",     type=int, default=1,
                   help="运行trials次数")
    
    # 输出配置
    p.add_argument("--output_dir",     default=str(DEFAULT_OUTPUT_DIR),
                   help="结果输出目录")
    p.add_argument("--save_trajectories",  action="store_true",
                   help="保存完整轨迹用于负迁移分析（~50-200MB）")
    
    # 模式
    p.add_argument("--summarize_only", action="store_true",
                   help="仅汇总结果，不运行实验")
    
    args = p.parse_args()
 
    # ===== 主程序逻辑 =====
    if args.summarize_only:
        # 仅汇总模式
        summarize(args.output_dir)
    else:
        # 运行实验
        if not args.model or not args.condition_name:
            p.error("--model and --condition_name are required")
        run(args)
        
        # 如果有多个结果，自动汇总
        if len(list(Path(args.output_dir).glob("*.json"))) > 1:
            summarize(args.output_dir)
 