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

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# 计算项目根目录
ROOT = Path(__file__).resolve().parents
sys.path.insert(0, str(ROOT))
LOGS_DIR = ROOT / "logs" / "portability"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "portability"

from openai import OpenAI

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld import (
    alfworld_projection,
    build_alfworld_envs,
)
from omegaconf import OmegaConf

# ──────────────────────────────────────────────────────────
# Provider配置
# ──────────────────────────────────────────────────────────

PROVIDER_CONFIGS = {
    "openkey":      {"base_url": "https://openkey.cloud/v1",
                    "api_key_env": "OPENKEY_API_KEY"},
}

# ──────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────

class Agent:
    def __init__(self, model_name: str, provider: str = "openai",
                 temperature: float = 0.0):
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
                    max_tokens=512, # 避免thinking内容截断action部分
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
        """
        从模型输出中提取动作，处理所有常见格式：
          1. <think>...</think><action>动作</action>  （thinking模式完整输出）
          2. </think>\n<action>动作</action>          （thinking被max_tokens截断）
          3. <action>动作</action>                    （只有action标签）
          4. Action: 动作                             （带前缀）
          5. 动作                                     （纯文本）
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
    TASK_KEYWORDS = {
        "pick_and_place":                 ["put", "place"],
        "pick_two_obj_and_place":         ["two", "both"],
        "look_at_obj_in_light":           ["examine", "look at", "light"],
        "pick_heat_then_place_in_recep":  ["heat", "warm", "microwave"],
        "pick_cool_then_place_in_recep":  ["cool", "fridge", "refrigerator"],
        "pick_clean_then_place_in_recep": ["clean", "wash"],
    }

    def __init__(self, path: str):
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
        task_type = next(
            (t for t, kws in self.TASK_KEYWORDS.items()
             if any(k in task_desc.lower() for k in kws)),
            None
        )
        skills = list(self.general)
        if task_type and task_type in self.task_specific:
            skills += self.task_specific[task_type]
        if not skills:
            return ""
        lines = ["## Relevant Experience\n"]
        for s in skills:
            if "title"        in s: lines.append(f"### {s['title']}")
            if "principle"    in s: lines.append(f"Principle: {s['principle']}")
            if "when_to_apply"in s: lines.append(f"When to apply: {s['when_to_apply']}")
            lines.append("")
        return "\n".join(lines)

# ──────────────────────────────────────────────────────────
# 环境构建
# ──────────────────────────────────────────────────────────

def build_env(env_num: int):
    alf_config_path = str(
        ROOT / "agent_system" / "environments" /
        "env_package" / "alfworld" / "configs" / "config_tw.yaml"
    )
    envs = build_alfworld_envs(
        alf_config_path, seed=1, env_num=env_num, group_n=1,
        is_train=False,
        env_kwargs={"eval_dataset": "eval_in_distribution"},
        resources_per_worker={"num_cpus": 0.05, "num_gpus": 0.0},
    )
    config = OmegaConf.create({
        "env": {
            "history_length": 5,
            "use_skills_only_memory": False,   # skill由agent层注入
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

def run(args):
    agent      = Agent(args.model, args.provider)
    skill_bank = SkillBank(args.skills_json) if args.use_skills else None
    env_manager = build_env(args.env_num)
 
    all_sr, task_history = [], defaultdict(list)
 
    for trial in range(args.test_times):
        logging.info(f"\n===== Trial {trial+1}/{args.test_times} =====")
        t0 = time.time()
 
        obs, infos  = env_manager.reset(kwargs={})
        dones       = [False] * args.env_num
        success     = np.zeros(args.env_num, dtype=bool)
        task_s, task_t = defaultdict(int), defaultdict(int)
        
        DEBUG_STEPS = 3
        DEBUG_ENV_IDX = 0

        for step in range(args.max_steps):
            logging.info(f"  step={step:3d} done={sum(dones)}/{args.env_num} "
                         f"sr={success.mean():.3f}")
 
            # 构建每个环境的obs文本（含skill注入）
            obs_texts = []
            for i in range(args.env_num):
                if dones[i]:
                    obs_texts.append(None); continue
                obs_text = obs["text"][i]
                if skill_bank is not None:
                    task_desc = (env_manager.tasks[i]
                                 if hasattr(env_manager, "tasks") and i < len(env_manager.tasks)
                                 else obs_text)
                    s = skill_bank.retrieve(task_desc)
                    if s:
                        obs_text = s + "\n\n" + obs_text
                obs_texts.append(obs_text)

            # ===== DEBUG: 打印 obs =====
            if step < DEBUG_STEPS:
                debug_obs = obs_texts[DEBUG_ENV_IDX]
                print("\n" + "="*40)
                print(f"[DEBUG] Step {step} | Env {DEBUG_ENV_IDX} OBS:")
                # print(debug_obs[:2000])  
                print(debug_obs)  # 打印全部，不截断
                print("="*40)

            # 并发调用API（所有活跃环境同时发请求）
            from concurrent.futures import ThreadPoolExecutor, as_completed
            actions = ["None"] * args.env_num
            raw_outputs = [""] * args.env_num # 接收 ra
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
            obs, rewards, new_dones, infos = env_manager.step(actions)
 
            for i in range(args.env_num):
                if dones[i] or not new_dones[i]: continue
                dones[i] = True
                won = bool(infos[i].get("won", False))
                success[i] = won
                gf = infos[i].get("extra.gamefile", "")
                matched = next((t for t in TASKS if t in gf), "other")
                task_t[matched] += 1
                if won: task_s[matched] += 1
 
            if all(dones): break
 
        sr = float(success.mean())
        all_sr.append(sr)
        logging.info(f"Trial {trial+1} SR={sr:.4f}  ({time.time()-t0:.0f}s)")
        for t in TASKS:
            if task_t.get(t, 0) > 0:
                r = task_s[t] / task_t[t]
                task_history[t].append(r)
                logging.info(f"  {t:<40}: {r:.4f} ({task_s[t]}/{task_t[t]})")
 
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
    os.makedirs(args.output_dir, exist_ok=True)
    out = Path(args.output_dir) / f"{args.condition_name}.json"
    out.write_text(json.dumps(result, indent=2))
    logging.info(f"[{args.condition_name}] SR={result['overall_sr_mean']:.4f} → {out}")
    return result
 
# ──────────────────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────────────────
 
def summarize(output_dir: str):
    results = {}
    for fp in Path(output_dir).glob("*.json"):
        d = json.loads(fp.read_text())
        results[d["condition"]] = d
    if not results:
        print("No results yet."); return
 
    print("\n" + "="*65)
    print("PORTABILITY RESULTS")
    print("="*65)
    for k, d in sorted(results.items()):
        flag = "✓ skill" if d["use_skills"] else "✗ no skill"
        print(f"  {k:<38} {d['overall_sr_mean']:>6.1%}  "
              f"[{d['model'][:22]} | {flag}]")
 
    def sr(k): return results[k]["overall_sr_mean"] if k in results else None
    print("\nKEY GAPS:")
    checks = [
        ("C3_llama_skill", "C4_llama_noskill", "Skill gain on LLaMA       (C3-C4)"),
        ("C3_llama_skill", "C5_gpt4o_skill",   "Portability gap LLaMA-GPT4o (C3-C5)"),
    ]
    for a, b, label in checks:
        if sr(a) is not None and sr(b) is not None:
            print(f"  {label}: {sr(a)-sr(b):+.1%}")
 
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
    p = argparse.ArgumentParser()
    p.add_argument("--provider",       default="together",
                   choices=list(PROVIDER_CONFIGS.keys()))
    p.add_argument("--model",          default="")
    p.add_argument("--condition_name", default="")
    p.add_argument("--use_skills",     action="store_true")
    p.add_argument("--skills_json",
                   default=str(ROOT / "memory_data" / "alfworld" / "claude_style_skills.json"))
    p.add_argument("--env_num",        type=int, default=134)
    p.add_argument("--max_steps",      type=int, default=50)
    p.add_argument("--test_times",     type=int, default=1)
    p.add_argument("--output_dir",     default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--summarize_only", action="store_true")
    args = p.parse_args()
 
    if args.summarize_only:
        summarize(args.output_dir)
    else:
        if not args.model or not args.condition_name:
            p.error("--model and --condition_name are required")
        run(args)
        if len(list(Path(args.output_dir).glob("*.json"))) > 1:
            summarize(args.output_dir)
 