import argparse
import glob
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from board import Board
from env import TockEnv
from eval import benchmark, log_eval_result
from model import Agent, OBS_DIM, ACTION_DIM

N_STEPS       = 2048      # rollout steps per env per rollout
N_ENVS        = 8         # parallel environments
PPO_EPOCHS    = 4
N_MINIBATCHES = 4
CLIP_EPS      = 0.2
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
LR            = 2.5e-4
VALUE_COEF    = 0.5
ENTROPY_COEF  = 0.015
MAX_GRAD_NORM = 0.5
TOTAL_STEPS   = 10_000_000
CHECKPOINT_EVERY = 100     # rollouts

POOL_SIZE         = 10     # max frozen opponents to keep
POOL_UPDATE_EVERY = 20     # rollouts between pool additions / env recreations

RANDOM_OPP_FRACTION = 0.5  # probability each opponent slot uses random instead of pool


def make_env(opponent_weights1, opponent_weights2):
    def _make():
        return TockEnv(opponent_weights=opponent_weights1, opponent_weights2=opponent_weights2)
    return _make


def _sample_opponent(opponent_pool):
    if opponent_pool and random.random() >= RANDOM_OPP_FRACTION:
        return random.choice(opponent_pool)
    return None


def build_envs(vec_cls, n_envs, opponent_pool):
    factories = [
        make_env(_sample_opponent(opponent_pool), _sample_opponent(opponent_pool))
        for _ in range(n_envs)
    ]
    return vec_cls(factories)


def masks_from_infos(infos: dict, n_envs: int) -> torch.Tensor:
    raw = infos.get("action_mask")
    if raw is None:
        return torch.ones(n_envs, ACTION_DIM, dtype=torch.bool)
    return torch.from_numpy(np.asarray(raw, dtype=bool))


def eval_and_log(agent, device, global_step, n_games, log_path, seed=42):
    agent.eval()
    print(f"  → running eval ({n_games} games)...", flush=True)
    wins, p_value = benchmark(agent, Board(), device, n_games=n_games, seed_offset=seed)
    win_pct = 100 * wins[0] / n_games
    log_eval_result(log_path, global_step, win_pct, p_value)
    print(f"  → eval: {win_pct:.1f}% win rate  p={p_value:.4f}  → {log_path}", flush=True)
    agent.train()


def save_checkpoint(agent, opt, rollout, global_step, ckpt_dir, tag=None):
    label = tag if tag else f"step_{global_step:09d}"
    path  = os.path.join(ckpt_dir, f"ckpt_{label}.pt")
    torch.save({
        "agent":       agent.state_dict(),
        "optimizer":   opt.state_dict(),
        "rollout":     rollout,
        "global_step": global_step,
    }, path)
    latest = os.path.join(ckpt_dir, "latest.pt")
    torch.save({
        "agent":       agent.state_dict(),
        "optimizer":   opt.state_dict(),
        "rollout":     rollout,
        "global_step": global_step,
    }, latest)
    print(f"  → checkpoint saved: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps",  type=int,   default=TOTAL_STEPS)
    parser.add_argument("--n-envs",           type=int,   default=N_ENVS)
    parser.add_argument("--checkpoint-every", type=int,   default=CHECKPOINT_EVERY)
    parser.add_argument("--checkpoint-dir",   type=str,   default="checkpoints")
    parser.add_argument("--resume",           type=str,   default=None,
                        help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--device",           type=str,   default="auto",
                        help="cpu | cuda | mps | auto")
    parser.add_argument("--sync",             action="store_true",
                        help="Use SyncVectorEnv instead of AsyncVectorEnv (debugging)")
    parser.add_argument("--eval-games",       type=int,   default=500,
                        help="Games to run after each checkpoint (0 = disable)")
    parser.add_argument("--eval-log",         type=str,   default="eval_log.csv",
                        help="CSV file to append eval results to")
    return parser.parse_args()


def main():
    args   = parse_args()
    n_envs = args.n_envs

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    vec_cls = gym.vector.SyncVectorEnv if args.sync else gym.vector.AsyncVectorEnv

    batch_size     = N_STEPS * n_envs
    minibatch_size = batch_size // N_MINIBATCHES
    num_rollouts = args.total_timesteps // batch_size

    agent = Agent().to(device)
    opt   = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    start_rollout = 0
    global_step   = 0

    # Opponent pool: list of CPU state_dicts of past agent snapshots.
    opponent_pool: list[dict] = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        opt.load_state_dict(ckpt["optimizer"])
        start_rollout = ckpt.get("rollout", 0) + 1
        global_step   = ckpt.get("global_step", 0)
        print(f"Resumed from {args.resume}  (rollout {start_rollout}, step {global_step})")

        all_ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, "ckpt_step_*.pt")))
        if all_ckpts:
            step = max(1, len(all_ckpts) // POOL_SIZE)
            sampled = all_ckpts[::step][-POOL_SIZE:]
            for path in sampled:
                c = torch.load(path, map_location="cpu", weights_only=False)
                opponent_pool.append({k: v.clone().cpu() for k, v in c["agent"].items()})
            print(f"  → opponent pool seeded with {len(opponent_pool)} checkpoints"
                  f" (from {os.path.basename(sampled[0])} to {os.path.basename(sampled[-1])})")

    envs = build_envs(vec_cls, n_envs, opponent_pool)

    print(
        f"Device: {device}  |  Envs: {n_envs}  |  "
        f"Batch: {batch_size}  |  Action space: {ACTION_DIM} "
    )

    obs_buf     = torch.zeros(N_STEPS, n_envs, OBS_DIM,    dtype=torch.float32)
    mask_buf    = torch.zeros(N_STEPS, n_envs, ACTION_DIM, dtype=torch.bool)
    action_buf  = torch.zeros(N_STEPS, n_envs,             dtype=torch.long)
    logprob_buf = torch.zeros(N_STEPS, n_envs,             dtype=torch.float32)
    reward_buf  = torch.zeros(N_STEPS, n_envs,             dtype=torch.float32)
    done_buf    = torch.zeros(N_STEPS, n_envs,             dtype=torch.float32)
    value_buf   = torch.zeros(N_STEPS, n_envs,             dtype=torch.float32)

    ep_rewards: list[float] = []
    ep_lengths: list[int]   = []
    ep_reward_acc = np.zeros(n_envs, dtype=np.float32)
    ep_length_acc = np.zeros(n_envs, dtype=np.int32)

    obs_np, infos = envs.reset()
    obs   = torch.from_numpy(obs_np).float()
    masks = masks_from_infos(infos, n_envs)

    start_time         = time.time()
    session_start_step = global_step

    rollout = start_rollout - 1
    for rollout in range(start_rollout, num_rollouts):

        # --- LR schedule (linear decay to 0) ---
        frac = 1.0 - global_step / args.total_timesteps
        opt.param_groups[0]["lr"] = frac * LR

        # --- Rollout ---
        agent.eval()
        with torch.no_grad():
            for step in range(N_STEPS):
                obs_buf[step]  = obs
                mask_buf[step] = masks

                actions, log_probs, _, values = agent.get_action_and_value(
                    obs.to(device), masks.to(device)
                )
                action_buf[step]  = actions.cpu()
                logprob_buf[step] = log_probs.cpu()
                value_buf[step]   = values.squeeze(-1).cpu()

                obs_np, rewards, terminated, truncated, infos = envs.step(
                    actions.cpu().numpy()
                )
                dones = terminated | truncated

                reward_buf[step] = torch.from_numpy(rewards.astype(np.float32))
                done_buf[step]   = torch.from_numpy(dones.astype(np.float32))

                ep_reward_acc += rewards
                ep_length_acc += 1
                for i in range(n_envs):
                    if dones[i]:
                        ep_rewards.append(float(ep_reward_acc[i]))
                        ep_lengths.append(int(ep_length_acc[i]))
                        ep_reward_acc[i] = 0.0
                        ep_length_acc[i] = 0

                global_step += n_envs
                obs   = torch.from_numpy(obs_np).float()
                masks = masks_from_infos(infos, n_envs)

        # --- GAE ---
        advantages = torch.zeros(N_STEPS, n_envs)
        last_adv   = torch.zeros(n_envs)
        for t in reversed(range(N_STEPS)):
            if t == N_STEPS - 1:
                next_val  = torch.zeros(n_envs)
                next_done = torch.zeros(n_envs)
            else:
                next_val  = value_buf[t + 1]
                next_done = done_buf[t]

            delta    = reward_buf[t] + GAMMA * next_val * (1 - next_done) - value_buf[t]
            last_adv = delta + GAMMA * GAE_LAMBDA * (1 - next_done) * last_adv
            advantages[t] = last_adv

        returns = advantages + value_buf

        b_obs        = obs_buf.reshape(-1, OBS_DIM).to(device)
        b_mask       = mask_buf.reshape(-1, ACTION_DIM).to(device)
        b_actions    = action_buf.reshape(-1).to(device)
        b_logprobs   = logprob_buf.reshape(-1).to(device)
        b_returns    = returns.reshape(-1).to(device)
        b_adv        = advantages.reshape(-1).to(device)
        b_old_values = value_buf.reshape(-1).to(device)

        # --- PPO Update ---
        agent.train()
        indices = np.arange(batch_size)

        pg_losses:  list[float] = []
        v_losses:   list[float] = []
        ent_losses: list[float] = []
        approx_kls: list[float] = []

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb = torch.as_tensor(
                    indices[start : start + minibatch_size], dtype=torch.long, device=device
                )

                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb], b_mask[mb], b_actions[mb]
                )

                log_ratio = new_logprob - b_logprobs[mb]
                ratio     = log_ratio.exp()

                with torch.no_grad():
                    approx_kls.append(((ratio - 1) - log_ratio).mean().item())

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()
                v_pred   = new_value.squeeze()
                v_clipped = b_old_values[mb] + (v_pred - b_old_values[mb]).clamp(-CLIP_EPS, CLIP_EPS)
                v_loss   = 0.5 * torch.max(
                    (v_pred - b_returns[mb]).pow(2),
                    (v_clipped - b_returns[mb]).pow(2),
                ).mean()
                ent_loss = -entropy.mean()
                loss     = pg_loss + VALUE_COEF * v_loss + ENTROPY_COEF * ent_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                opt.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())

        sps            = (global_step - session_start_step) / (time.time() - start_time)
        mean_ep_reward = np.mean(ep_rewards[-20:]) if ep_rewards else float("nan")
        mean_ep_length = np.mean(ep_lengths[-20:]) if ep_lengths else float("nan")

        print(
            f"rollout={rollout:5d}  "
            f"steps={global_step:9d}  "
            f"sps={sps:6.0f}  "
            f"ep_rew={mean_ep_reward:+.3f}  "
            f"ep_len={mean_ep_length:6.1f}  "
            f"approx_kl={np.mean(approx_kls):.4f}  "
            f"entropy={-np.mean(ent_losses):.3f}  "
            f"pg_loss={np.mean(pg_losses):.4f}  "
            f"v_loss={np.mean(v_losses):.4f}  "
            f"pool={len(opponent_pool)}"
        )

        if (rollout + 1) % args.checkpoint_every == 0:
            save_checkpoint(agent, opt, rollout, global_step, args.checkpoint_dir)
            if args.eval_games > 0:
                eval_and_log(agent, device, global_step, args.eval_games, args.eval_log)

        # --- Opponent pool update ---
        if (rollout + 1) % POOL_UPDATE_EVERY == 0:
            pool_entry = {k: v.clone().cpu() for k, v in agent.state_dict().items()}
            opponent_pool.append(pool_entry)
            if len(opponent_pool) > POOL_SIZE:
                opponent_pool.pop(0)

            # Recreate envs with freshly sampled opponents from the pool.
            envs.close()
            envs = build_envs(vec_cls, n_envs, opponent_pool)
            obs_np, infos = envs.reset()
            obs   = torch.from_numpy(obs_np).float()
            masks = masks_from_infos(infos, n_envs)
            ep_reward_acc[:] = 0.0
            ep_length_acc[:] = 0

    envs.close()
    if rollout >= start_rollout:
        save_checkpoint(agent, opt, rollout, global_step, args.checkpoint_dir, tag="final")
    print("Training complete.")


if __name__ == "__main__":
    main()