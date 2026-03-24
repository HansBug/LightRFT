from typing import Dict

from lightrft.trainer.spmd_ppo_trainer import SPMDPPOTrainerVL


class MathPRMSPMDPPOTrainerVL(SPMDPPOTrainerVL):
    _ROLLOUT_KEY_SOURCES = {
        "reward": ("rollout_reward", "step_reward_mean", "reward"),
        "reward_std": ("rollout_reward_std", "step_reward_std"),
        "outcome_correct": ("rollout_outcome_correct", "outcome_correct_mean", "reward_metrics/outcome_correct"),
        "has_drop_moment": ("rollout_has_drop_moment", "has_drop_moment_mean", "reward_metrics/has_drop_moment"),
        "model_reward": ("rollout_model_reward", "model_reward_mean", "reward_metrics/model_reward"),
        "response_length": ("rollout_response_length", "response_length_mean", "response_length"),
    }
    _TRAIN_KEY_SOURCES = {
        "policy_loss": ("policy_loss",),
        "kl": ("kl",),
        "actor_lr": ("actor_lr",),
        "critic_loss": ("critic_loss",),
        "critic_lr": ("critic_lr",),
        "values": ("values",),
        "values_std": ("values_std",),
        "reward": ("reward",),
        "reward_std": ("step_reward_std",),
        "return": ("return",),
        "return_std": ("returns_std",),
        "response_length": ("response_length",),
        "total_length": ("total_length",),
        "num_actions": ("num_actions",),
        "approx_kl": ("approx_kl",),
        "clipfrac": ("clipfrac",),
        "ratio_mean": ("ratio_mean",),
        "ratio_max": ("ratio_max",),
        "advantages": ("advantages_mean",),
        "advantages_std": ("advantages_std",),
        "ptx_loss": ("ptx_loss",),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.define_metric("rollout/*", step_metric=None, step_sync=False, overwrite=True)
            self._wandb.define_metric("train/*", step_metric=None, step_sync=False, overwrite=True)

    def _build_rollout_metrics(self, logs_dict: Dict[str, float]) -> Dict[str, float]:
        rollout_metrics = {}
        for target_key, source_keys in self._ROLLOUT_KEY_SOURCES.items():
            for source_key in source_keys:
                if source_key in logs_dict:
                    rollout_metrics[target_key] = logs_dict[source_key]
                    break
        return rollout_metrics

    def _build_train_metrics(self, logs_dict: Dict[str, float]) -> Dict[str, float]:
        train_metrics = {}
        for target_key, source_keys in self._TRAIN_KEY_SOURCES.items():
            for source_key in source_keys:
                if source_key in logs_dict:
                    train_metrics[target_key] = logs_dict[source_key]
                    break
        return train_metrics

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}, episode=0):
        if global_step % args.logging_steps == 0:
            rollout_metrics = self._build_rollout_metrics(logs_dict)
            train_metrics = self._build_train_metrics(logs_dict)

            if self._wandb is not None and self.strategy.is_rank_0():
                all_wandb_logs = {}

                for key, value in rollout_metrics.items():
                    all_wandb_logs[f"rollout/{key}"] = value
                all_wandb_logs["rollout/episode"] = episode

                for key, value in train_metrics.items():
                    all_wandb_logs[f"train/{key}"] = value
                all_wandb_logs["train/episode"] = episode

                if all_wandb_logs:
                    self.wandb_log_counter += 1
                    self._wandb.log(all_wandb_logs, step=self.wandb_log_counter, commit=True)
                    self._update_wandb_summary(all_wandb_logs)

            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for key, value in rollout_metrics.items():
                    self._tensorboard.add_scalar(f"rollout/{key}", value, global_step)
                for key, value in train_metrics.items():
                    self._tensorboard.add_scalar(f"train/{key}", value, global_step)

        if global_step % args.eval_steps == 0 and self.eval_dataloader is not None:
            raw_eval_metrics = self.evaluate(self.eval_dataloader, global_step)

            if raw_eval_metrics and self.strategy.is_rank_0():
                self.eval_step_counter += 1

                if self._wandb is not None:
                    eval_logs = {}
                    for key, value in raw_eval_metrics.items():
                        clean_key = key.replace("eval_", "") if key.startswith("eval_") else key
                        eval_logs[f"eval/{clean_key}"] = value

                    eval_logs["eval/global_step"] = self.eval_step_counter
                    eval_logs["eval/train_step"] = global_step
                    eval_logs["eval/episode"] = episode

                    self.wandb_log_counter += 1
                    self._wandb.log(eval_logs, step=self.wandb_log_counter, commit=True)
                    self._update_wandb_summary(eval_logs)

                elif self._tensorboard is not None:
                    for key, value in raw_eval_metrics.items():
                        clean_key = key.replace("eval_", "") if key.startswith("eval_") else key
                        self._tensorboard.add_scalar(f"eval/{clean_key}", value, global_step)

        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def save_trajectories(self, global_step: int):
        if self.trajectory_saver is not None and self.replay_buffer.items:
            self.trajectory_saver.save_trajectories(
                experiences=self.replay_buffer.items,
                step=global_step,
                num_samples=self.num_trajectories_to_save,
                prefix="trajectories",
                compute_stats=self.args.trajectory_analysis,
            )
