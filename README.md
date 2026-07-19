# DQN Atari — Formative 3 Assignment (Deep Q-Learning)

Group project training and evaluating a Deep Q-Network (DQN) agent on an
Atari environment using Stable-Baselines3 and Gymnasium.

**Environment used:** `ALE/Pong-v5`

**Group members and areas of experimentation:**

| Member  | Focus                               | Experiments |
| ------- | ----------------------------------- | ----------- |
| Sharif  | Policy architecture + Learning Rate | Exp 01–10   |
| Sheryl  | Gamma (discount factor)             | Exp 11–20   |
| Miracle | Batch size + Exploration (epsilon)  | Exp 21–30   |

Each member ran 10 experiments individually. The best configuration from
each member's sweep will be combined into one final configuration, trained
for 1,000,000 timesteps, and used to produce the `play.py` gameplay demo
required for submission.

---

## Repository Structure

```
.
├── train.py              # CLI training script (DQN via Stable-Baselines3)
├── play.py               # Loads a trained model and runs it greedily
├──notebooks                # notebooks containing all experiments
├── README.md               # This file
├── models/                # Saved best model
├── logs/                  # TensorBoard event logs for best experiment
└── video/                # Gameplay recording from play.py (--record)
```

---

## Setup

```bash
pip install stable-baselines3[extra] gymnasium[atari] ale-py
```

This works identically on a local machine, Google Colab, and Kaggle.
The only difference across environments is that Colab and Kaggle reset
their Python environment each session, so this install step needs to be
re-run at the start of every fresh session; locally you only need it once.

### Running on each platform

**Local machine**

```bash
git clone <this-repo-url>
python train.py --experiment-name exp01 --policy CnnPolicy --lr 1e-4
```

**Google Colab**

```python
!git clone <this-repo-url>
!pip install stable-baselines3[extra] gymnasium[atari] ale-py
!python train.py --experiment-name exp01 --policy CnnPolicy --lr 1e-4
```

Colab sessions are ephemeral — mount Google Drive if you want results to
survive a disconnect:

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Kaggle**

```bash
!git clone <this-repo-url>
!pip install stable-baselines3[extra] gymnasium[atari] ale-py
!python train.py --experiment-name exp01 --policy CnnPolicy --lr 1e-4
```

Note: `train.py` currently writes all outputs (models/logs/checkpoints)
under `/kaggle/working/` by default. This is fine on Kaggle as long as you
**commit/save the notebook version** afterward — `/kaggle/working` is
otherwise wiped when the session ends. If running locally or on Colab,
either point Kaggle's path to your own Drive/local folder by editing
`base_dir` inside `train.py`, or add a `--base-dir` CLI argument so the
output location can be set per platform without editing the script.

### Playing back a trained model

`play.py` opens a live rendering window by default, which only works on a
machine with an actual display (i.e. locally). Colab and Kaggle are
headless, so any run there **must** use `--record` to save an MP4 instead
of trying to open a window:

```bash
# Local — live window
python play.py --policy cnn --model models/exp05/dqn_model.zip

# Colab / Kaggle — headless, must record
!python play.py --policy cnn --model models/exp05/dqn_model.zip --record
```

Recorded video is saved to `videos/dqn-play-*.mp4`.

---

## Sharif's Experiments — Policy Architecture & Learning Rate

Fixed across all runs: `env=ALE/Pong-v5`, `timesteps=200,000`, `seed=42`,
`gamma=0.99`, `batch_size=32`, `buffer_size=100,000`,
`learning_starts=20,000`, `train_freq=4`, `target_update_interval=10,000`,
`exploration 1.0 → 0.05` over the first 10% of training.

| Exp | Policy | Learning Rate | Final Reward | Best Reward | Behavior Noted                                                          |
| --- | ------ | ------------- | ------------ | ----------- | ----------------------------------------------------------------------- |
| 01  | Mlp    | 1e-4          | -20.90       | -20.10      | No real learning — MLP can't extract useful features from raw frames    |
| 02  | Cnn    | 1e-4          | -16.90       | -16.10      | Large jump vs MLP — CNN baseline established                            |
| 03  | Cnn    | 5e-5          | -19.60       | -17.80      | Worse than baseline — LR too conservative to converge in 200k steps     |
| 04  | Cnn    | 7.5e-5        | -18.10       | -17.96      | Better than 5e-5, still below the 1e-4 baseline                         |
| 05  | Cnn    | 1.5e-4        | -17.45       | -16.66      | **Best result overall** — longest episode length too (best 1803 steps)  |
| 06  | Cnn    | 2.5e-4        | -20.97       | -20.60      | Collapsed — reward stuck at -21 the entire run                          |
| 07  | Cnn    | 5e-4          | -20.96       | -20.60      | Collapsed, same pattern as Exp 06                                       |
| 08  | Cnn    | 7.5e-4        | -20.98       | -20.60      | Collapsed                                                               |
| 09  | Cnn    | 1e-3          | -20.98       | -20.60      | Collapsed — confirms divergence at high LR is consistent, not noise     |
| 10  | Cnn    | 2e-5          | -20.15       | -20.09      | Too slow — starts learning around step 70k but can't catch up in budget |

**Key insight:** CNN policy is essential for a pixel-based Atari game — the
MLP baseline barely learns anything. Within the CNN policy, there is a
narrow usable learning-rate window of roughly **1e-4 to 1.5e-4**. Below
that, training is simply too slow to converge in 200k steps. Above ~2e-4,
training diverges almost immediately and gets stuck at the worst possible
reward for the rest of the run — consistent with DQN's known sensitivity
to learning rate destabilizing its bootstrapped Q-value targets.

**Best configuration found:** `policy=CnnPolicy, lr=1.5e-4`

---

## Sheryl's Experiments — Gamma (Discount Factor)

Fixed: same baseline hyperparameters as above, varying only `gamma`.

| Exp | Gamma (γ) | Final Reward | Best Reward | Behavior Noted                                                                                                     |
| --- | --------- | ------------ | ----------- | ------------------------------------------------------------------------------------------------------------------ |
| 11  | 0.90      | -16.00       | -15.10      | Reasonable result despite low gamma — Pong's short rallies don't need much long-horizon credit                     |
| 12  | 0.92      | -17.50       | -17.50      | Weakest of the low-gamma group — likely run noise, not a real gamma effect                                         |
| 13  | 0.94      | -16.60       | -15.10      | Recovered, matches Exp 11 — confirms Exp 12 was an outlier                                                         |
| 14  | 0.96      | -16.00       | -16.00      | Steady, in line with other mid-range gammas                                                                        |
| 15  | 0.97      | **-14.90**   | **-14.90**  | **Best result overall** — outperforms every other gamma tested, including the 0.99 baseline used in Sharif's sweep |
| 16  | 0.98      | -16.20       | -15.40      | Worse than 0.97 despite being the next step up — 0.97 looks like a local peak                                      |
| 17  | 0.995     | -18.20       | -18.20      | **Worst result overall** — sharp drop-off entering near-1 gamma territory                                          |
| 18  | 0.997     | -17.60       | -15.90      | Recovered slightly from 0.995, still one of the weaker results                                                     |
| 19  | 0.999     | -17.20       | -17.00      | Plateaus at the same weak level as 0.997 — no further degradation                                                  |
| 20  | 0.9995    | -16.70       | -15.60      | Best of the high-gamma group, still well short of the 0.97 peak                                                    |

**Key insight:** performance does **not** improve monotonically as gamma approaches 1 — it peaks around **gamma=0.97**, better than even the standard 0.99 value used as baseline, then drops sharply once gamma exceeds ~0.99. The viable window for this training budget is roughly **0.94–0.98**; gamma ≥0.995 consistently trains the least reliably, likely because near-1 discount factors need more than 200k steps to stabilize the value function — consistent with DQN's known sensitivity to how far ahead it bootstraps its Q-value targets.

**Best configuration found:** `gamma=0.97`

## Miracle's Experiments — Batch Size & Exploration

Fixed: same baseline hyperparameters as above, varying `batch_size` or
one exploration parameter (`eps_end`, `eps_fraction`, `eps_start`) at a time.

| Exp | Parameter Changed | Value | Final Reward | Best Reward | Behavior Noted |
| --- | ----------------- | ----- | ------------ | ----------- | -------------- |
| 21  | batch_size        | 16    | _pending_    | _pending_   | _pending_      |
| 22  | batch_size        | 64    | _pending_    | _pending_   | _pending_      |
| 23  | batch_size        | 128   | _pending_    | _pending_   | _pending_      |
| 24  | batch_size        | 256   | _pending_    | _pending_   | _pending_      |
| 25  | eps_end           | 0.01  | _pending_    | _pending_   | _pending_      |
| 26  | eps_end           | 0.10  | _pending_    | _pending_   | _pending_      |
| 27  | eps_fraction      | 0.05  | _pending_    | _pending_   | _pending_      |
| 28  | eps_fraction      | 0.20  | _pending_    | _pending_   | _pending_      |
| 29  | eps_start         | 0.90  | _pending_    | _pending_   | _pending_      |
| 30  | eps_start         | 0.80  | _pending_    | _pending_   | _pending_      |

**Best configuration found:** _to be filled in once Miracle's runs complete_

---

## Final Model

Once all three sweeps are complete, the best value found for each
hyperparameter (policy, learning rate, gamma, batch size, exploration
schedule) will be combined into a single final configuration and trained
for **1,000,000 timesteps** — a longer budget than any individual sweep run,
to give the best-known configuration room to fully converge.

**Final configuration:** _to be filled in after combining all three members' best results_

```bash
python train.py \
  --experiment-name final_model \
  --policy CnnPolicy \
  --lr <best_lr> \
  --gamma <best_gamma> \
  --batch-size <best_batch_size> \
  --eps-end <best_eps_end> \
  --eps-fraction <best_eps_fraction> \
  --eps-start <best_eps_start> \
  --timesteps 1000000
```

This final model's `dqn_model.zip` (or `best_model.zip`, whichever scores
higher under evaluation) is what will be loaded by `play.py` to produce the
gameplay video required for submission.

```bash
python play.py --policy cnn --model models/final_model/best_model.zip --record --episodes 5
```

---

## Reproducing Everything (from scratch)

1. Clone the repo and install dependencies (see **Setup** above).
2. Run each individual experiment exactly as logged in the tables above,
   e.g.:
   ```bash
   python train.py --experiment-name sharif_exp05_lr_1_5e4 --policy CnnPolicy \
     --timesteps 200000 --seed 42 --lr 1.5e-4 --gamma 0.99 --batch-size 32 \
     --buffer-size 100000 --learning-starts 20000 --train-freq 4 \
     --target-update 10000 --eps-start 1.0 --eps-end 0.05 --eps-fraction 0.10
   ```
   Every experiment writes its own `config.json` alongside its saved model,
   so exact hyperparameters used for any run are always recoverable from
   that file rather than relying on this README alone.
3. Inspect TensorBoard logs / reward curves per experiment from the
   `logs/<experiment-name>/` folder.
4. Once all three members' 10 experiments are done, compare all 30 results,
   pick the best value per hyperparameter, and run the **Final Model**
   command above with 1,000,000 timesteps.
5. Run `play.py --record` against the final model to generate the
   submission gameplay video.

---

## Assignment Requirements Checklist

- [x] `train.py` — trains a DQN agent, compares `MlpPolicy` vs `CnnPolicy`
- [x] `play.py` — loads a trained model, runs it greedily (`deterministic=True`), renders/records gameplay
- [x] Hyperparameter tuning across learning rate, gamma, batch size, and epsilon (10 experiments per member, 30 total)
- [ ] Final model trained for 1,000,000 timesteps using best combined config _(pending all members' results)_
- [ ] Gameplay video from `play.py` for submission _(pending final model)_
- [ ] Hyperparameter table and video added to this README _(gamma and batch/exploration tables to be filled in)_
- [ ] Group presentation slot booked with the Coach
