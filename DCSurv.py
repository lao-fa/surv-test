import copy
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EPS = 1e-8


CSV_PATH = r""
OUTPUT_DIR = r""
EXTERNAL_CSV_PATH = None  
TIME_COLUMN = ""
EVENT_COLUMN = ""
DROP_COLUMNS = [""]

SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
DEVICE_STR = "cuda"  

DIFFUSION_STEPS = 100
DIFFUSION_EPOCHS = 150
DIFFUSION_HIDDEN = 128
DIFFUSION_LR = 1e-3
DIFFUSION_BATCH_SIZE = 128
BETA_START = 1e-4
BETA_END = 2e-2

W_NOISE = 1.0
W_WASSERSTEIN = 0.1
W_TIME = 0.2
W_PARTIAL_DIFFUSION = 0.1
WASSERSTEIN_PROJECTIONS = 50

SYNTHETIC_MULTIPLIER = 1.0 
COX_PENALIZER = 0.1

RISK_EPOCHS = 250
RISK_HIDDEN = 64
RISK_LAYERS = 2
RISK_DROPOUT = 0.1
RISK_LR = 1e-3
RISK_WEIGHT_DECAY = 1e-5
RISK_EARLY_STOPPING = 30

SYNTHETIC_SAMPLE_WEIGHT = 0.3   
SYNTHETIC_RISK_PENALTY = 0.05  


IBS_POINTS = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_survival_csv(path, time_column, event_column, drop_columns):
    """Load a survival CSV and return (frame, times, events, feature_columns)."""
    frame = pd.read_csv(path)
    if time_column not in frame or event_column not in frame:
        raise ValueError(f"CSV must contain '{time_column}' and '{event_column}'")
    drop = set(drop_columns or []) | {time_column, event_column}
    feature_columns = [c for c in frame.columns if c not in drop]

    frame = frame.replace([np.inf, -np.inf], np.nan)
    valid = frame[time_column].notna() & frame[event_column].notna()
    frame = frame.loc[valid].reset_index(drop=True)

    y = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(float)
    e = pd.to_numeric(frame[event_column], errors="coerce").to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(e) & (y >= 0)
    frame = frame.loc[valid].reset_index(drop=True)
    y = y[valid].astype(np.float32)
    e = (e[valid] > 0).astype(np.float32)

    if len(np.unique(e)) < 2:
        raise ValueError("Data must contain both events and censored observations")
    return frame, y, e, feature_columns


def create_splits(n_samples, event, test_size, validation_size, seed):
    indices = np.arange(n_samples)
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=event
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=validation_size,
        random_state=seed + 1,
        stratify=event[train_val_idx],
    )
    return {"train": train_idx, "validation": val_idx, "test": test_idx}


class SurvivalPreprocessor:

    def __init__(self, columns):
        self.columns = list(columns)
        self.numeric = []
        self.categorical = []
        self.medians = {}
        self.categories = {}
        self.output_columns = []
        self.scaler = StandardScaler()

    def fit(self, frame):
        for col in self.columns:
            if pd.api.types.is_numeric_dtype(frame[col]):
                self.numeric.append(col)
                self.medians[col] = float(pd.to_numeric(frame[col], errors="coerce").median())
            else:
                self.categorical.append(col)
                values = frame[col].fillna("__MISSING__").astype(str)
                self.categories[col] = sorted(values.unique())
        raw = self._to_raw(frame)
        self.output_columns = list(raw.columns)
        self.scaler.fit(raw[self.output_columns].astype(float))
        return self

    def _to_raw(self, frame):
        result = pd.DataFrame(index=frame.index)
        for col in self.numeric:
            result[col] = pd.to_numeric(frame[col], errors="coerce").fillna(self.medians[col])
        for col in self.categorical:
            values = frame[col].fillna("__MISSING__").astype(str)
            for category in self.categories[col]:
                result[f"{col}={category}"] = (values == category).astype(float)
        return result

    def transform(self, frame):
        raw = self._to_raw(frame).reindex(columns=self.output_columns, fill_value=0.0)
        return self.scaler.transform(raw.astype(float)).astype(np.float32)


class CensoringModel:

    def __init__(self, penalizer=COX_PENALIZER):
        self.model = CoxPHFitter(penalizer=penalizer)
        self.columns = []
        self.times = None
        self.cumhaz = None
        self.max_time = 1.0

    def fit(self, x, y, delta):
        self.columns = [f"x{i}" for i in range(x.shape[1])]
        frame = pd.DataFrame(x, columns=self.columns)
        frame["time"] = y.astype(float)
        frame["censor_event"] = (1 - delta).astype(int)  # censoring process
        self.model.fit(frame, duration_col="time", event_col="censor_event")

        baseline = self.model.baseline_cumulative_hazard_
        self.times = baseline.index.to_numpy(float)
        self.cumhaz = baseline.iloc[:, 0].to_numpy(float)
        self.max_time = float(np.max(y))
        if len(self.times) == 0 or self.times[0] > 0.0:
            self.times = np.concatenate(([0.0], self.times))
            self.cumhaz = np.concatenate(([0.0], self.cumhaz))
        return self

    def linear_predictor(self, x):
        frame = pd.DataFrame(x, columns=self.columns)
        return np.log(np.asarray(self.model.predict_partial_hazard(frame)).reshape(-1))

    def survival(self, x, times):
        base = np.interp(np.maximum(times, 0.0), self.times, self.cumhaz,
                         left=0.0, right=float(self.cumhaz[-1]))
        return np.exp(-np.outer(np.exp(self.linear_predictor(x)), base)).clip(EPS, 1.0)

    def sample_times(self, x, rng):
        u = rng.uniform(EPS, 1.0 - EPS, len(x))
        target = -np.log(u) / np.exp(self.linear_predictor(x))
        return np.maximum(
            np.interp(target, self.cumhaz, self.times, left=0.0, right=self.max_time), 0.0
        )

def sliced_wasserstein_distance(x, y, num_projections=WASSERSTEIN_PROJECTIONS, p=2):

    if x.size(0) == 0 or y.size(0) == 0:
        return torch.tensor(0.0, device=x.device)

    n, d = x.shape
    m, _ = y.shape

    directions = torch.randn(num_projections, d, device=x.device)
    directions = directions / torch.norm(directions, dim=1, keepdim=True)

    distances = []
    for direction in directions:
        x_proj, _ = torch.sort(x @ direction)
        y_proj, _ = torch.sort(y @ direction)

        if n != m:
            if n < m:
                x_proj = F.interpolate(
                    x_proj.unsqueeze(0).unsqueeze(0), size=m, mode="linear", align_corners=True
                ).squeeze()
            else:
                y_proj = F.interpolate(
                    y_proj.unsqueeze(0).unsqueeze(0), size=n, mode="linear", align_corners=True
                ).squeeze()

        if p == 1:
            distances.append(torch.mean(torch.abs(x_proj - y_proj)))
        else:
            distances.append(torch.sqrt(torch.mean((x_proj - y_proj) ** 2)))

    return torch.stack(distances).mean()


class DiffusionNet(nn.Module):
    def __init__(self, input_dim, hidden, time_embed_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.body = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.head = nn.Linear(hidden, input_dim)

    def forward(self, x, step):
        scale = max(float(step.max().detach().cpu()), 1.0)
        embed = self.time_embed((step.float() / scale).reshape(-1, 1))
        return self.head(self.body(torch.cat([x, embed], dim=1)))


def predict_x0(x_t, noise_pred, alpha_bar_t):
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t).reshape(-1, 1)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t).reshape(-1, 1)
    return (x_t - sqrt_one_minus * noise_pred) / sqrt_alpha_bar



def phased_time_loss(time_pred, time_true, event):

    time_pred = time_pred.reshape(-1)
    time_true = time_true.reshape(-1)
    event = event.reshape(-1)

    is_event = event > 0.5
    is_censored = ~is_event

    loss = torch.tensor(0.0, device=time_pred.device)
    n_terms = 0

    if is_event.any():
        loss = loss + torch.sum((time_pred[is_event] - time_true[is_event]) ** 2)
        n_terms += int(is_event.sum().item())

    if is_censored.any():
        # penalize only time_pred < time_true ("premature" under-prediction)
        deficit = torch.clamp(time_true[is_censored] - time_pred[is_censored], min=0.0)
        loss = loss + torch.sum(deficit ** 2)
        n_terms += int(is_censored.sum().item())

    if n_terms == 0:
        return loss
    return loss / n_terms


def diffusion_partial_likelihood_loss(time_pred, time_true, event):

    eta = -time_pred.reshape(-1)
    time_true = time_true.reshape(-1)
    event = event.reshape(-1)

    event_indices = torch.where(event > 0.5)[0]
    if len(event_indices) == 0:
        return eta.sum() * 0.0

    loss = torch.tensor(0.0, device=eta.device)
    for idx in event_indices:
        risk_set = time_true >= time_true[idx]
        log_sum = torch.logsumexp(eta[risk_set], dim=0)
        loss = loss - (eta[idx] - log_sum)
    return loss / len(event_indices)


class WassersteinDiffusion:

    def __init__(self, input_dim, hidden, steps, beta_start, beta_end, device,
                 w_noise=W_NOISE, w_wasserstein=W_WASSERSTEIN,
                 w_time=W_TIME, w_partial=W_PARTIAL_DIFFUSION):
        self.input_dim = input_dim
        self.steps = steps
        self.device = device
        self.model = DiffusionNet(input_dim, hidden).to(device)
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.w_noise = w_noise
        self.w_wasserstein = w_wasserstein
        self.w_time = w_time
        self.w_partial = w_partial

    def train_step(self, x_batch, y_batch, e_batch, optimizer):
        self.model.train()
        batch_size = x_batch.size(0)

        step = torch.randint(0, self.steps, (batch_size,), device=self.device)
        noise = torch.randn_like(x_batch)
        alpha_bar_t = self.alpha_bar[step]

        sqrt_alpha_bar = torch.sqrt(alpha_bar_t).reshape(-1, 1)
        sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t).reshape(-1, 1)
        x_t = sqrt_alpha_bar * x_batch + sqrt_one_minus * noise

        noise_pred = self.model(x_t, step)

        # 1. Denoising loss
        loss_noise = F.mse_loss(noise_pred, noise)

        # Reconstructed z0 estimate, used by losses 2, 3, 4
        x0_pred = predict_x0(x_t, noise_pred, alpha_bar_t)
        time_pred = x0_pred[:, -1]
        time_true = x_batch[:, -1]

        # 2. Wasserstein constraint (distribution-level, full batch)
        loss_wasserstein = sliced_wasserstein_distance(x0_pred, x_batch)

        # 3. Phased time loss (per-sample, split by censoring status)
        loss_time = phased_time_loss(time_pred, time_true, e_batch)

        # 4. Partial likelihood (ranking) loss on reconstructed times
        loss_partial = diffusion_partial_likelihood_loss(time_pred, time_true, e_batch)

        loss = (
            self.w_noise * loss_noise
            + self.w_wasserstein * loss_wasserstein
            + self.w_time * loss_time
            + self.w_partial * loss_partial
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        return {
            "loss": float(loss.item()),
            "loss_noise": float(loss_noise.item()),
            "loss_wasserstein": float(loss_wasserstein.item()),
            "loss_time": float(loss_time.item()),
            "loss_partial": float(loss_partial.item()),
        }

    def fit(self, x, y, e, epochs, lr, batch_size, verbose=True):
        """Train on real (x, y, e) tuples. Returns a per-epoch history list."""
        data = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        times = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        events = torch.as_tensor(e, dtype=torch.float32, device=self.device)

        dataset = torch.utils.data.TensorDataset(data, times, events)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=min(batch_size, len(data)), shuffle=True
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        history = []
        for epoch in range(epochs):
            epoch_losses = []
            for x_batch, y_batch, e_batch in loader:
                epoch_losses.append(self.train_step(x_batch, y_batch, e_batch, optimizer))

            epoch_summary = {
                key: float(np.mean([step[key] for step in epoch_losses]))
                for key in epoch_losses[0]
            }
            epoch_summary["epoch"] = epoch + 1
            history.append(epoch_summary)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"[diffusion] epoch {epoch + 1}/{epochs} "
                    f"loss={epoch_summary['loss']:.4f} "
                    f"noise={epoch_summary['loss_noise']:.4f} "
                    f"wasserstein={epoch_summary['loss_wasserstein']:.4f} "
                    f"time={epoch_summary['loss_time']:.4f} "
                    f"partial={epoch_summary['loss_partial']:.4f}"
                )
        return history

    @torch.no_grad()
    def sample(self, n, seed):
        torch.manual_seed(seed)
        x = torch.randn((n, self.input_dim), device=self.device)
        self.model.eval()
        for step_idx in range(self.steps - 1, -1, -1):
            step = torch.full((n,), step_idx, device=self.device, dtype=torch.long)
            beta, alpha, alpha_bar = self.betas[step_idx], self.alphas[step_idx], self.alpha_bar[step_idx]
            noise_pred = self.model(x, step)
            mean = (x - beta / torch.sqrt(1.0 - alpha_bar) * noise_pred) / torch.sqrt(alpha)
            x = mean + torch.sqrt(beta) * torch.randn_like(x) if step_idx > 0 else mean
        return x.cpu().numpy().astype(np.float32)

def generate_synthetic_survival(diffusion, censoring, n_samples, time_mean, time_std, seed):
    raw = diffusion.sample(n_samples, seed)
    features = raw[:, :-1]
    log_time = raw[:, -1] * time_std + time_mean
    candidate_time = np.maximum(np.exp(log_time), 0.0)

    rng = np.random.default_rng(seed)
    censor_time = censoring.sample_times(features, rng)

    observed_time = np.minimum(candidate_time, censor_time).astype(np.float32)
    event = (candidate_time <= censor_time).astype(np.float32)
    return features, observed_time, event


class RiskNetwork(nn.Module):
    def __init__(self, input_dim, hidden, layers, dropout):
        super().__init__()
        blocks = []
        current = input_dim
        for _ in range(max(1, layers)):
            blocks += [nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(dropout)]
            current = hidden
        blocks.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(x).reshape(-1)


def confidence_weighted_cox_loss(risk, time, event, weight):

    risk, time = risk.reshape(-1), time.reshape(-1)
    event, weight = event.reshape(-1), weight.reshape(-1)

    event_indices = torch.where(event > 0.5)[0]
    if len(event_indices) == 0:
        return risk.sum() * 0.0

    loss = torch.tensor(0.0, device=risk.device)
    normalizer = weight[event_indices].sum().clamp_min(EPS)
    for idx in event_indices:
        risk_set = time >= time[idx]
        log_sum = torch.logsumexp(
            risk[risk_set] + torch.log(weight[risk_set].clamp_min(EPS)), dim=0
        )
        loss = loss - weight[idx] * (risk[idx] - log_sum)
    return loss / normalizer


def generated_sample_penalty(risk_synthetic):

    if risk_synthetic.numel() == 0:
        return torch.tensor(0.0, device=risk_synthetic.device)
    return torch.mean(risk_synthetic ** 2)



def train_risk_model(
    x_train, y_train, e_train, x_val, y_val, e_val,
    input_dim, device,
    synthetic=None,
    synthetic_weight=SYNTHETIC_SAMPLE_WEIGHT,
    synthetic_penalty=SYNTHETIC_RISK_PENALTY,
    epochs=RISK_EPOCHS, lr=RISK_LR, weight_decay=RISK_WEIGHT_DECAY,
    early_stopping=RISK_EARLY_STOPPING, verbose=True,
):
   
    if synthetic is None:
        x_fit, y_fit, e_fit = x_train, y_train, e_train
        weights = np.ones(len(x_fit), dtype=np.float32)
        n_real = len(x_fit)
    else:
        sx, sy, se = synthetic
        x_fit = np.concatenate([x_train, sx])
        y_fit = np.concatenate([y_train, sy])
        e_fit = np.concatenate([e_train, se])
        weights = np.concatenate([
            np.ones(len(x_train), dtype=np.float32),
            np.full(len(sx), synthetic_weight, dtype=np.float32),
        ])
        n_real = len(x_train)

    model = RiskNetwork(input_dim, RISK_HIDDEN, RISK_LAYERS, RISK_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    tx = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    ty = torch.as_tensor(y_fit, dtype=torch.float32, device=device)
    te = torch.as_tensor(e_fit, dtype=torch.float32, device=device)
    tw = torch.as_tensor(weights, dtype=torch.float32, device=device)
    vx = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    vy = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    ve = torch.as_tensor(e_val, dtype=torch.float32, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val, wait = float("inf"), 0
    history = []

    for epoch in range(epochs):
        model.train()
        risk = model(tx)
        cox_term = confidence_weighted_cox_loss(risk, ty, te, tw)
        penalty_term = generated_sample_penalty(risk[n_real:]) if synthetic is not None else torch.tensor(0.0, device=device)
        loss = cox_term + synthetic_penalty * penalty_term

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = confidence_weighted_cox_loss(model(vx), vy, ve, torch.ones_like(ve)).item()

        history.append({
            "epoch": epoch + 1,
            "train_cox_loss": float(cox_term.item()),
            "train_penalty_loss": float(penalty_term.item()),
            "val_cox_loss": float(val_loss),
        })

        if val_loss < best_val - 1e-6:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= early_stopping:
                if verbose:
                    print(f"[risk] early stopping at epoch {epoch + 1}")
                break

        if verbose and (epoch + 1) % 25 == 0:
            print(f"[risk] epoch {epoch + 1}/{epochs} train_cox={cox_term.item():.4f} "
                  f"penalty={penalty_term.item():.4f} val_cox={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model, history, best_val

def predict_risk(model, x, device):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()


def compute_baseline_hazard(model, x, y, event, device):
    risk = predict_risk(model, x, device)
    event_mask = event.astype(bool)
    if not np.any(event_mask):
        return np.array([0.0]), np.array([0.0])

    times = np.sort(np.unique(y[event_mask]))
    cumulative, running = [], 0.0
    for t in times:
        n_events = np.sum((y == t) & event_mask)
        at_risk = y >= t
        risk_sum = np.sum(np.exp(np.clip(risk[at_risk], -30.0, 30.0)))
        running += float(n_events) / max(float(risk_sum), EPS)
        cumulative.append(running)
    return times.astype(float), np.array(cumulative, dtype=float)


def predict_survival(model, x, times, baseline_times, baseline_cumhaz, device):
    risk = predict_risk(model, x, device)
    baseline = np.interp(times, baseline_times, baseline_cumhaz, left=0.0,
                         right=float(baseline_cumhaz[-1]))
    return np.exp(-np.outer(np.exp(np.clip(risk, -30.0, 30.0)), baseline)).clip(0.0, 1.0)


def integrated_brier_score(model, x, y, event, train_x, train_y, train_event,
                           censoring, device, n_points=IBS_POINTS):
    baseline_times, baseline_cumhaz = compute_baseline_hazard(model, train_x, train_y, train_event, device)
    max_time = min(float(np.max(train_y)), censoring.max_time)
    times = np.linspace(0.0, max_time, max(2, n_points))
    surv = predict_survival(model, x, times, baseline_times, baseline_cumhaz, device)

    brier = []
    for j, t in enumerate(times):
        g_t = censoring.survival(x, np.array([t]))[:, 0]
        g_y = np.diag(censoring.survival(x, np.maximum(y, 0.0)))
        target = (y > t).astype(float)

        weights = np.zeros_like(y, dtype=float)
        still_at_risk = y > t
        weights[still_at_risk] = 1.0 / np.maximum(g_t[still_at_risk], 0.05)
        observed_before_t = (y <= t) & (event > 0.5)
        weights[observed_before_t] = 1.0 / np.maximum(g_y[observed_before_t], 0.05)

        brier.append(np.mean(weights * (target - surv[:, j]) ** 2))

    return float(np.trapz(brier, times) / max(times[-1] - times[0], EPS))


def evaluate_survival_model(model, x, y, event, train_x, train_y, train_event, censoring, device):
    risk = predict_risk(model, x, device)
    c_index = float(concordance_index(y, -risk, event))
    ibs = integrated_brier_score(model, x, y, event, train_x, train_y, train_event, censoring, device)
    return {"c_index": c_index, "ibs": ibs}



if __name__ == "__main__":

    set_seed(SEED)
    device = get_device(DEVICE_STR)
    print(f"[step0] device = {device}")

    frame, y_all, e_all, feature_columns = load_survival_csv(
        CSV_PATH, TIME_COLUMN, EVENT_COLUMN, DROP_COLUMNS
    )
    print(f"[step1] loaded {len(frame)} rows, {len(feature_columns)} feature columns")
    print(f"[step1] overall event rate = {float(np.mean(e_all)):.4f}")

    splits = create_splits(len(frame), e_all, TEST_SIZE, VALIDATION_SIZE, SEED)
    print(f"[step2] split sizes = { {k: len(v) for k, v in splits.items()} }")

    preprocessor = SurvivalPreprocessor(feature_columns).fit(frame.iloc[splits["train"]])

    x_train = preprocessor.transform(frame.iloc[splits["train"]])
    y_train = y_all[splits["train"]]
    e_train = e_all[splits["train"]]

    x_val = preprocessor.transform(frame.iloc[splits["validation"]])
    y_val = y_all[splits["validation"]]
    e_val = e_all[splits["validation"]]

    x_test = preprocessor.transform(frame.iloc[splits["test"]])
    y_test = y_all[splits["test"]]
    e_test = e_all[splits["test"]]

    print(f"[step3] x_train shape = {x_train.shape}, encoded feature count = {x_train.shape[1]}")
    print(f"[step3] train event rate = {float(np.mean(e_train)):.4f}, "
          f"validation event rate = {float(np.mean(e_val)):.4f}, "
          f"test event rate = {float(np.mean(e_test)):.4f}")

    censoring_model = CensoringModel().fit(x_train, y_train, e_train)
    censoring_log_likelihood = float(censoring_model.model.log_likelihood_)
    print(f"[step4] censoring Cox model fit on train split")
    print(f"[step4] censoring model log partial likelihood = {censoring_log_likelihood:.4f}")
    print(f"[step4] censoring model concordance = {float(censoring_model.model.concordance_index_):.4f}")

    log_time_train = np.log(np.maximum(y_train, 0.0) + 1e-3)
    time_mean = float(log_time_train.mean())
    time_std = float(log_time_train.std() + 1e-6)
    log_time_normalized = (log_time_train - time_mean) / time_std

    diffusion_input = np.column_stack([x_train, log_time_normalized]).astype(np.float32)
    print(f"[step5] diffusion_input shape = {diffusion_input.shape} "
          f"(time_mean={time_mean:.4f}, time_std={time_std:.4f})")


    diffusion = WassersteinDiffusion(
        input_dim=diffusion_input.shape[1],
        hidden=DIFFUSION_HIDDEN,
        steps=DIFFUSION_STEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        device=device,
    )

    diffusion_history = diffusion.fit(
        diffusion_input, y_train, e_train,
        epochs=DIFFUSION_EPOCHS, lr=DIFFUSION_LR, batch_size=DIFFUSION_BATCH_SIZE, verbose=True,
    )
    diffusion_history_df = pd.DataFrame(diffusion_history)
    print("[step6] final diffusion epoch losses:")
    print(diffusion_history_df.tail(1).to_string(index=False))

    n_synthetic = max(1, int(round(len(x_train) * SYNTHETIC_MULTIPLIER)))
    synthetic_x, synthetic_y, synthetic_e = generate_synthetic_survival(
        diffusion, censoring_model, n_synthetic, time_mean, time_std, seed=SEED + 1000
    )
    print(f"[step7] generated {len(synthetic_x)} synthetic records")
    print(f"[step7] synthetic event rate = {float(np.mean(synthetic_e)):.4f} "
          f"(train event rate = {float(np.mean(e_train)):.4f})")

    risk_model, risk_history, best_val_cox_loss = train_risk_model(
        x_train, y_train, e_train,
        x_val, y_val, e_val,
        input_dim=x_train.shape[1],
        device=device,
        synthetic=(synthetic_x, synthetic_y, synthetic_e),
        verbose=True,
    )
    risk_history_df = pd.DataFrame(risk_history)
    print(f"[step8] best validation Cox loss = {best_val_cox_loss:.4f}")

    validation_metrics = evaluate_survival_model(
        risk_model, x_val, y_val, e_val, x_train, y_train, e_train, censoring_model, device
    )
    test_metrics = evaluate_survival_model(
        risk_model, x_test, y_test, e_test, x_train, y_train, e_train, censoring_model, device
    )
    print(f"[step9] validation: C-index={validation_metrics['c_index']:.4f}, IBS={validation_metrics['ibs']:.4f}")
    print(f"[step9] test:       C-index={test_metrics['c_index']:.4f}, IBS={test_metrics['ibs']:.4f}")

    external_metrics = None
    if EXTERNAL_CSV_PATH is not None:
        ext_frame, ext_y, ext_e, ext_columns = load_survival_csv(
            EXTERNAL_CSV_PATH, TIME_COLUMN, EVENT_COLUMN, DROP_COLUMNS
        )
        missing = sorted(set(feature_columns) - set(ext_columns))
        if missing:
            raise ValueError(f"External data missing feature columns: {', '.join(missing)}")
        ext_x = preprocessor.transform(ext_frame[feature_columns])
        external_metrics = evaluate_survival_model(
            risk_model, ext_x, ext_y, ext_e, x_train, y_train, e_train, censoring_model, device
        )
        print(f"[step10] external: C-index={external_metrics['c_index']:.4f}, IBS={external_metrics['ibs']:.4f}")
    else:
        print("[step10] no external CSV configured, skipping external validation")

    results = {
        "dataset": os.path.basename(CSV_PATH),
        "device": str(device),
        "feature_count": len(feature_columns),
        "encoded_feature_count": int(x_train.shape[1]),
        "split_sizes": {name: int(len(idx)) for name, idx in splits.items()},
        "event_rates": {
            "train": float(np.mean(e_train)),
            "validation": float(np.mean(e_val)),
            "test": float(np.mean(e_test)),
        },
        "censoring_model": {
            "log_partial_likelihood": censoring_log_likelihood,
            "concordance": float(censoring_model.model.concordance_index_),
        },
        "diffusion_loss_weights": {
            "noise": W_NOISE,
            "wasserstein": W_WASSERSTEIN,
            "time": W_TIME,
            "partial_likelihood": W_PARTIAL_DIFFUSION,
        },
        "validation": {**validation_metrics, "best_val_cox_loss": best_val_cox_loss},
        "test": test_metrics,
        "external": external_metrics,
        "synthetic_size": int(len(synthetic_x)),
        "synthetic_event_rate": float(np.mean(synthetic_e)),
    }

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    diffusion_history_df.to_csv(os.path.join(OUTPUT_DIR, "diffusion_history.csv"), index=False)
    risk_history_df.to_csv(os.path.join(OUTPUT_DIR, "risk_history.csv"), index=False)
    torch.save(risk_model.state_dict(), os.path.join(OUTPUT_DIR, "risk_model.pt"))
    torch.save(diffusion.model.state_dict(), os.path.join(OUTPUT_DIR, "diffusion_model.pt"))

    print(f"[step11] results saved to {OUTPUT_DIR}")
    print(json.dumps(results, indent=2))
