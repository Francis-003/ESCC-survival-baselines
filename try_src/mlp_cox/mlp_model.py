"""Independent, deterministic CPU MLP Cox model with full risk set Efron loss."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch import nn


def cox_ph_loss(
    log_risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    ties: str = "efron",
) -> torch.Tensor:
    """Mean negative Cox partial log likelihood over observed events.

    Efron's approximation uses all subjects at the event time in the risk set,
    including those censored at that time. Accumulation uses float64 so very
    large finite log risks do not require exponentiation in the input scale.
    """
    if ties != "efron":
        raise ValueError("only Efron ties are supported")
    if not all(isinstance(v, torch.Tensor) for v in (log_risk, time, event)):
        raise TypeError("log_risk, time, and event must be tensors")
    if log_risk.ndim != 1 or time.ndim != 1 or event.ndim != 1:
        raise ValueError("log_risk, time, and event must be one dimensional")
    if not (len(log_risk) == len(time) == len(event)) or len(time) == 0:
        raise ValueError("log_risk, time, and event must have equal nonzero length")
    if log_risk.device != time.device or log_risk.device != event.device:
        raise ValueError("all tensors must be on the same device")
    if not torch.is_floating_point(log_risk) or not torch.is_floating_point(time):
        raise ValueError("log_risk and time must be floating point")
    if not bool(torch.isfinite(log_risk).all()) or not bool(torch.isfinite(time).all()):
        raise ValueError("log_risk and time must be finite")
    if not bool((time > 0).all()) or not bool(((event == 0) | (event == 1)).all()):
        raise ValueError("time must be positive and event must be binary")

    observed = event.to(dtype=torch.bool)
    n_events = int(observed.sum().item())
    if n_events == 0:
        raise ValueError("Cox loss requires at least one observed event")

    return _CoxLossPlan(time, observed).loss(log_risk)


class _CoxLossPlan:
    """Cache label-only sorting and Efron groups for repeated full-batch losses."""

    def __init__(self, time: torch.Tensor, observed: torch.Tensor) -> None:
        self.sorted_index = torch.argsort(time, descending=True, stable=True)
        death_index = torch.nonzero(observed, as_tuple=True)[0]
        death_order = torch.argsort(time[death_index], descending=True, stable=True)
        self.death_index = death_index[death_order]
        death_times, self.group_counts = torch.unique_consecutive(
            time[self.death_index], return_counts=True
        )
        self.risk_end = torch.searchsorted(
            -time[self.sorted_index].contiguous(), -death_times.contiguous(), right=True
        ) - 1
        self.group_index = torch.repeat_interleave(
            torch.arange(len(self.group_counts), device=time.device), self.group_counts
        )
        offsets = torch.cumsum(self.group_counts, dim=0) - self.group_counts
        position = torch.arange(len(self.death_index), device=time.device)
        self.efron_fraction = (
            (position - offsets[self.group_index]).to(torch.float64)
            / self.group_counts[self.group_index].to(torch.float64)
        )
        self.n_events = len(self.death_index)

    def loss(self, log_risk: torch.Tensor) -> torch.Tensor:
        risk = log_risk.to(dtype=torch.float64)
        # Descending time puts everyone at or above each event time in a prefix.
        log_riskset = torch.logcumsumexp(risk[self.sorted_index], dim=0)[self.risk_end]
        deaths = risk[self.death_index]
        group_max = torch.segment_reduce(deaths, reduce="max", lengths=self.group_counts)
        shifted = torch.exp(deaths - group_max[self.group_index])
        group_sum = torch.segment_reduce(shifted, reduce="sum", lengths=self.group_counts)
        log_deaths = group_max + torch.log(group_sum)
        death_fraction = torch.exp(log_deaths - log_riskset)
        log_efron_denom = log_riskset[self.group_index] + torch.log1p(
            -death_fraction[self.group_index] * self.efron_fraction
        )
        return (log_efron_denom.sum() - deaths.sum()) / self.n_events


def _checked_x(x: np.ndarray, input_dim: int) -> np.ndarray:
    array = np.asarray(x, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != input_dim or array.shape[0] == 0:
        raise ValueError(f"x must have shape (n, {input_dim}) with n > 0")
    if not np.isfinite(array).all():
        raise ValueError("x must contain only finite values")
    return np.ascontiguousarray(array)


def _checked_y(y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(y)
    if array.ndim != 1 or len(array) != n or array.dtype.names is None:
        raise ValueError("y must be a structured survival array matching x")
    if "event" not in array.dtype.names or "time" not in array.dtype.names:
        raise ValueError("y must contain event and time fields")
    try:
        event = np.asarray(array["event"], dtype=np.float64)
        time = np.asarray(array["time"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("event and time must be numeric") from exc
    if not np.isfinite(event).all() or not np.isin(event, (0, 1)).all():
        raise ValueError("event must be binary and finite")
    if not np.isfinite(time).all() or not (time > 0).all():
        raise ValueError("time must be positive and finite")
    if not np.any(event):
        raise ValueError("at least one observed event is required")
    # Structured fields may have record-size strides, which torch rejects.
    return np.ascontiguousarray(time), np.ascontiguousarray(event.astype(bool))


class MLPCox:
    """Deterministic small MLP fitted with full risk sets and AdamW.

    ``epochs`` in fit requests a fixed training length. Otherwise an optional
    validation set selects the best epoch and triggers patience based stopping.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, ...] = (32,),
        dropout: float = 0.2,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        seed: int = 11,
        max_epochs: int = 150,
        patience: int = 20,
        min_delta: float = 1e-4,
    ) -> None:
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        hidden = tuple(hidden)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in hidden):
            raise ValueError("hidden widths must be positive integers")
        if not 0 <= dropout < 1 or not np.isfinite(dropout):
            raise ValueError("dropout must be finite and in [0, 1)")
        if not np.isfinite(lr) or lr <= 0 or not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("lr must be positive and weight_decay nonnegative")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (max_epochs, patience)):
            raise ValueError("max_epochs and patience must be positive integers")
        if not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        self.input_dim = input_dim
        self.hidden = hidden
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = float(min_delta)

    def _make_network(self) -> nn.Module:
        layers: list[nn.Module] = []
        width = self.input_dim
        for next_width in self.hidden:
            layers.extend((nn.Linear(width, next_width), nn.ReLU()))
            if self.dropout:
                layers.append(nn.Dropout(self.dropout))
            width = next_width
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
        epochs: int | None = None,
    ) -> MLPCox:
        if (x_valid is None) != (y_valid is None):
            raise ValueError("x_valid and y_valid must be supplied together")
        if epochs is not None and (isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1):
            raise ValueError("epochs must be a positive integer")
        train_x = _checked_x(x, self.input_dim)
        train_time, train_event = _checked_y(y, len(train_x))
        if x_valid is not None:
            valid_x = _checked_x(x_valid, self.input_dim)
            valid_time, valid_event = _checked_y(y_valid, len(valid_x))

        # CPU and one thread give repeatable order for this small full batch model.
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.network_ = self._make_network()
            optimizer = torch.optim.AdamW(
                self.network_.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            x_t = torch.from_numpy(train_x)
            time_t = torch.from_numpy(train_time)
            event_t = torch.from_numpy(train_event)
            train_plan = _CoxLossPlan(time_t, event_t)
            if x_valid is not None:
                vx_t = torch.from_numpy(valid_x)
                vt_t = torch.from_numpy(valid_time)
                ve_t = torch.from_numpy(valid_event)
                valid_plan = _CoxLossPlan(vt_t, ve_t)

            self.training_history_: list[dict[str, float | int | None]] = []
            best_score = float("inf")
            best_state = None
            stale = 0
            limit = self.max_epochs if epochs is None else epochs
            for epoch in range(1, limit + 1):
                self.network_.train()
                optimizer.zero_grad()
                loss = train_plan.loss(self.network_(x_t).flatten())
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                optimizer.step()

                self.network_.eval()
                with torch.no_grad():
                    train_loss = float(train_plan.loss(self.network_(x_t).flatten()))
                    valid_loss = (
                        float(valid_plan.loss(self.network_(vx_t).flatten()))
                        if x_valid is not None else None
                    )
                if not np.isfinite(train_loss) or (valid_loss is not None and not np.isfinite(valid_loss)):
                    raise FloatingPointError("nonfinite fitted loss")
                self.training_history_.append(
                    {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
                )
                if epochs is None and x_valid is not None:
                    if valid_loss < best_score - self.min_delta:
                        best_score = valid_loss
                        self.best_epoch_ = epoch
                        best_state = copy.deepcopy(self.network_.state_dict())
                        stale = 0
                    else:
                        stale += 1
                        if stale >= self.patience:
                            break
            if best_state is not None:
                self.network_.load_state_dict(best_state)
            else:
                self.best_epoch_ = len(self.training_history_)

        self.network_.eval()
        with torch.no_grad():
            train_log_risk = self.network_(torch.from_numpy(train_x)).flatten().numpy().astype(np.float64)
        self.train_risk_mean_ = float(train_log_risk.mean())
        centered = train_log_risk - self.train_risk_mean_
        event_times = np.unique(train_time[train_event])
        log_increments = []
        for t in event_times:
            deaths = np.count_nonzero(train_event & (train_time == t))
            log_denom = np.logaddexp.reduce(centered[train_time >= t])
            log_increments.append(np.log(deaths) - log_denom)
        self.baseline_times_ = event_times
        self.baseline_log_cumulative_hazard_ = np.logaddexp.accumulate(log_increments)
        with np.errstate(over="ignore", under="ignore"):
            self.baseline_cumulative_hazard_ = np.exp(self.baseline_log_cumulative_hazard_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "network_") or not hasattr(self, "train_risk_mean_"):
            raise RuntimeError("model is not fitted")
        array = _checked_x(x, self.input_dim)
        with torch.no_grad():
            raw = self.network_(torch.from_numpy(array)).flatten().numpy().astype(np.float64)
        centered = raw - self.train_risk_mean_
        if not np.isfinite(centered).all():
            raise FloatingPointError("nonfinite predicted log risk")
        return centered

    def predict_survival(self, x: np.ndarray, times: np.ndarray) -> np.ndarray:
        grid = np.asarray(times, dtype=np.float64)
        if grid.ndim != 1 or not np.isfinite(grid).all() or np.any(grid < 0):
            raise ValueError("times must be a finite nonnegative one dimensional grid")
        risk = self.predict(x)
        index = np.searchsorted(self.baseline_times_, grid, side="right") - 1
        log_hazard = np.where(
            index >= 0, self.baseline_log_cumulative_hazard_[np.maximum(index, 0)], -np.inf
        )
        # Evaluate S(t|x)=exp(-exp(log H0(t)+log risk)) in log space. The
        # outer exponential naturally tends to 0 or 1 at representational
        # extremes; no training scores or hazards are clipped.
        with np.errstate(over="ignore", under="ignore"):
            return np.exp(-np.exp(risk[:, None] + log_hazard[None, :]))

    def fitted_state_digest(self) -> str:
        if not hasattr(self, "baseline_times_"):
            raise RuntimeError("model is not fitted")
        digest = hashlib.sha256()
        for name, value in self.network_.state_dict().items():
            data = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(data.shape).encode("ascii"))
            digest.update(data.tobytes())
        for name in (
            "train_risk_mean_", "baseline_times_", "baseline_cumulative_hazard_",
            "baseline_log_cumulative_hazard_",
        ):
            data = np.ascontiguousarray(np.asarray(getattr(self, name), dtype=np.float64))
            digest.update(name.encode("ascii"))
            digest.update(str(data.shape).encode("ascii"))
            digest.update(data.tobytes())
        return digest.hexdigest()
