"""
model_numpy.py
==============
A small, dependency-light DEEP discrete-time COMPETING-RISKS model, implemented
in pure NumPy with manual backprop so the benchmark runs ANYWHERE (no torch,
no internet). This is the DeepHit objective (Lee et al., AAAI 2018) applied to
a feed-forward net.

The PRODUCTION model (a GRU that learns trajectory features end-to-end) lives in
model_torch.py and uses the identical loss; this NumPy net trains on engineered
landmark features instead. Both share the same output parameterisation, so their
metrics are directly comparable.

Output parameterisation
-----------------------
For K causes and J horizon bins, the network emits K*J + 1 logits. A single
softmax gives a joint distribution q over {(cause k, bin j)} plus one "no event
within horizon" mass. Then:
    P(event via k in bin j)          = q[k, j]
    CIF_k(by bin j | landmark)       = sum_{j' <= j} q[k, j']
    P(progress within W | landmark)  = sum_{k, j: bin j <= W} q[k, j]

DeepHit log-likelihood (unified form)
-------------------------------------
Each sample defines an "allowed set" A of output cells consistent with what we
observed, and the loss is  -log( sum_{m in A} q_m ):
    * event (k*, j*) : A = {(k*, j*)}                         -> ordinary CE
    * censored at c  : A = {survive} U {(k, j): j > c}        -> right-censoring
The gradient wrt logits z has the clean closed form
    dL/dz = q - (q * 1_A) / (q . 1_A)
"""

from __future__ import annotations
import numpy as np


class DiscreteTimeCompetingRisks:
    def __init__(self, d_in, n_causes, n_bins, hidden=64, seed=0):
        self.K, self.J = n_causes, n_bins
        self.out = n_causes * n_bins + 1
        self.surv = self.out - 1
        rng = np.random.default_rng(seed)
        # He init
        self.W1 = rng.normal(0, np.sqrt(2 / d_in), (hidden, d_in))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden), (self.out, hidden))
        self.b2 = np.zeros(self.out)
        self._init_adam()

    # ----- adam state
    def _init_adam(self):
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0

    def _params(self):
        return dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    # ----- forward
    def _forward(self, X):
        z1 = X @ self.W1.T + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2.T + self.b2
        z2 -= z2.max(1, keepdims=True)
        e = np.exp(z2)
        q = e / e.sum(1, keepdims=True)
        return q, a1, z1

    def predict_q(self, X):
        return self._forward(X)[0]

    # ----- build per-sample allowed-set mask (N, out)
    def _allowed_mask(self, cause, tbin, event):
        N = len(cause)
        M = np.zeros((N, self.out))
        for i in range(N):
            if event[i]:
                M[i, cause[i] * self.J + tbin[i]] = 1.0
            else:
                c = tbin[i]
                M[i, self.surv] = 1.0
                for k in range(self.K):
                    for j in range(c + 1, self.J):
                        M[i, k * self.J + j] = 1.0
        return M

    # ----- one training step
    def step(self, X, cause, tbin, event, lr=2e-3, l2=1e-5):
        q, a1, z1 = self._forward(X)
        M = self._allowed_mask(cause, tbin, event)
        S = (q * M).sum(1, keepdims=True) + 1e-12
        loss = -np.log(S).mean()
        # grad wrt logits
        gz2 = q - (q * M) / S                      # (N, out)
        gz2 /= len(X)
        gW2 = gz2.T @ a1 + l2 * self.W2
        gb2 = gz2.sum(0)
        ga1 = gz2 @ self.W2
        gz1 = ga1 * (z1 > 0)
        gW1 = gz1.T @ X + l2 * self.W1
        gb1 = gz1.sum(0)
        self._adam_update(dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2), lr)
        return float(loss)

    def _adam_update(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g * g)
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            getattr(self, k)[...] -= lr * mhat / (np.sqrt(vhat) + eps)

    def fit(self, X, cause, tbin, event, epochs=120, batch=256, lr=2e-3,
            l2=1e-5, verbose=False):
        N = len(X)
        rng = np.random.default_rng(0)
        for ep in range(epochs):
            idx = rng.permutation(N)
            losses = []
            for s in range(0, N, batch):
                b = idx[s:s + batch]
                losses.append(self.step(X[b], cause[b], tbin[b], event[b],
                                        lr=lr, l2=l2))
            if verbose and (ep % 20 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:3d}  loss {np.mean(losses):.4f}")
        return self

    # ----- convenience outputs
    def cif(self, X):
        """Cumulative incidence per cause at each bin. Returns (N, K, J)."""
        q = self.predict_q(X)[:, :self.surv].reshape(-1, self.K, self.J)
        return np.cumsum(q, axis=2)

    def risk_within(self, X, max_bin):
        """P(progress via any cause within bins 0..max_bin). Returns (N,)."""
        q = self.predict_q(X)[:, :self.surv].reshape(-1, self.K, self.J)
        return q[:, :, :max_bin + 1].sum(axis=(1, 2))

    def risk_score(self, X):
        """Continuous discrimination score = area under the all-cause CIF
        (sum over bins of CIF). Higher => earlier / more likely event, so it
        is monotone with risk and ideal for concordance. Returns (N,)."""
        q = self.predict_q(X)[:, :self.surv].reshape(-1, self.K, self.J)
        cif_allcause = q.sum(axis=1).cumsum(axis=1)        # (N, J)
        return cif_allcause.sum(axis=1)
