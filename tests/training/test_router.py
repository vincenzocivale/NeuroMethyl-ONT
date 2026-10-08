import numpy as np

from neuromethyl_ont.training.router import predict_domain_proba, train_router


def _toy(n=200, d=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    signal = np.where(y[:, None] == 1, 1.0, -1.0) * (np.arange(d) < 40)
    x = np.where(rng.random((n, d)) < 0.5, 1.0, -1.0) * (np.arange(d) >= 40) + signal
    return x.astype(np.float32), y


def test_router_learns_separable_domains_and_returns_probabilities():
    x, y = _toy()
    xv, yv = _toy(seed=1)
    result = train_router(x, y, xv, yv, device="cpu", max_epochs=15, lr=0.05)
    proba = predict_domain_proba(result.model, xv)
    assert proba.shape == (len(xv), 2)
    assert np.allclose(proba.sum(1), 1.0, atol=1e-5)
    assert (proba.argmax(1) == yv).mean() > 0.9
