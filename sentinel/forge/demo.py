"""A made-up 'normal' machine, for demonstrations and tests ONLY. A neuron forged from it is marked `demo` and can never be approved:
the whole point of the Forge is to fit the real machine's own normal life."""
import numpy as np

from ..features import FEATURES


def synthetic_baseline(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        act = rng.gamma(2.0, 0.5)
        r = {f: 0.0 for f in FEATURES}
        r["n_proc_new"] = rng.poisson(1.0 * act); r["n_proc_user_dir"] = rng.poisson(0.1 * act); r["n_script_host"] = rng.poisson(0.2 * act)
        r["n_conn_new"] = rng.poisson(4 * act); r["n_conn_novel"] = rng.poisson(0.8 * act)
        r["n_remote_ips"] = min(r["n_conn_new"], rng.poisson(2 * act)); r["n_public_conn"] = rng.poisson(3 * act)
        r["n_file_events"] = rng.poisson(6 * act); r["n_file_created"] = rng.poisson(1.5 * act); r["n_file_renamed"] = rng.poisson(0.2 * act)
        r["dns_queries"] = rng.poisson(5 * act); r["dns_nx"] = rng.binomial(int(r["dns_queries"]), 0.05)
        r["dns_entropy"] = rng.normal(3.2, 0.2) if r["dns_queries"] > 0 else 0.0
        r["net_bytes_out"] = rng.lognormal(9 + act, 1); r["net_bytes_in"] = rng.lognormal(11 + act, 1); r["net_pkts"] = r["net_bytes_in"] / 900
        out.append(r)
    return out
