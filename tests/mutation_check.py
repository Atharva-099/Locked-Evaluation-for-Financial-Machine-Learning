"""Plant one known bug at a time in a throwaway copy of the code and confirm the
test suite fails for each. A bug the tests miss means a test is missing.

Run: python tests/mutation_check.py [file]   (not collected by pytest; e.g. file = sweeps.py)
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]

# Only the tests that cover each file are re-run, to keep this check light.
TESTS_FOR = {
    "splits.py": ["tests/test_contract_splits.py"], "contract.py": ["tests/test_contract_splits.py"],
    "tasks/volatility.py": ["tests/test_volatility.py", "tests/test_clean.py"], "data/clean.py": ["tests/test_clean.py", "tests/test_volatility.py"],
    "models.py": ["tests/test_models_runner.py"], "runner.py": ["tests/test_models_runner.py"],
    "metrics.py": ["tests/test_report.py"], "uncertainty.py": ["tests/test_uncertainty.py", "tests/test_report.py"],
    "misses.py": ["tests/test_report.py"], "importance.py": ["tests/test_report.py"], "sweeps.py": ["tests/test_sweeps.py"],
}

MUTANTS = [
    ("splits.py", "tr = usable & (la <= train_end)", "tr = usable & (la <= c)", "train uses labels known only at fit time"),
    ("splits.py", "        usable = la < reserved_start\n", "        usable = pd.Series(True, index=panel.index)\n", "exploratory may see reserved labels"),
    ("splits.py", "            in_test &= pt >= reserved_start\n", "            pass\n", "locked tests before reserved start"),
    ("splits.py", "c = pt[in_test].min()", "c = pt[in_test].max()", "fit time after first forecast"),
    ("splits.py", "va = usable & (la > train_end) & (la <= c)", "va = usable & (la > train_end)", "validation includes future labels"),
    ("contract.py", 'df["feature_available_time"] > df["prediction_time"]', 'df["feature_available_time"] > df["label_end_time"]', "feature-time check weakened"),
    ("contract.py", 'df["label_available_time"] < df["label_end_time"]', 'df["label_available_time"] < df["prediction_time"]', "label-availability check weakened"),
    ("contract.py", 'bad = df["label_available_time"] > cutoff', 'bad = df["label_available_time"] > cutoff + pd.Timedelta(days=400)', "known-label guard loosened"),
    ("contract.py", 'bad = df["label_available_time"] > cutoff', 'bad = df["label_available_time"] >= cutoff', "known-label guard off by one"),
    ("contract.py", 'bad = df["prediction_time"] < fit_cutoff', 'bad = df["prediction_time"] < fit_cutoff - pd.Timedelta(days=5)', "forecast-time guard loosened"),
    ("contract.py", "            if overlap.size:", "            if overlap.size > 10**9:", "overlap check broken"),
    ("tasks/volatility.py", "    lab_s = first_idx.loc[months + 1].to_numpy()\n    lab_e = last_idx.loc[months + 1].to_numpy()", "    lab_s = first_idx.loc[months].to_numpy()\n    lab_e = last_idx.loc[months].to_numpy()", "answer taken from month t instead of t+1"),
    ("tasks/volatility.py", "    end = last_idx.loc[months].to_numpy()", "    end = last_idx.loc[months + 1].to_numpy()", "inputs computed with next month's data (lookahead)"),
    ("tasks/volatility.py", '("m", 22, 15)', '("m", 21, 15)', "22-day window shortened to 21"),
    ("tasks/volatility.py", "    target_rv = cfg.days_basis * _safe_div(", "    target_rv = _safe_div(", "answer not scaled to 21 days"),
    ("tasks/volatility.py", 'P, V, S = np.abs(wide["prc"]), wide["vol"], wide["shrout"]', 'P, V, S = wide["prc"], wide["vol"], wide["shrout"]', "negative daily prices not made positive"),
    ("tasks/volatility.py", 'prc = np.abs(univ["prc"].to_numpy(dtype="float64"))', 'prc = univ["prc"].to_numpy(dtype="float64")', "negative monthly prices not made positive"),
    ("tasks/volatility.py", '(out["n_label_days"] < cfg.min_valid_label)', '(out["n_label_days"] <= cfg.min_valid_label)', "answer-month threshold off by one"),
    ("tasks/volatility.py", 'window_short = out["n_window_days"] < cfg.min_valid_window', 'window_short = out["n_window_days"] <= cfg.min_valid_window', "history threshold off by one"),
    ("tasks/volatility.py", "return cal[max(0, e0 - (history_days - 1)): e1 + 1]", "return cal[max(0, e0 - (history_days - 2)): e1 + 1]", "yearly chunk one day short of history"),
    ("tasks/volatility.py", 'd = d.merge(membership, on=["permno", "month"], how="inner")', 'd = d.merge(membership, on=["permno", "month"], how="left")', "market return includes non-universe stocks"),
    ("models.py", 'return _floor(df["rv_m"].to_numpy())', 'return _floor(df["rv_w"].to_numpy())', "persistence uses the wrong window"),
    ("models.py", "return float(np.exp(r.mean() + r.var() / 2))", "return 1.0", "log forecasts not rescaled"),
    ("models.py", "return float(np.exp(r.mean() + r.var() / 2))", "return float(np.mean(np.exp(r)))", "multiplier reverts to the fragile plain average"),
    ("data/clean.py", '(d["ret"].abs() > FAKE_JUMP_THRESHOLD)', '(d["ret"].abs() >= FAKE_JUMP_THRESHOLD)', "fake-jump threshold off by one"),
    ("data/clean.py", ' & ((d["vol"] == 0) | d["vol"].isna())', "", "fake-jump rule ignores trading volume"),
    ("tasks/volatility.py", "    d, counts = clean_daily(raw.load_daily_year(year))", "    d, counts = raw.load_daily_year(year), {\"zero_trade_jumps_removed\": 0, \"leftovers_after_zero_trade\": 0}", "table built from uncleaned data"),
    ("tasks/volatility.py", "    return c_full[end] - c_full[lo] + x_now[end]", "    return c_full[end + 1] - c_full[lo]", "reversal rule applied to the forecast day itself (lookahead)"),
    ("tasks/volatility.py", "    r_full = np.where(flags, np.nan, r_now)", "    r_full = r_now", "reversal rule not applied"),
    ("models.py", 'best = min(trials, key=lambda t: t["val_qlike"])', 'best = max(trials, key=lambda t: t["val_qlike"])', "HAR keeps its worst form"),
    ("models.py", "            if best is None or score < best[0]:\n                best = (score, pipe, s, a)", "            if best is None or score > best[0]:\n                best = (score, pipe, s, a)", "ridge keeps its worst penalty"),
    ("models.py", "            if best is None or score < best[0]:\n                best = (score, m, s, leaves)", "            if best is None or score > best[0]:\n                best = (score, m, s, leaves)", "lightgbm keeps its worst setting"),
    ("models.py", 'd = df["rv_d"].fillna(df["rv_w"]).fillna(df["rv_m"])', 'd = df["rv_d"].fillna(df["rv_m"])', "HAR daily fallback skips weekly"),
    ("models.py", "        return np.clip(np.asarray(X, dtype=\"float64\"), self.lo_, self.hi_)", "        X = np.asarray(X, dtype=\"float64\")\n        return np.clip(X, np.nanquantile(X, self.lower, axis=0), np.nanquantile(X, self.upper, axis=0))", "caps learned from the data being transformed"),
    ("models.py", '        Xtr, ytr = train[self.features], train["target_log_rv"].to_numpy()\n        Xva, y_val = val[self.features], val["target_rv"].to_numpy()\n        trials, best = [], None\n        for a in self.alphas:', '        Xtr, ytr = pd.concat([train, val])[self.features], pd.concat([train, val])["target_log_rv"].to_numpy()\n        Xva, y_val = val[self.features], val["target_rv"].to_numpy()\n        trials, best = [], None\n        for a in self.alphas:', "ridge also learns from tuning rows"),
    ("runner.py", "train, val, test = by_id.loc[f.train_ids].reset_index(), by_id.loc[f.val_ids].reset_index(), by_id.loc[f.test_ids].reset_index()", "train, val, test = by_id.loc[np.concatenate([f.train_ids, f.test_ids])].reset_index(), by_id.loc[f.val_ids].reset_index(), by_id.loc[f.test_ids].reset_index()", "runner trains on test rows"),
    ("runner.py", "    work = panel[quick_sample(panel[\"permno\"])].reset_index(drop=True) if quick else panel", "    work = panel", "quick mode ignores the firm sample"),
    ("runner.py", '    if lock["spec_hash"] != _hash(s):', '    if False:', "locked run skips the spec check"),
    ("runner.py", "    if previous and not allow_rerun:", "    if False:", "locked run can be silently repeated"),
    ("runner.py", '        "code_hash": started_code_hash,', '        "code_hash": code_hash(),', "code fingerprint taken at the end of the run"),
    ("metrics.py", "        d = (q[model] - q[base]).to_numpy()", "        d = (q[base] - q[model]).to_numpy()", "comparison sign flipped"),
    ("uncertainty.py", "    return pd.Series(values).groupby(np.asarray(months)).mean().sort_index()", "    return pd.Series(values).groupby(np.asarray(months)).sum().sort_index()", "months weighted by number of stocks"),
    ("uncertainty.py", "    block = max(1, min(block, n))", "    block = 1", "error bars ignore related months"),
    ("uncertainty.py", "        running = max(running, (m - rank) * p[i])", "        running = max(running, p[i])", "no multiple-testing correction"),
    ("misses.py", "    if hi < 0:\n        return \"better\"", "    if lo < 0:\n        return \"better\"", "verdict 'better' when interval spans zero"),
    ("misses.py", "    price = np.round(np.exp(out[\"log_price\"]), 4)", "    price = np.exp(out[\"log_price\"])", "price boundary rounding removed"),
    ("importance.py", "        idx = pos[months == m]\n        perm[idx] = rng.permutation(idx)", "        pass\n    perm = rng.permutation(pos)", "scramble mixes months"),
    ("importance.py", "np.random.default_rng([seed, int(fold), zlib.crc32(label.encode())])", "np.random.default_rng()", "scramble not reproducible"),
    ("importance.py", "    if _is_month_level(X, cols, months):", "    if False:", "month-level inputs scrambled within month (always zero)"),
    ("sweeps.py", "                    if best is None or v < best[0]:", "                    if best is None:", "greedy ignores the validation score"),
    ("sweeps.py", "        window = d.train_all[d.train_all[\"label_available_time\"] > d.train_end - pd.DateOffset(years=years)]", "        window = d.train_all", "data sweep ignores the training window"),
    ("sweeps.py", "        vals = vals + rng.normal(0.0, 1.0, vals.shape) * stds[cols].to_numpy() * noise_sd", "        vals = vals", "noise never added"),
    ("sweeps.py", "    data = SweepData(train_all, keep(train_all), keep(val), keep(ev), f.fit_cutoff, f.train_end)", "    data = SweepData(train_all, keep(train_all), keep(val), ev, f.fit_cutoff, f.train_end)", "evaluation not restricted to the fixed firm sample"),
    ("sweeps.py", "            iv = block_bootstrap((m - self.ref_monthly).to_numpy(), block=12)", "            iv = block_bootstrap((m - self.ref_monthly.mean()).to_numpy(), block=12)", "difference vs HAR not paired by month"),
]


def main(only: str | None = None) -> int:
    missed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for fname, old, new, desc in MUTANTS:
            if only and fname != only:
                continue
            work = Path(tmp) / "mutant"
            if work.exists():
                shutil.rmtree(work)
            shutil.copytree(SRC, work, ignore=shutil.ignore_patterns("results", ".git", "archive", "__pycache__", ".pytest_cache"))
            target = work / "p3_modellab" / fname
            text = target.read_text()
            if text.count(old) != 1:
                print(f"STALE   {desc}: anchor text not found exactly once; update this check")
                missed += 1
                continue
            target.write_text(text.replace(old, new))
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TESTS_FOR.get(fname, [])],
                               cwd=work, capture_output=True, text=True)
            caught = r.returncode != 0
            missed += not caught
            print(f"{'CAUGHT' if caught else 'MISSED'}  {desc}")
    print("all planted bugs caught" if not missed else f"{missed} planted bug(s) not caught")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))  # optional: only check one file, e.g. sweeps.py
