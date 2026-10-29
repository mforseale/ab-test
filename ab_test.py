"""Historical randomized e-mail experiment: design, inference and diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from math import ceil
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_confint, proportion_effectsize


ROOT = Path(__file__).resolve().parent
CONTROL, TREATMENT = "No E-Mail", "Mens E-Mail"
ALPHA, POWER, MDE_USD, ASSUMED_SD = 0.05, 0.8, 0.50, 15.0
BOOTSTRAP_REPS, AA_REPS, SEED = 3000, 500, 42
ORIGINAL_URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
SOURCE_PAGE = "https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html"
PRE_TREATMENT_COVARIATES = {"history"}


def validate_data(frame: pd.DataFrame, expected_rows: int | None = None) -> pd.DataFrame:
    required = {"recency", "history_segment", "history", "mens", "womens", "zip_code",
                "newbie", "channel", "segment", "visit", "conversion", "spend"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {required - set(frame.columns)}")
    clean = frame.copy().reset_index(drop=True)
    if expected_rows is not None and len(clean) != expected_rows:
        raise ValueError(f"Expected {expected_rows} customer rows, found {len(clean)}")
    if clean[list(required)].isna().any().any():
        raise ValueError("Missing observations require investigation")
    numeric = ["recency", "history", "mens", "womens", "newbie", "visit", "conversion", "spend"]
    for column in numeric:
        clean[column] = pd.to_numeric(clean[column], errors="raise")
    if not np.isfinite(clean[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite numeric observations")
    for column in ["mens", "womens", "newbie", "visit", "conversion"]:
        if not clean[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be binary")
    if (clean[["history", "spend"]] < 0).any().any() or not clean.recency.between(1, 12).all():
        raise ValueError("Invalid spending history, outcome or recency")
    if not clean.segment.isin([CONTROL, TREATMENT, "Womens E-Mail"]).all():
        raise ValueError("Unexpected original treatment labels")
    if not {CONTROL, TREATMENT}.issubset(set(clean.segment)):
        raise ValueError("Both original A/B groups are required")
    # This is a technical row ID for the fixed snapshot, not a real customer ID.
    clean["record_id"] = np.arange(len(clean), dtype=np.int64)
    return clean


def load_data(root: Path = ROOT, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    csv_file, metadata_file = raw / "hillstrom.csv", raw / "source.json"
    if refresh or not csv_file.exists():
        request = Request(ORIGINAL_URL, headers={"User-Agent": "ABPortfolio/1.0"})
        with urlopen(request, timeout=45) as response:
            payload = response.read()
        validate_data(pd.read_csv(BytesIO(payload)), expected_rows=64000)
        metadata = {"url": ORIGINAL_URL, "source_page": SOURCE_PAGE,
                    "original_source": "Kevin Hillstrom / MineThatData (2008)",
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "sha256": hashlib.sha256(payload).hexdigest()}
        csv_file.write_bytes(payload)
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    payload = csv_file.read_bytes()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError("The cached source checksum changed")
    return validate_data(pd.read_csv(BytesIO(payload)), expected_rows=64000), metadata


def select_ab(frame: pd.DataFrame) -> pd.DataFrame:
    ab = frame.loc[frame.segment.isin([CONTROL, TREATMENT])].copy()
    ab["group"] = ab.segment.map({CONTROL: "A", TREATMENT: "B"})
    ab["treatment"] = ab.group.eq("B").astype(int)
    if set(ab.group) != {"A", "B"}:
        raise ValueError("Both original A/B arms must be present")
    return ab


def finite_sample(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("Expected at least two finite observations in a one-dimensional sample")
    return array


def welch_effect(test, control, alpha: float = ALPHA) -> dict:
    test, control = finite_sample(test), finite_sample(control)
    result = stats.ttest_ind(test, control, equal_var=False, alternative="two-sided")
    interval = result.confidence_interval(confidence_level=1 - alpha)
    if not np.isfinite([result.pvalue, interval.low, interval.high]).all():
        raise ValueError("Welch inference is undefined for the supplied samples")
    return {"effect": float(test.mean() - control.mean()), "ci_low": float(interval.low),
            "ci_high": float(interval.high), "pvalue": float(result.pvalue),
            "se": float(np.sqrt(test.var(ddof=1) / len(test) + control.var(ddof=1) / len(control)))}


def sample_size_mean(delta: float, sigma: float, alpha=ALPHA, power=POWER) -> int:
    if delta <= 0 or sigma <= 0 or not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("MDE and SD must be positive; alpha and power must be between zero and one")
    estimate = TTestIndPower().solve_power(effect_size=delta / sigma, alpha=alpha,
                                            power=power, ratio=1, alternative="two-sided")
    return ceil(float(estimate))


def plan_experiment(results: Path) -> dict:
    n_per_arm = sample_size_mean(MDE_USD, ASSUMED_SD)
    conversion_effect = abs(proportion_effectsize(0.013, 0.01))
    n_conversion = ceil(float(NormalIndPower().solve_power(effect_size=conversion_effect,
                        alpha=ALPHA, power=POWER, ratio=1, alternative="two-sided")))
    plan = {"alpha": ALPHA, "power": POWER, "allocation": "50/50", "window_days": 14,
            "mde_usd": MDE_USD, "assumed_sd_usd": ASSUMED_SD,
            "n_per_arm_primary": n_per_arm, "n_total_primary": 2 * n_per_arm,
            "conversion_example": {"assumed_baseline": 0.01, "mde_pp": 0.3,
                                   "n_per_arm": n_conversion}}
    (results / "sample_size.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    sensitivity = [{"mde_usd": delta, "assumed_sd_usd": sd,
                    "n_per_arm": sample_size_mean(delta, sd)}
                   for sd in [10., 15., 20.] for delta in [0.25, 0.50, 1.0]]
    pd.DataFrame(sensitivity).to_csv(results / "sample_size_sensitivity.csv", index=False)
    return plan


def bootstrap_difference(test, control, repetitions=BOOTSTRAP_REPS, seed=SEED) -> np.ndarray:
    test, control = finite_sample(test), finite_sample(control)
    if repetitions < 2:
        raise ValueError("Need at least two bootstrap replicates")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for start in range(0, repetitions, 64):
        batch = min(64, repetitions - start)
        test_indices = rng.integers(len(test), size=(batch, len(test)), dtype=np.int32)
        control_indices = rng.integers(len(control), size=(batch, len(control)), dtype=np.int32)
        estimates[start:start + batch] = test[test_indices].mean(axis=1) - control[control_indices].mean(axis=1)
    return estimates


def cuped_adjustment(ab: pd.DataFrame, covariate="history") -> dict:
    if covariate not in PRE_TREATMENT_COVARIATES:
        raise ValueError("CUPED only permits a documented pre-treatment covariate")
    x = finite_sample(ab[covariate])
    y = finite_sample(ab.spend)
    treatment = ab.treatment.to_numpy(dtype=float)
    if set(treatment) != {0., 1.} or not (np.isfinite(treatment).all()):
        raise ValueError("Expected both binary treatment arms")
    scale = x.std(ddof=1)
    if scale <= 0:
        raise ValueError("The pre-treatment covariate has no variance")
    centered = x - x.mean()
    design = np.column_stack([np.ones(len(y)), treatment, centered / scale])
    fitted = sm.OLS(y, design).fit(cov_type="HC3", use_t=True)
    theta = float(fitted.params[2] / scale)
    adjusted = y - theta * centered
    # Regression treatment coefficient equals the difference in CUPED-adjusted means.
    delta = adjusted[treatment == 1].mean() - adjusted[treatment == 0].mean()
    np.testing.assert_allclose(delta, fitted.params[1], atol=1e-10)
    ci = fitted.conf_int(alpha=ALPHA)[1]
    raw_within = sum(np.sum((y[treatment == arm] - y[treatment == arm].mean()) ** 2) for arm in [0, 1])
    adjusted_within = sum(np.sum((adjusted[treatment == arm] - adjusted[treatment == arm].mean()) ** 2) for arm in [0, 1])
    return {"effect": float(fitted.params[1]), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "pvalue": float(fitted.pvalues[1]), "se": float(fitted.bse[1]), "theta": theta,
            "covariate": covariate, "variance_reduction_pct": float(100 * (1 - adjusted_within / raw_within)),
            "history_outcome_correlation": float(np.corrcoef(x, y)[0, 1])}


def hash_group(record_id, salt: str) -> str:
    if not salt:
        raise ValueError("A nonempty experiment salt is required")
    digest = hashlib.sha256(f"{salt}:{record_id}".encode("utf-8")).digest()
    return "A" if int.from_bytes(digest[:8], "big") < 2 ** 63 else "B"


def aa_diagnostics(control: pd.DataFrame, results: Path, repetitions=AA_REPS) -> tuple[dict, np.ndarray]:
    groups = control.record_id.map(lambda key: hash_group(key, "aa-demo-v1"))
    demo = welch_effect(control.loc[groups.eq("B"), "spend"], control.loc[groups.eq("A"), "spend"])
    rng = np.random.default_rng(SEED)
    values = control.spend.to_numpy(dtype=float)
    pvalues = np.empty(repetitions)
    for iteration in range(repetitions):
        shuffled = rng.permutation(values)
        midpoint = len(shuffled) // 2
        pvalues[iteration] = stats.ttest_ind(shuffled[:midpoint], shuffled[midpoint:], equal_var=False).pvalue
    rejected = int((pvalues < ALPHA).sum())
    lower, upper = proportion_confint(rejected, repetitions, alpha=ALPHA, method="wilson")
    summary = {"demo": demo, "demo_a": int(groups.eq("A").sum()), "demo_b": int(groups.eq("B").sum()),
               "repetitions": repetitions, "rejected": rejected, "rejection_rate": rejected / repetitions,
               "rejection_rate_ci_low": float(lower), "rejection_rate_ci_high": float(upper),
               "scope": "Offline resplitting of the original untreated arm, not a live product A/A"}
    (results / "aa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame({"replicate": np.arange(repetitions), "pvalue": pvalues}).to_csv(results / "aa_pvalues.csv", index=False)
    return summary, pvalues


def group_summary(ab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, frame in ab.groupby("group", sort=True):
        n, values = len(frame), frame.spend.to_numpy()
        half_width = stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)
        low, high = proportion_confint(frame.conversion.sum(), n, alpha=ALPHA, method="wilson")
        rows.append({"group": group, "original_segment": frame.segment.iloc[0], "n": n,
                     "buyers": int(frame.conversion.sum()), "revenue_per_customer": float(values.mean()),
                     "revenue_ci_low": float(values.mean() - half_width), "revenue_ci_high": float(values.mean() + half_width),
                     "conversion_rate": float(frame.conversion.mean()), "conversion_ci_low": float(low),
                     "conversion_ci_high": float(high), "visit_rate": float(frame.visit.mean()),
                     "zero_spend_share": float((values == 0).mean())})
    return pd.DataFrame(rows)


def balance_table(ab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ["history", "recency", "mens", "womens", "newbie"]:
        a = ab.loc[ab.group.eq("A"), name].to_numpy(dtype=float)
        b = ab.loc[ab.group.eq("B"), name].to_numpy(dtype=float)
        pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append({"pre_experiment_feature": name, "mean_a": a.mean(), "mean_b": b.mean(),
                     "standardized_mean_difference": (b.mean() - a.mean()) / pooled_sd})
    return pd.DataFrame(rows)


def make_plots(summary, raw, cuped, bootstrap, aa_pvalues, aa, plan, results):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    colors = ["#536b87", "#176bb6"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, mean, low, high, scale, title, unit in [
        (axes[0], "revenue_per_customer", "revenue_ci_low", "revenue_ci_high", 1, "Выручка на участника за 14 дней", "USD"),
        (axes[1], "conversion_rate", "conversion_ci_low", "conversion_ci_high", 100, "Доля совершивших покупку", "%")]:
        values = summary[mean].to_numpy() * scale
        errors = np.vstack([(summary[mean] - summary[low]) * scale, (summary[high] - summary[mean]) * scale])
        axis.bar(["A: без письма", "B: рассылка"], values, color=colors, width=0.55)
        axis.errorbar([0, 1], values, yerr=errors, fmt="none", color="#233548", capsize=6)
        axis.set(title=title, ylabel=unit)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Исходные группы: все назначенные покупатели, 95% интервалы", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(results / "group_metrics.png", dpi=160)
    plt.close(fig)

    low, high = np.quantile(bootstrap, [0.025, 0.975])
    effects = [raw["effect"], raw["effect"], cuped["effect"]]
    lows, highs = [raw["ci_low"], low, cuped["ci_low"]], [raw["ci_high"], high, cuped["ci_high"]]
    fig, axis = plt.subplots(figsize=(10, 4.3))
    for index, (value, lo, hi) in enumerate(zip(effects, lows, highs)):
        axis.errorbar(value, index, xerr=[[value - lo], [hi - value]], fmt="o", color="#176bb6", capsize=5, markersize=7)
    axis.set_yticks([0, 1, 2], ["Welch — основной", "Bootstrap — проверка", "CUPED / ANCOVA, HC3"])
    axis.axvline(0, color="#6b7280", ls="--", label="Нет эффекта")
    axis.axvline(MDE_USD, color="#c27526", ls=":", label="Условный MDE: 0,50 USD")
    axis.set(title="Дополнительная выручка на участника: B − A и 95% интервалы", xlabel="USD за 14 дней")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(results / "effect_intervals.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.hist(aa_pvalues, bins=np.linspace(0, 1, 21), color="#176bb6", edgecolor="white")
    axis.axvline(ALPHA, color="#c27526", lw=2, label="alpha = 0,05")
    axis.axhline(len(aa_pvalues) / 20, color="#536b87", ls="--", label="Ориентир равномерного распределения")
    axis.set(title=f"Учебный A/A: {len(aa_pvalues)} разбиений группы без письма",
             xlabel="p-value Welch", ylabel="Число разбиений")
    axis.text(0.98, 0.94, f"Ложные срабатывания: {aa['rejection_rate']:.1%}", ha="right", va="top", transform=axis.transAxes)
    axis.set_ylim(0, max(np.histogram(aa_pvalues, bins=20, range=(0, 1))[0]) * 1.35)
    axis.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(results / "aa_calibration.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4.5))
    sizes = np.linspace(500, 60000, 100)
    for delta in [0.25, 0.5, 1.0]:
        noncentrality = delta / ASSUMED_SD * np.sqrt(sizes / 2)
        cutoff = stats.norm.ppf(1 - ALPHA / 2)
        powers = stats.norm.sf(cutoff - noncentrality) + stats.norm.cdf(-cutoff - noncentrality)
        axis.plot(sizes, powers, label=f"MDE = {delta:.2f} USD")
    axis.axhline(POWER, color="#6b7280", ls="--", label="Мощность 80%")
    axis.axvline(plan["n_per_arm_primary"], color="#176bb6", ls=":", label="План при MDE = 0,50 USD")
    axis.set(title="План будущего теста: нормальное приближение, SD = 15 USD",
             xlabel="Покупателей в каждой группе", ylabel="Мощность", ylim=(0, 1.02))
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(results / "power_curve.png", dpi=160)
    plt.close(fig)


def write_report(frame, ab, summary, raw, mw, cuped, bootstrap, aa, plan, checks, source, results):
    control, test = summary.set_index("group").loc["A"], summary.set_index("group").loc["B"]
    relative = 100 * raw["effect"] / control.revenue_per_customer
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    if checks["srm_pvalue"] < 0.001:
        decision = "Обнаружен SRM: продуктовое решение по эффекту откладывается до выяснения причин несоответствия размеров групп."
    elif raw["pvalue"] < ALPHA and raw["effect"] > 0:
        decision = "По основной метрике есть статистические свидетельства роста выручки от рассылки. Прибыльность и допустимость масштабирования требуют данных о затратах, марже и негативной реакции клиентов."
    else:
        decision = "Основной анализ не даёт достаточных свидетельств положительного влияния рассылки на выручку. Отсутствие значимости не доказывает отсутствия эффекта."
    mde_comment = ("Нижняя граница интервала выше условного MDE: данные поддерживают эффект как минимум такого размера."
                   if raw["ci_low"] > MDE_USD else
                   "95% интервал включает эффекты ниже условного MDE: превышение бизнес-порога не установлено, даже если точечная оценка больше него.")
    table = "\n".join(f"| {row.group}: {row.original_segment} | {row.n:,} | {row.revenue_per_customer:.3f} | {100 * row.conversion_rate:.3f}% | {100 * row.visit_rate:.2f}% |" for row in summary.itertuples())
    report = f"""# Приносит ли рассылка дополнительную выручку?

## Решение

{decision}

Выручка на назначенного покупателя выросла на **{raw['effect']:.3f} USD** ({relative:+.1f}% к контролю); 95% CI **[{raw['ci_low']:.3f}; {raw['ci_high']:.3f}] USD**, Welch p-value = **{raw['pvalue']:.5g}**. Это эффект письма с товарами мужской коллекции против отсутствия письма за 14 дней в историческом эксперименте, без переноса на другую аудиторию.

{mde_comment}

## Данные и группы

[Hillstrom / MineThatData](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html), 2008 год. Использованы {len(ab):,} строк из {len(frame):,}. Владелец данных описывает случайное назначение. Исходные группы не пересоздаются; `Womens E-Mail` исключена заранее по плану, а не по результатам. Название рассылки не обозначает пол участника.

| Группа | Участников | Выручка на участника, USD | Купили за 14 дней | Посетили сайт |
| --- | ---: | ---: | ---: | ---: |
{table}

Единица анализа — покупатель. Считаем всех назначенных, включая нулевые траты; средний чек только среди купивших не заменяет основную метрику. В исходном файле нет настоящего user_id: `record_id` — номер строки снимка для воспроизводимости. Нельзя независимо проверить повторное участие одного человека. Одинаковые строки признаков не удалены автоматически.

SHA-256 CSV: `{source['sha256']}`. Проверены схема, 64 000 строк, отсутствие пропусков/нечисловых значений, бинарные индикаторы и неотрицательные траты. Технические диагностики: {checks['exact_feature_duplicates']} полностью одинаковых строк исходных признаков; {checks['spend_conversion_mismatches']} несовпадений признака покупки с положительными тратами. Исходный файл сохранён без изменений.

SRM для выбранных A/B-групп при ожидаемом 1:1: p = {checks['srm_pvalue']:.4f}; диагностический порог = 0.001. Это проверка размеров, а не доказательство корректности назначения. Баланс предэкспериментальных признаков дан в `pre_experiment_balance.csv` через standardized mean difference без перебора p-value.

![Метрики групп](group_metrics.png)

## План аналогичного будущего эксперимента

Основная гипотеза H0: средняя выручка B − A равна нулю. Альтернатива двусторонняя; alpha = 5%, power = 80%. При учебных допущениях SD = 15 USD и MDE = 0.50 USD нужны **{plan['n_per_arm_primary']:,} покупателей на группу**, **{plan['n_total_primary']:,} всего**. Эти допущения записаны до анализа результатов и не выдаются за измеренные параметры компании. Расчёт t-test power предполагает общий SD; для реального запуска его нужно оценить на независимой истории и проверить достаточность выборки при распределении выручки с большим числом нулей.

Для дополнительного примера конверсии 1% → 1.3% (+0.3 п.п.) нужны {plan['conversion_example']['n_per_arm']:,} участников на группу при тех же alpha/power. Это отдельный сценарий дизайна, не замена основной метрики. Таблица чувствительности к MDE и SD сохранена в `sample_size_sensitivity.csv`.

Нужно набрать выборку и дождаться полных 14 дней для последнего включённого покупателя. Длительность набора без данных о трафике неизвестна. Досрочная остановка по ежедневному p-value не предусмотрена. Наблюдаемая мощность по найденному эффекту не рассчитывается. Исторический эксперимент был проведён автором; здесь разработан план аналогичного теста.

![Мощность и размер выборки](power_curve.png)

## t-test, Mann–Whitney и bootstrap

- **Основной метод:** Welch t-test для разницы средних, без предположения равных дисперсий. Нули и крупные покупки не исключаются.
- **Mann–Whitney:** p = {mw['pvalue']:.5g}; U/(nA × nB) = {mw['probability_superiority']:.5f}. Использована асимптотическая поправка на совпадающие ранги. Этот результат касается рангов/распределений, а не размера средней дополнительной выручки. Метод не выбран вместо Welch по величине p-value.
- **Bootstrap:** {BOOTSTRAP_REPS} перевыборок покупателей отдельно внутри каждой группы; 95% percentile CI разницы средних [{low:.3f}; {high:.3f}] USD. Это дополнительная оценка неопределённости, без выбора удобного интервала.
- Конверсия и посещения — вторичные описательные метрики. Их интервалы в таблицах точечные, без коррекции множественных сравнений; решение по ним отдельно не принимается.

## CUPED / ковариатная корректировка

Ковариата `history` — сумма покупок за год до эксперимента. Использована регрессия `spend ~ treatment + centered_history` с HC3-ошибками. Её treatment-коэффициент равен разнице средних после вычитания общей линейной зависимости от предыстории (CUPED / ANCOVA). Оценка theta учитывается через регрессионный расчёт неопределённости; постэкспериментальные `visit`, `conversion` и `spend` не могут выступать ковариатами.

Эффект после корректировки: {cuped['effect']:.3f} USD; 95% CI [{cuped['ci_low']:.3f}; {cuped['ci_high']:.3f}], p = {cuped['pvalue']:.5g}. Корреляция history и spend: {cuped['history_outcome_correlation']:.4f}; уменьшение объединённой внутригрупповой дисперсии: {cuped['variance_reduction_pct']:.2f}%. Эффект метода на этих данных измерен, а не обещан заранее. Исходный Welch остаётся основным анализом.

![Разница средних и интервалы](effect_intervals.png)

## Учебный A/A и сплит с солью

Внутри исходного `No E-Mail` выполнен детерминированный SHA-256-сплит с солью `aa-demo-v1`: A = {aa['demo_a']:,}, B = {aa['demo_b']:,}, Welch p = {aa['demo']['pvalue']:.4f}. Ни одна из этих групп письма не получала. Соль не подбиралась под p-value. Этот сплит не заменяет исходное A/B-назначение и не выдаётся за live A/A.

В {aa['repetitions']} дополнительных случайных разбиениях контроля получено {aa['rejected']} p-value ниже 0.05: частота {aa['rejection_rate']:.1%}, 95% Wilson CI [{aa['rejection_rate_ci_low']:.1%}; {aa['rejection_rate_ci_high']:.1%}]. Это калибровочная диагностика на эмпирическом распределении контроля; она не доказывает контроль ошибки первого рода при любых распределениях и не проверяет исходную систему назначения компании. Все повторения сохранены в `aa_pvalues.csv`.

![A/A-диагностика](aa_calibration.png)

## Продуктовая интерпретация и ограничения

Разница с контрольной группой оценивает дополнительные покупки относительно органического поведения, а не приписывает письму всю выручку получателей. Для вывода о прибыли нужны маржа, стоимость контакта и сопутствующие расходы. При условной марже m прирост вклада на контакт оценивается как `m × дополнительная выручка − стоимость контакта`; параметры в данных отсутствуют.

Отписок, жалоб, возвратов и долгосрочного удержания нет. Их следует добавить как защитные метрики нового запуска. Из архива нельзя подтвердить фактическую доставку письма или отсутствие иных воздействий; анализ следует исходному назначению. Вывод относится к двухнедельному окну и этой аудитории. План анализа создан сейчас для повторного исследования, он не был регистрацией до исторического запуска 2008 года.

## Формулировка для анкеты

«Проанализировал реальный рандомизированный эксперимент с CRM-рассылкой: сравнил {len(ab):,} покупателей в тесте и контроле, оценил дополнительную выручку на участника с помощью Welch t-test и bootstrap, применил Mann–Whitney с отдельной интерпретацией. Рассчитал выборку для аналогичного будущего эксперимента, проверил SRM и баланс предыстории, реализовал учебный A/A и CUPED-корректировку. Эффект по выручке составил {raw['effect']:.2f} USD на участника за 14 дней, 95% CI [{raw['ci_low']:.2f}; {raw['ci_high']:.2f}]; описал ограничения масштабирования».
"""
    (results / "report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Download the original public CSV again")
    args = parser.parse_args()
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    plan = plan_experiment(results)
    print(f"Planned sample: {plan['n_per_arm_primary']} per arm", flush=True)
    frame, metadata = load_data(refresh=args.refresh)
    ab = select_ab(frame)
    processed = ROOT / "data" / "processed"
    processed.mkdir(exist_ok=True)
    ab.to_csv(processed / "ab_customers.csv", index=False)
    summary = group_summary(ab)
    summary.to_csv(results / "group_metrics.csv", index=False)
    balance_table(ab).to_csv(results / "pre_experiment_balance.csv", index=False)
    control = ab.loc[ab.group.eq("A")]
    test = ab.loc[ab.group.eq("B")]
    raw = welch_effect(test.spend, control.spend)
    ranked = stats.mannwhitneyu(test.spend, control.spend, alternative="two-sided", method="asymptotic")
    mw = {"u_statistic": float(ranked.statistic), "pvalue": float(ranked.pvalue),
          "probability_superiority": float(ranked.statistic / (len(test) * len(control)))}
    print(f"A = {len(control)}, B = {len(test)}; Welch effect = {raw['effect']:.6f}", flush=True)
    print("Bootstrap: customer-level resampling", flush=True)
    bootstrap = bootstrap_difference(test.spend, control.spend)
    pd.DataFrame({"delta_revenue_usd": bootstrap}).to_csv(results / "bootstrap_effects.csv", index=False)
    cuped = cuped_adjustment(ab)
    print("A/A: offline calibration on the untreated arm", flush=True)
    aa, pvalues = aa_diagnostics(control, results)
    counts = [len(control), len(test)]
    checks = {"srm_pvalue": float(stats.chisquare(counts).pvalue),
              "exact_feature_duplicates": int(frame.drop(columns="record_id").duplicated().sum()),
              "spend_conversion_mismatches": int((frame.spend.gt(0) != frame.conversion.eq(1)).sum()),
              "original_assignment_preserved": bool((ab.treatment.eq(1) == ab.segment.eq(TREATMENT)).all())}
    analysis = {"welch": raw, "mann_whitney": mw, "cuped": cuped,
                "bootstrap_ci": list(map(float, np.quantile(bootstrap, [0.025, 0.975]))),
                "checks": checks, "source_sha256": metadata["sha256"], "seed": SEED,
                "bootstrap_repetitions": BOOTSTRAP_REPS, "aa_repetitions": AA_REPS,
                "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (results / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    make_plots(summary, raw, cuped, bootstrap, pvalues, aa, plan, results)
    write_report(frame, ab, summary, raw, mw, cuped, bootstrap, aa, plan, checks, metadata, results)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps({"welch": raw, "cuped": cuped, "srm_pvalue": checks["srm_pvalue"]}, indent=2), flush=True)
    print(f"Report: {results / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
