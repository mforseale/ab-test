"""Generate and execute the explanatory A/B analysis notebook."""

import json
import os
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent


def main():
    if not (ROOT / "results/analysis.json").exists():
        raise SystemExit("Run ab_test.py before building the notebook")
    kernel_root = ROOT / ".jupyter"
    kernel_dir = kernel_root / "kernels" / "ab-test"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    specification = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                     "display_name": "A/B test (project environment)", "language": "python"}
    (kernel_dir / "kernel.json").write_text(json.dumps(specification), encoding="utf-8")
    previous = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(kernel_root) + (os.pathsep + previous if previous else "")
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook = nbf.v4.new_notebook(cells=[
        md("# Рассылка приносит дополнительные покупки или получает органическую выручку?\n\nРеальный рандомизированный эксперимент [Hillstrom / MineThatData](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html): покупатели, письмо и контроль, двухнедельное окно. Методика подходит для задач CRM продуктовой команды. Это повторный анализ исторического эксперимента."),
        md("## 1. Продуктовый вопрос и план\n\nОсновная метрика — **выручка на каждого назначенного участника за 14 дней**, включая нулевые траты. Основная проверка — двусторонний Welch, alpha = 5%.\n\nСравнение `Mens E-Mail` и `No E-Mail` выбрано заранее. `Mens` обозначает коллекцию в письме, а не пол покупателя. Группы не пересоздаются. [План анализа](docs/design.md) сохранён до просмотра результатов; исходный эксперимент уже был проведён автором в 2008 году."),
        code("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import Image, Markdown, display\nfrom ab_test import load_data, select_ab, welch_effect, hash_group\n\nROOT = Path.cwd()\nframe, source = load_data(ROOT)\nab = select_ab(frame)\nprint('Строк в источнике:', len(frame), '; в выбранном A/B:', len(ab))\nprint('SHA-256:', source['sha256'])\nassert (ab.treatment.eq(1) == ab.segment.eq('Mens E-Mail')).all()\ndisplay(frame.segment.value_counts().to_frame('Участников исходного эксперимента'))"),
        md("## 2. Метрика и единица анализа\n\nОдна строка — один покупатель по описанию автора. Считаем всех назначенных: исключение некупивших поменяло бы вопрос и создало отбор по результату воздействия. `record_id` — технический номер строки фиксированного CSV, не реальный пользовательский идентификатор. Одинаковые признаки разных строк автоматически не удаляются."),
        code("summary = pd.read_csv(ROOT / 'results/group_metrics.csv')\nanalysis = json.loads((ROOT / 'results/analysis.json').read_text(encoding='utf-8'))\ndisplay(summary[['group', 'original_segment', 'n', 'buyers', 'revenue_per_customer', 'conversion_rate', 'visit_rate', 'zero_spend_share']])\nprint('Проверки качества:', analysis['checks'])\ndisplay(Image(filename=str(ROOT / 'results/group_metrics.png')))"),
        md("## 3. Планируем аналогичный будущий эксперимент\n\nH0: средняя выручка B − A = 0. MDE = 0,50 USD, SD = 15 USD, alpha = 5%, power = 80%, 50/50. Это условные допущения, зафиксированные до анализа; они не выданы за реальные исторические параметры.\n\nРазмер выборки показывает потребность в участниках. Длительность набора требует прогноза трафика; после набора последнего участника нужны полные 14 дней. По результату исторического теста «наблюдаемая мощность» не рассчитывается."),
        code("plan = json.loads((ROOT / 'results/sample_size.json').read_text(encoding='utf-8'))\nprint('Покупателей на группу:', plan['n_per_arm_primary'])\nprint('Всего:', plan['n_total_primary'])\nprint('Пример для конверсии 1% → 1,3%:', plan['conversion_example'])\ndisplay(pd.read_csv(ROOT / 'results/sample_size_sensitivity.csv'))\ndisplay(Image(filename=str(ROOT / 'results/power_curve.png')))"),
        md("## 4. Диагностика назначения\n\nSRM проверяет размеры групп против ожидаемого 1:1; порог диагностики = 0,001. Баланс `history`, `recency` и других признаков до письма показываем через standardized mean difference. Отсутствие SRM не доказывает корректность исходной системы назначения."),
        code("print('SRM p-value:', analysis['checks']['srm_pvalue'])\ndisplay(pd.read_csv(ROOT / 'results/pre_experiment_balance.csv'))\nprint('Демонстрация SHA-256 с солью; этот сплит НЕ заменяет исходные группы:')\ndisplay(pd.DataFrame({'Технический ID': range(5), 'Учебная группа': [hash_group(i, 'example-v1') for i in range(5)]}))"),
        md("## 5. Welch, Mann–Whitney и bootstrap\n\n**Welch** отвечает на основной вопрос о разнице средних. **Mann–Whitney** проверяет ранговое различие распределений; без дополнительных условий он не является тестом средней выручки или медиан. Много совпадающих нулей учитывается асимптотической поправкой.\n\n**Bootstrap** перевыбирает целых покупателей отдельно внутри каждой группы. Три подхода не используются для поиска удобного p-value; решение заранее привязано к Welch. Конверсия и визиты — вторичные описательные метрики без самостоятельного решения по значимости."),
        code("a = ab.loc[ab.group.eq('A'), 'spend']\nb = ab.loc[ab.group.eq('B'), 'spend']\ncalculated = welch_effect(b, a)\nassert abs(calculated['effect'] - analysis['welch']['effect']) < 1e-10\nprint('Welch:', analysis['welch'])\nprint('Mann–Whitney:', analysis['mann_whitney'])\nprint('Bootstrap 95% CI разницы средних:', analysis['bootstrap_ci'])"),
        md("## 6. CUPED: можно ли убрать часть шума?\n\n`history` содержит траты за год **до** рассылки. Оцениваем общую линейную зависимость от этой предыстории: `Y_adj = Y − theta × (history − mean(history))`. Коэффициент treatment в регрессии `spend ~ treatment + centered_history` даёт разницу скорректированных средних; HC3 учитывает неодинаковую дисперсию и оценку коэффициентов.\n\n`visit`, `conversion` и текущий `spend` нельзя подставлять как ковариаты: они измерены после воздействия. CUPED здесь — анализ чувствительности; исходный Welch не заменяется после просмотра результата."),
        code("cuped = analysis['cuped']\ndisplay(pd.DataFrame([{'Метод': name, **analysis[key]} for name, key in [('Welch', 'welch'), ('CUPED / ANCOVA HC3', 'cuped')]]))\nprint('Уменьшение внутригрупповой дисперсии, %:', cuped['variance_reduction_pct'])\ndisplay(Image(filename=str(ROOT / 'results/effect_intervals.png')))"),
        md("## 7. Учебный A/A\n\nИспользуем только исходную группу без письма. Один пример назначается стабильным хэшем с фиксированной солью, затем 500 случайных разбиений дают эмпирическую частоту p-value < 0,05. Публикуются все повторы; соль и seed не подбираются.\n\nЭто офлайн-диагностика расчёта, а не live A/A и не доказательство корректности реального назначения. Отдельный значимый A/A возможен случайно; p > 0,05 не доказывает эквивалентности групп."),
        code("aa = json.loads((ROOT / 'results/aa_summary.json').read_text(encoding='utf-8'))\nprint('Хэш-сплит:', aa['demo_a'], aa['demo_b'], '; p-value:', aa['demo']['pvalue'])\nprint('Повторов:', aa['repetitions'], '; ложных срабатываний:', aa['rejected'])\nprint('Частота:', aa['rejection_rate'], '; 95% CI:', aa['rejection_rate_ci_low'], aa['rejection_rate_ci_high'])\ndisplay(Image(filename=str(ROOT / 'results/aa_calibration.png')))"),
        md("## 8. Что решаем по результатам\n\nОтделяем статистическое свидетельство эффекта от превышения условного бизнес-порога и прибыльности. Если 95% интервал включает эффект меньше 0,50 USD, достижение MDE не доказано. Стоимость контакта, маржа, жалобы, отписки и долгосрочное удержание отсутствуют."),
        code("report = (ROOT / 'results/report.md').read_text(encoding='utf-8')\ndisplay(Markdown(report.split('## Решение', 1)[1].split('## Данные и группы', 1)[0]))"),
        md("## 9. Вопросы для собеседования\n\n- Почему нельзя сравнивать только средние чеки совершивших покупку?\n- Чем отличается рост выручки получателей от инкрементального эффекта?\n- Зачем различать статистическую значимость и MDE?\n- Почему Mann–Whitney нельзя автоматически назвать проверкой средней выручки?\n- Когда CUPED помогает и откуда должна браться ковариата?\n- Чем офлайн A/A отличается от реального эксперимента на продукте?\n\n[Полный отчёт и формулировка для анкеты](results/report.md)."),
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": sys.version.split()[0]}})
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook_cell_{index}", "exec")
    NotebookClient(notebook, timeout=180, kernel_name="ab-test",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.validate(notebook)
    nbf.write(notebook, ROOT / "analysis.ipynb")
    print(f"Executed notebook: {ROOT / 'analysis.ipynb'}")


if __name__ == "__main__":
    main()
