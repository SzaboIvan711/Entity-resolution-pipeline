Massenreferenz — Dedup/Entity Resolution

Проект решает задачу объединения дублей (record linkage / entity resolution) на нормализованных данных:

Blocking → Matching (правила/модель) → Clustering → Canonicalization

Вход: data/clear_data.csv
Выход: out/pairs_pred.csv, out/rows_with_entity_id.csv, out/entities.csv

Структура проекта
data/
  clear_data.csv           # нормализованные входные записи
  pair_model.joblib        # (опц.) обученная модель пар
  pair_model_meta.json     # (опц.) мета к модели: {"features": [...], "threshold": 0.84}

out/
  cand_pairs.csv           # кандидаты после blocking (из blocking.ipynb)
  pairs_pred.csv           # пары, признанные совпадениями (matching)
  rows_with_entity_id.csv  # исходные строки + entity_id
  entities.csv             # канонизированные сущности (по одной строке на entity)

src/
  pipline.py               # end-to-end пайплайн (matching → clustering → canonicalization)
  rules.py                 # нормализация/фичи пары/правила/утилиты
  cluster.py               # сборка кластеров и сводные метрики кластеров
  canonicalize.py          # каноникализация сущностей
  # ноутбуки/скрипты разработки:
  # blocking.ipynb         # построение out/cand_pairs.csv
  # rules_baseline.ipynb   # end-to-end на правилах (демо)
  # model.ipynb            # обучение и оценка модели пар, сохранение артефактов
tests/
  *.py                     # юнит-тесты (pytest)

Установка
python -m venv .venv
# Windows:
. .venv/Scripts/activate
# Mac/Linux:
# source .venv/bin/activate

pip install -r requirements.txt


requirements.txt (минимум):

pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
rapidfuzz>=3.0
joblib>=1.3
pytest>=8


Рекомендуется добавить в корень pytest.ini:

[pytest]
pythonpath = .
testpaths = tests

Быстрый старт

Убедитесь, что есть кандидаты после блокинга: out/cand_pairs.csv
(получается из blocking.ipynb).

Запустите пайплайн:

python -m src.pipline
# или
python src/pipline.py


Ожидаемый лог:

>> load data
>> load candidate pairs
candidates: 318
>> matching
[matching] using model: pair_model.joblib
predicted matches: 315
>> clustering
... summary ...
>> canonicalization
>> save outputs
done.


Результаты в out/:

pairs_pred.csv — предсказанные матч-пары (i, j)

rows_with_entity_id.csv — исходные строки + entity_id (кластер)

entities.csv — «паспорт» сущности (каноникализация)

Как это работает (в двух словах)

Blocking (в blocking.ipynb): строим простые ключи (например, email_domain+zip, phone_last4+zip, name0+zip) → генерируем кандидатов out/cand_pairs.csv.
Метрики блокинга: Pair Completeness / Reduction Ratio / Pairs Quality.

Matching (в src/pipline.py):

если есть data/pair_model.joblib + pair_model_meta.json, используем модель (логрег);

иначе fallback — правила is_match из src/rules.py.
Модель возвращает вероятность proba; порог (threshold) берём из pair_model_meta.json.

Clustering (src/cluster.py): по предсказанным парам строим компоненты связности → присваиваем каждой строке entity_id.

Canonicalization (src/canonicalize.py): на каждый entity_id делаем одну строку:
name=longest, street/city/zip=majority, email/phone=most_frequent_valid (можно менять).

Обучение/обновление модели (опционально)

В ноутбуке model.ipynb:

Собираем датасет пар Xy из кандидатов (out/cand_pairs.csv), считаем pair_features.

Делаем разбиение без утечки по uid, добавляем немного негативов в train/test при необходимости.

Обучаем LogisticRegression, подбираем threshold по PR-кривой (обычно максимум F1 или фиксируем требуемый precision).

Сохраняем:

data/pair_model.joblib (либо {'clf', 'feat_cols', 'threshold'} как dict),

data/pair_model_meta.json, например:

{
  "features": ["name_sim","street_sim","zip_eq","city_eq","email_user_eq","phone_last4_eq"],
  "threshold": 0.84
}


pipline.py сам подхватит эти файлы.

Замечание: uid используется только для обучения/валидации и метрик (true_pairs). Для боевого запуска не требуется.

Тесты

Запуск:

pytest -v


Ожидаемо: 6 passed.

Покрытие:

признаки пары и правило is_match,

построение кластеров и сводка,

каноникализация,

end-to-end smoke на правилах.

Конфигурация и параметры

Вход по умолчанию — data/clear_data.csv (уже нормализованные данные).
Если хотите поддержать сырой вход (raw.csv), можно в начале пайплайна вызывать prepare_aux_cols(df) при отсутствии *_norm колонок.

Порог threshold берётся из pair_model_meta.json.
Чем выше порог — тем выше precision/ниже recall (и наоборот).

Фичи пары перечислены в pair_model_meta.json; считаются в src/rules.py::pair_features.
street_sim в модели масштабируется как в обучении (деление на 100).

Трюки и отладка

pairs_pred == cand_pairs?
Значит matching пропускает всех (слишком мягкие правила или threshold=0.0). Проверь is_match и мета к модели.

Нет rapidfuzz / не видится src?
Убедитесь, что тесты/скрипты запускаете тем же Python, где стоят зависимости.
Добавьте pytest.ini → pythonpath = ..

Кластеры «монстры»?
Смотрите summarize_clusters: размеры, долю топ-uid, cohesion; при необходимости ужмите блоки/ужесточите правила.

Лицензия

<добавьте при необходимости>