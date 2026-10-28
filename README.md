# social-analysis-

A time-aware model of a university cohort's social life: friendships, groups,
romance, conflict and how they change semester by semester. Built for a cohort of
30 AUT mechanical engineering students. It is tested here on a small invented dataset.

```
python run_analysis.py data/example_10p_2sem
```

That reads the six CSV files and writes `outputs/<dataset>/report.html` (an
interactive report) and `results.json` (every number behind it).

## What the pipeline does

| Step | Module | Output |
|---|---|---|
| Load and validate, fill skipped answers | `cohort/io.py` | rating tensors per layer and wave, shared-context counts |
| Describe each wave | `cohort/network.py` | density, reciprocity, transitivity, homophily (E-I), tie turnover, structural balance, groups (Louvain) tracked across waves, roles |
| Explain change | `cohort/models.py` | tie-change regression with bootstrap intervals (a stand-in for RSiena) |
| Simulate | `cohort/simulate.py` | one-step and free-run validation, forecast of the next waves (1,000 runs) |
| Report | `cohort/report.py`, `report_template.html` | findings, predictions and per-person profiles generated from the numbers |

## Input files

Put them in one folder. Every person is a pseudonym. No file contains names.

| File | Columns | One row per |
|---|---|---|
| `waves.csv` | wave, label, date, semester | survey wave |
| `persons.csv` | id, gender, origin, entry_age, extraversion, agreeableness, conscientiousness, neuroticism, openness, attractiveness_peer | person |
| `person_wave.csv` | id, wave, housing, gpa, status | person and wave |
| `foci.csv` | id, focus, type, wave | membership of a context (dorm block, class section, chat, study group, association) at a wave |
| `dyads.csv` | rater, target, wave, layer, value | one person's rating of another on one layer at one wave |
| `events.csv` | date, type, actors, third_parties, intensity, note | dated incident; actors separated by `;` |

Dyad layers: `friendship` 0–5, `trust` 0–5, `hostility` 0–5, `romantic` 0 none / 1 interest / 2 dating / 3 ex,
`contact` 0–5. Ratings are directed: A's rating of B and B's rating of A are separate rows.

## Example dataset

`data/example_10p_2sem/` is synthetic: 10 invented students, 2 semesters, 4 waves.
`examples/make_example_data.py` generates it from a written storyline and adds the noise
real survey answers have (rating bias, recall noise, perception gaps, skipped answers).
The analysis never reads that script, only the CSV files.

## Setup

```
pip install -r requirements.txt
```
