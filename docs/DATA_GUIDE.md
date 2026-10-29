# How to put your data in

## Folder and command

One dataset is one folder with six CSV files (UTF-8, comma-separated, header row
first):

```
data/aut_mech_1403/
  waves.csv
  persons.csv
  person_wave.csv
  foci.csv
  dyads.csv
  events.csv
```

Then run:

```
pip install -r requirements.txt
python run_analysis.py data/aut_mech_1403
```

The results go to `outputs/aut_mech_1403/`: `report.html` (open it in a browser) and
`results.json` (every number in the report). On the 10-person pilot a full run takes
about 10 seconds. Add `--runs 2000` for a steadier forecast or `--steps 4` to look
further ahead.

## IDs and privacy

- Give every person a pseudonym such as `P01` to `P30`, and use exactly the same IDs in every file.
- Keep the file that links names to IDs on one offline machine, never in this repository.
- A pseudonym is not enough on its own in a cohort everyone knows. "The girl from Rasht in dorm B with the top GPA" identifies a person. Record origin in broad bands (Tehran / other city, or a region) when that is enough for the question.
- Get informed consent before collecting the romantic and hostility layers, which are the most sensitive. Decide in advance what you will do with answers people give about others who did not consent.

## waves.csv

One row per survey wave, in time order.

| Column | Example | Notes |
|---|---|---|
| wave | `W1` | short ID used in every other file |
| label | `S1 mid (Aban 1403)` | shown in the report |
| date | `2024-11-15` | ISO date the wave refers to; events are assigned to the interval between waves |
| semester | `S1` | waves in the same semester share this; the model compares later semesters with the first |

## persons.csv

One row per person. Attributes that do not change.

| Column | Example | Notes |
|---|---|---|
| id | `P07` | pseudonym |
| gender | `F` / `M` | |
| origin | `Tehran`, `Shiraz` | anything that is not `Tehran` counts as "other city" in the homophily measure |
| entry_age | `18` | |
| extraversion, agreeableness, conscientiousness, neuroticism, openness | `4.2` | Big Five, 1–5, from a short questionnaire such as the BFI-10 |
| attractiveness_peer | `3.6` | 1–5, the **mean of several peers' ratings**, never one person's opinion |

## person_wave.csv

One row per person per wave.

| Column | Example | Notes |
|---|---|---|
| id | `P02` | |
| wave | `W3` | |
| housing | `family`, `dorm`, `rented` | keep the same few words throughout |
| gpa | `18.2` | out of 20; repeat the semester GPA on both waves of a semester |
| status | `active` | `active`, `left` or `graduated` |

## foci.csv

One row per person, context and wave. A context is any place or group that puts people
together regularly.

| Column | Example | Notes |
|---|---|---|
| id | `P05` | |
| focus | `dorm-A`, `dynamics-sec1`, `study-group` | a name for the context |
| type | `dorm`, `class`, `online`, `study`, `rented`, `extracurricular` | the forecast keeps dorms and flats, reshuffles classes, and keeps study and extracurricular groups with some probability |
| wave | `W3` | list the person again at every wave they are in it |

A context that includes the whole cohort, such as the class-wide group chat, is ignored
automatically.

## dyads.csv

The core file: one row per person, per person they rate, per wave, per layer.

| Column | Example | Notes |
|---|---|---|
| rater | `P04` | who answered |
| target | `P08` | who they answered about |
| wave | `W3` | |
| layer | `friendship` | one of `friendship`, `trust`, `hostility`, `romantic`, `contact` |
| value | `4` | whole number in the layer's range |

Anchors to print on the questionnaire:

| Layer | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| friendship | do not know them | acquaintance | friendly | friend | close friend | best friend |
| trust | none | | | would share a private matter | | complete |
| hostility | none | irritation | | open dislike | | enemy |
| romantic | none | interest or crush | dating | ex | | |
| contact | never | rarely | monthly | weekly | several times a week | daily |

For 30 people, 6 waves and 5 layers this file has 30 × 29 × 6 × 5 = 26,100 rows. Nobody
should type that by hand. Collect each layer and wave as a square sheet (first column:
the person answering; header row: the people being rated), save each sheet as
`<layer>_<wave>.csv`, for example `friendship_W1.csv`, and convert the whole folder:

```
python -m cohort.convert path/to/matrices data/aut_mech_1403/dyads.csv
```

Leave a cell empty when someone skipped it. The loader fills a skipped rating with the
other person's rating of the same tie and reports how many it filled.

## events.csv

One row per dated incident.

| Column | Example | Notes |
|---|---|---|
| date | `2025-04-06` | the month is enough; use the 15th if unsure |
| type | `breakup` | see below |
| actors | `P03;P08` | the people the event is between, separated by `;` |
| third_parties | `P09;P04;P10` | others who were pulled in; may be empty |
| intensity | `5` | 1 minor to 5 life-changing |
| note | `right after Nowruz` | optional, free text, no names |

Recognized types:

- **Negative:** breakup, fight, public_fight, cold_shoulder, crush_rejected, gossip, group_dissolved, betrayal
- **Positive:** started_dating, group_formed, support, moved_housing, reconciliation

Other types are kept in the report but not used by the model.

## Collecting three years after the fact

The cohort is three years in, so most data will be recalled, not observed. Some ways to
make recall as reliable as possible:

- **Life-history calendar.** Give each person a timeline with the semesters, exam periods, Nowruz and summers marked, and ask them to place housing moves, relationships and fights on it first. Rate the dyads afterwards, one wave at a time, with the calendar in front of them.
- **Ask both sides.** Every tie is rated by both people. Where they disagree strongly about an event (one says dating, the other says nothing), check the event log and mark the tie as uncertain rather than picking one side.
- **Start with the present.** Recent waves are remembered best. Rate the latest wave first and work backwards.
- **Rate one layer per page.** Mixing friendship and hostility on one page invites people to give one overall impression.

## Checklist before running

- [ ] Every ID in every file appears in `persons.csv`.
- [ ] Every wave in every file appears in `waves.csv`.
- [ ] Values are in range: 0–5, or 0–3 for romantic.
- [ ] Nobody rates themselves (self-ratings are dropped with a warning).
- [ ] Each person appears in `person_wave.csv` at every wave.
- [ ] Event dates fall between the first and last wave dates, or they are ignored by the model.

The loader stops with a clear message on errors (unknown IDs, unknown layers,
out-of-range values) and warns about things it can fix (duplicates, self-ratings,
skipped answers). Warnings are listed at the top of the report.
