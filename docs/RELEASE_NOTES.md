# Release notes

## v0.1.0: model and pilot test

This is the first release of the cohort model. It does three things:

- **The model.** It takes per-wave relationship ratings, shared contexts and dated events for a student cohort, and turns them into a description of the network over time, an explanation of why ties changed, a validated simulation and a forecast.
- **The data format.** Six CSV files that a survey can be converted into, plus a tool that turns square survey sheets into the main ratings file.
- **The pilot test.** An invented cohort of 10 students over 2 semesters, run through the full pipeline. The results are checked against the story that was written into the data.

### What is in the repository

| Path | Contents |
|---|---|
| `run_analysis.py` | runs the whole pipeline on one dataset folder |
| `cohort/io.py` | loads and validates the six files, fills skipped answers |
| `cohort/network.py` | per-wave measures, structural balance, groups tracked across waves, roles |
| `cohort/models.py` | tie-change regression with bootstrap intervals |
| `cohort/simulate.py` | agent-based simulation: validation and forecast |
| `cohort/report.py`, `cohort/report_template.html` | findings, predictions, profiles and the interactive report |
| `cohort/convert.py` | survey matrices to `dyads.csv` |
| `data/example_10p_2sem/` | the pilot's invented dataset |
| `examples/make_example_data.py` | the story and the noise that produced it |
| `outputs/example_10p_2sem/` | the pilot's report and results |
| `docs/` | this document's chapters as Markdown, the figures, and the script that builds this PDF |

### Pilot results in brief

- The pipeline ran end to end on 1,785 noisy ratings with 3 skipped answers, in about 10 seconds.
- Of 12 patterns written into the story, it found 8, found 2 partly, missed 1, and 1 could not be tested with this data.
- One step ahead, it predicts friendship ties with half the error of assuming nothing changes. Run freely from the first wave, it reproduces the overall structure but not the conflict cascade of semester 2.
- The forecast for semester 3 includes a 65% chance that P06 and P10 start dating, 45% that P03 and P09 break up, and 84% that P03 ends up with at most one friend.

### Known limits

- Ten people and four waves are enough to test the machinery, not to draw conclusions.
- The tie-change regression stands in for RSiena, which should be used on the full data.
- Six rates in the simulation (side-taking, jealousy, rivalry, fights, hostility decay, balance resolution) are assumptions. The pilot suggests the side-taking and jealousy rates are too low.

### Rebuilding this document

```
python run_analysis.py data/example_10p_2sem
python docs/build_release_pdf.py
```
