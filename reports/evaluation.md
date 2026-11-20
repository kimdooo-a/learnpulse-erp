# LearnPulse - evaluation on the synthetic dataset

All numbers below are produced by `python -m learnpulse build` on a **synthetic** school (seed 2026). They show that the pipeline works end to end; they are not results from a real school.

## Dataset
- 1,200 students, 30 classes, 40 weeks, 6 subjects
- ERP rows: 1,105,386 (students 1,200, attendance 240,000, assignments 576,000, assessments 43,200, timetable 186, lms 240,000, fees 4,800)
- Injected trajectories: 180 students (disengagement 45, subject 45, weekday 45, acute 45)
- Injected section slips: 7-A Maths (teacher change week 18); 8-E Science (teacher change week 24)
- Injected late attendance marking: 7-F week 26; 6-E week 34; 6-B week 35

## Model (time-ordered split)
- Backend: lightgbm, 41 features; fit weeks [5, 18], calibration weeks [19, 22], test weeks [29, 34]
- Test rows 7,200, positive rate 0.1013

| Score | PR-AUC | ROC-AUC |
|---|---|---|
| LearnPulse model | 0.5649 | 0.8609 |
| Rules only (severity) | 0.4175 | 0.8062 |
| Attendance below 75% | 0.2064 | 0.6105 |

## Weekly list (top 5 per class, test weeks)

| Ordering | Students listed | On an injected trajectory | Declined within 6 weeks |
|---|---|---|---|
| Priority = 0.6 model + 0.4 rules | 895 | 0.5743 | 0.486 |
| Rules only | 895 | 0.6358 | 0.4034 |

## Recall and lead time (rules-only list, whole year)
- Trajectories measurable: 180
- Listed within 4 weeks of onset: 0.7056
- A ranking signal fired within 4 weeks of onset: 0.9444
- By trajectory: disengagement 0.8222 (n=45); weekday 0.6889 (n=45); subject 0.3111 (n=45); acute 1.0 (n=45)
- Median lead time before the term-end review: 8.0 weeks (151 students)

## Section alerts (S1) and data-quality gate
- S1 alerts: 10-B Science (week 25); 6-F Science (week 13); 7-A Maths (week 18); 8-E Science (week 25); 9-B Hindi (week 30)
- Classes held back as 'data incomplete': 6-B week 35; 6-E week 34; 7-F week 26

## Text fidelity
- Reason texts checked: 26,047; every number found in the fact list: 26,047 (100%)

## Runtime on the build machine (seconds)

| Step | Seconds |
|---|---|
| generate | 0.733 |
| load erp tables | 1.151 |
| connector and arrays | 2.137 |
| features and 12 signals 40 weeks | 0.252 |
| train model | 0.209 |
| score 40 weeks | 2.159 |
| one week score and rank | 0.054 |
| persist results | 1.229 |
