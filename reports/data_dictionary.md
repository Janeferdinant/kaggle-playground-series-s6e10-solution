# Dataset Inventory and Data Dictionary

**Competition:** Predicting Airline Satisfaction (`playground-series-s6e10`)  
**Data Acquired:** 5,000 train rows, 1,500 test rows, 1,500 submission rows.

## 1. File Summary

| File | Rows | Columns | Memory Usage |
| :--- | :---: | :---: | :---: |
| `train.csv` | 5,000 | 24 | 1.17 MB |
| `test.csv` | 1,500 | 23 | 0.32 MB |
| `sample_submission.csv` | 1,500 | 2 | 0.06 MB |

## 2. Feature Schema & Missing Value Audit

| Column Name | Type | Train Nulls (%) | Test Nulls (%) |
| :--- | :---: | :---: | :---: |
| `id` | int64 | 0.0% | 0.0% |
| `Gender` | str | 0.0% | 0.0% |
| `Customer Type` | str | 0.0% | 0.0% |
| `Age` | int64 | 0.0% | 0.0% |
| `Type of Travel` | str | 0.0% | 0.0% |
| `Class` | str | 0.0% | 0.0% |
| `Flight Distance` | int64 | 0.0% | 0.0% |
| `Inflight wifi service` | int64 | 0.0% | 0.0% |
| `Departure/Arrival time convenient` | int64 | 0.0% | 0.0% |
| `Ease of Online booking` | int64 | 0.0% | 0.0% |
| `Gate location` | int64 | 0.0% | 0.0% |
| `Food and drink` | int64 | 0.0% | 0.0% |
| `Online boarding` | int64 | 0.0% | 0.0% |
| `Seat comfort` | int64 | 0.0% | 0.0% |
| `Inflight entertainment` | int64 | 0.0% | 0.0% |
| `On-board service` | int64 | 0.0% | 0.0% |
| `Leg room service` | int64 | 0.0% | 0.0% |
| `Baggage handling` | int64 | 0.0% | 0.0% |
| `Checkin service` | int64 | 0.0% | 0.0% |
| `Inflight service` | int64 | 0.0% | 0.0% |
| `Cleanliness` | int64 | 0.0% | 0.0% |
| `Departure Delay in Minutes` | int64 | 0.0% | 0.0% |
| `Arrival Delay in Minutes` | float64 | 1.0% | 1.0% |
| `satisfaction` | str | 0.0% | -% |

## 3. Target Distribution

```
satisfaction
satisfied                  0.72
neutral or dissatisfied    0.28
```

## 4. Integrity Verification

- **ID Alignment:** Verified. Test set IDs exactly match Sample Submission row-by-row.
- **Duplicates:** 0 duplicate rows detected in training data.
- **Data Leakage Risk:** None identified in raw partitions.
