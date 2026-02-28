# Lab 1 - MLOps (IE-7374)

## Student Grade Manager

An OOP-based student grade tracking system with GPA computation, letter grade
assignment, and class-level statistics — with full CI/CD via GitHub Actions.

---

## Class Design

### `Student`
| Method | Description |
|---|---|
| `add_grade(grade)` | Add a numeric grade (0-100) |
| `get_average()` | Compute numeric average of all grades |
| `get_letter_grade()` | Return letter grade (A/B/C/D/F) |
| `get_gpa()` | Return GPA on 4.0 scale |
| `is_passing()` | Return True if GPA >= 1.0 |
| `get_grade_count()` | Return number of grades recorded |

### `GradeBook`
| Method | Description |
|---|---|
| `add_student(student)` | Add a Student to the gradebook |
| `get_student(name)` | Retrieve a student by name |
| `get_class_average()` | Average GPA across all students |
| `get_top_student()` | Student with highest numeric average |
| `get_passing_students()` | List of all passing students |
| `get_failing_students()` | List of all failing students |
| `get_grade_distribution()` | Count of students per letter grade |
| `get_student_count()` | Total number of students |

---

## Setup

```bash
python -m venv lab_01
source lab_01/bin/activate     # Mac/Linux
lab_01\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Running Tests

```bash
# Pytest
pytest test/test_pytest.py -v

# Unittest
python -m unittest test.test_unittest -v
```

## CI/CD

GitHub Actions automatically runs both Pytest and Unittest on every push to `main`.
