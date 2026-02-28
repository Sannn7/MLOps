"""
Pytest test suite for grade_manager.py
MLOps Lab 1 - IE 7374
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from grade_manager import Student, GradeBook


# ── Student Tests ──────────────────────────

def test_student_creation():
    s = Student("Alice")
    assert s.name == "Alice"
    assert s.get_grade_count() == 0

def test_student_empty_name_raises():
    with pytest.raises(ValueError):
        Student("")

def test_add_grade_valid():
    s = Student("Bob")
    s.add_grade(85)
    assert s.get_grade_count() == 1

def test_add_grade_too_high_raises():
    s = Student("Carol")
    with pytest.raises(ValueError):
        s.add_grade(105)

def test_add_grade_negative_raises():
    s = Student("Dave")
    with pytest.raises(ValueError):
        s.add_grade(-5)

def test_get_average_no_grades():
    s = Student("Eve")
    assert s.get_average() == 0.0

def test_get_average_multiple_grades():
    s = Student("Frank")
    for g in [90, 80, 70]:
        s.add_grade(g)
    assert s.get_average() == 80.0

def test_letter_grade_A():
    s = Student("Grace")
    s.add_grade(95)
    assert s.get_letter_grade() == "A"

def test_letter_grade_B():
    s = Student("Heidi")
    s.add_grade(85)
    assert s.get_letter_grade() == "B"

def test_letter_grade_C():
    s = Student("Ivan")
    s.add_grade(75)
    assert s.get_letter_grade() == "C"

def test_letter_grade_D():
    s = Student("Judy")
    s.add_grade(65)
    assert s.get_letter_grade() == "D"

def test_letter_grade_F():
    s = Student("Karl")
    s.add_grade(55)
    assert s.get_letter_grade() == "F"

def test_gpa_A():
    s = Student("Leo")
    s.add_grade(92)
    assert s.get_gpa() == 4.0

def test_gpa_B():
    s = Student("Mia")
    s.add_grade(83)
    assert s.get_gpa() == 3.0

def test_is_passing_true():
    s = Student("Nina")
    s.add_grade(75)
    assert s.is_passing() is True

def test_is_passing_false():
    s = Student("Oscar")
    s.add_grade(45)
    assert s.is_passing() is False


# ── GradeBook Tests ────────────────────────

def test_gradebook_creation():
    gb = GradeBook("MLOps")
    assert gb.course_name == "MLOps"
    assert gb.get_student_count() == 0

def test_gradebook_empty_name_raises():
    with pytest.raises(ValueError):
        GradeBook("")

def test_add_student():
    gb = GradeBook("MLOps")
    s = Student("Pam")
    s.add_grade(88)
    gb.add_student(s)
    assert gb.get_student_count() == 1

def test_add_duplicate_student_raises():
    gb = GradeBook("MLOps")
    s1 = Student("Quinn")
    s1.add_grade(80)
    gb.add_student(s1)
    s2 = Student("Quinn")
    s2.add_grade(70)
    with pytest.raises(ValueError):
        gb.add_student(s2)

def test_get_student():
    gb = GradeBook("MLOps")
    s = Student("Rose")
    s.add_grade(91)
    gb.add_student(s)
    assert gb.get_student("Rose").name == "Rose"

def test_get_missing_student_raises():
    gb = GradeBook("MLOps")
    with pytest.raises(KeyError):
        gb.get_student("Ghost")

def test_class_average_empty():
    gb = GradeBook("MLOps")
    assert gb.get_class_average() == 0.0

def test_class_average():
    gb = GradeBook("MLOps")
    for name, grade in [("S1", 95), ("S2", 75), ("S3", 55)]:
        s = Student(name)
        s.add_grade(grade)
        gb.add_student(s)
    # A=4.0, C=2.0, F=0.0 -> avg = 2.0
    assert gb.get_class_average() == 2.0

def test_top_student():
    gb = GradeBook("MLOps")
    for name, grade in [("Tom", 70), ("Uma", 95), ("Vic", 80)]:
        s = Student(name)
        s.add_grade(grade)
        gb.add_student(s)
    assert gb.get_top_student().name == "Uma"

def test_top_student_empty_raises():
    gb = GradeBook("MLOps")
    with pytest.raises(ValueError):
        gb.get_top_student()

def test_passing_students_count():
    gb = GradeBook("MLOps")
    for name, grade in [("Win", 80), ("Xena", 30)]:
        s = Student(name)
        s.add_grade(grade)
        gb.add_student(s)
    assert len(gb.get_passing_students()) == 1

def test_failing_students_count():
    gb = GradeBook("MLOps")
    for name, grade in [("Yara", 80), ("Zack", 30)]:
        s = Student(name)
        s.add_grade(grade)
        gb.add_student(s)
    assert len(gb.get_failing_students()) == 1

def test_grade_distribution():
    gb = GradeBook("MLOps")
    for name, grade in [("P1", 95), ("P2", 85), ("P3", 55)]:
        s = Student(name)
        s.add_grade(grade)
        gb.add_student(s)
    dist = gb.get_grade_distribution()
    assert dist["A"] == 1
    assert dist["B"] == 1
    assert dist["F"] == 1
