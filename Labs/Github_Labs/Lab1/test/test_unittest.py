"""
Unittest test suite for grade_manager.py
MLOps Lab 1 - IE 7374
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from grade_manager import Student, GradeBook


class TestStudent(unittest.TestCase):

    def setUp(self):
        """Set up a fresh student before each test."""
        self.student = Student("Alice")

    def test_student_name(self):
        self.assertEqual(self.student.name, "Alice")

    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            Student("")

    def test_add_valid_grade(self):
        self.student.add_grade(88)
        self.assertEqual(self.student.get_grade_count(), 1)

    def test_add_invalid_grade_raises(self):
        with self.assertRaises(ValueError):
            self.student.add_grade(110)

    def test_average_no_grades(self):
        self.assertEqual(self.student.get_average(), 0.0)

    def test_average_multiple_grades(self):
        for g in [90, 80, 70]:
            self.student.add_grade(g)
        self.assertEqual(self.student.get_average(), 80.0)

    def test_letter_grade_A(self):
        self.student.add_grade(92)
        self.assertEqual(self.student.get_letter_grade(), "A")

    def test_letter_grade_B(self):
        self.student.add_grade(85)
        self.assertEqual(self.student.get_letter_grade(), "B")

    def test_letter_grade_C(self):
        self.student.add_grade(75)
        self.assertEqual(self.student.get_letter_grade(), "C")

    def test_letter_grade_F(self):
        self.student.add_grade(50)
        self.assertEqual(self.student.get_letter_grade(), "F")

    def test_gpa_A(self):
        self.student.add_grade(95)
        self.assertEqual(self.student.get_gpa(), 4.0)

    def test_gpa_B(self):
        self.student.add_grade(85)
        self.assertEqual(self.student.get_gpa(), 3.0)

    def test_is_passing_true(self):
        self.student.add_grade(75)
        self.assertTrue(self.student.is_passing())

    def test_is_passing_false(self):
        self.student.add_grade(40)
        self.assertFalse(self.student.is_passing())


class TestGradeBook(unittest.TestCase):

    def setUp(self):
        """Set up a gradebook with three students before each test."""
        self.gb = GradeBook("MLOps")
        for name, grade in [("Alice", 93), ("Bob", 75), ("Carol", 52)]:
            s = Student(name)
            s.add_grade(grade)
            self.gb.add_student(s)

    def test_course_name(self):
        self.assertEqual(self.gb.course_name, "MLOps")

    def test_student_count(self):
        self.assertEqual(self.gb.get_student_count(), 3)

    def test_get_student(self):
        self.assertEqual(self.gb.get_student("Bob").name, "Bob")

    def test_missing_student_raises(self):
        with self.assertRaises(KeyError):
            self.gb.get_student("Nobody")

    def test_duplicate_student_raises(self):
        duplicate = Student("Alice")
        duplicate.add_grade(80)
        with self.assertRaises(ValueError):
            self.gb.add_student(duplicate)

    def test_class_average(self):
        # Alice=A(4.0), Bob=C(2.0), Carol=F(0.0) -> avg=2.0
        self.assertEqual(self.gb.get_class_average(), 2.0)

    def test_top_student(self):
        self.assertEqual(self.gb.get_top_student().name, "Alice")

    def test_passing_students_count(self):
        self.assertEqual(len(self.gb.get_passing_students()), 2)

    def test_failing_students_count(self):
        self.assertEqual(len(self.gb.get_failing_students()), 1)

    def test_grade_distribution(self):
        dist = self.gb.get_grade_distribution()
        self.assertEqual(dist["A"], 1)
        self.assertEqual(dist["C"], 1)
        self.assertEqual(dist["F"], 1)

    def test_empty_gradebook_average(self):
        empty_gb = GradeBook("Empty Course")
        self.assertEqual(empty_gb.get_class_average(), 0.0)

    def test_empty_gradebook_top_student_raises(self):
        empty_gb = GradeBook("Empty Course")
        with self.assertRaises(ValueError):
            empty_gb.get_top_student()


if __name__ == "__main__":
    unittest.main()
