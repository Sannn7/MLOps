"""
Student Grade Manager
MLOps Lab 1 - IE 7374
OOP-based grade tracking and GPA computation system.
"""


class Student:
    """Represents a single student with a name and list of grades."""

    GRADE_SCALE = [
        (90, "A"),
        (80, "B"),
        (70, "C"),
        (60, "D"),
        (0,  "F"),
    ]
    PASSING_GPA = 1.0  # D or above

    GPA_MAP = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}

    def __init__(self, name: str):
        if not name or not name.strip():
            raise ValueError("Student name cannot be empty.")
        self.name = name.strip()
        self._grades = []

    def add_grade(self, grade: float) -> None:
        """Add a numeric grade (0-100) for this student."""
        if not (0 <= grade <= 100):
            raise ValueError(f"Grade must be between 0 and 100, got {grade}.")
        self._grades.append(float(grade))

    def get_average(self) -> float:
        """Return the numeric average of all grades."""
        if not self._grades:
            return 0.0
        return round(sum(self._grades) / len(self._grades), 2)

    def get_letter_grade(self) -> str:
        """Return the letter grade based on the numeric average."""
        avg = self.get_average()
        for threshold, letter in self.GRADE_SCALE:
            if avg >= threshold:
                return letter
        return "F"

    def get_gpa(self) -> float:
        """Return the GPA (4.0 scale) based on the letter grade."""
        return self.GPA_MAP[self.get_letter_grade()]

    def is_passing(self) -> bool:
        """Return True if the student GPA meets the passing threshold."""
        return self.get_gpa() >= self.PASSING_GPA

    def get_grade_count(self) -> int:
        """Return the number of grades recorded."""
        return len(self._grades)

    def __repr__(self) -> str:
        return (
            f"Student(name={self.name!r}, avg={self.get_average()}, "
            f"grade={self.get_letter_grade()}, gpa={self.get_gpa()})"
        )


class GradeBook:
    """Manages a collection of students and computes class-level statistics."""

    def __init__(self, course_name: str):
        if not course_name or not course_name.strip():
            raise ValueError("Course name cannot be empty.")
        self.course_name = course_name.strip()
        self._students = {}

    def add_student(self, student: Student) -> None:
        """Add a Student object to the gradebook."""
        if not isinstance(student, Student):
            raise TypeError("Expected a Student instance.")
        if student.name in self._students:
            raise ValueError(f"Student '{student.name}' already exists.")
        self._students[student.name] = student

    def get_student(self, name: str) -> Student:
        """Retrieve a student by name."""
        if name not in self._students:
            raise KeyError(f"Student '{name}' not found.")
        return self._students[name]

    def get_class_average(self) -> float:
        """Return the average GPA across all students."""
        if not self._students:
            return 0.0
        total = sum(s.get_gpa() for s in self._students.values())
        return round(total / len(self._students), 2)

    def get_top_student(self) -> Student:
        """Return the student with the highest numeric average."""
        if not self._students:
            raise ValueError("No students in gradebook.")
        return max(self._students.values(), key=lambda s: s.get_average())

    def get_passing_students(self):
        """Return a list of all passing students."""
        return [s for s in self._students.values() if s.is_passing()]

    def get_failing_students(self):
        """Return a list of all failing students."""
        return [s for s in self._students.values() if not s.is_passing()]

    def get_grade_distribution(self) -> dict:
        """Return a count of students per letter grade."""
        distribution = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for student in self._students.values():
            distribution[student.get_letter_grade()] += 1
        return distribution

    def get_student_count(self) -> int:
        """Return the total number of students."""
        return len(self._students)

    def __repr__(self) -> str:
        return (
            f"GradeBook(course={self.course_name!r}, "
            f"students={self.get_student_count()}, "
            f"class_avg_gpa={self.get_class_average()})"
        )
