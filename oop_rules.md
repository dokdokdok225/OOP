# Comprehensive Guide to Object-Oriented Programming (OOP) Rules & Architecture

**Course:** DSC 2135 – Object Oriented Programming  
**Institution:** Ateneo de Davao University  
**Context:** Lecture Reference & Implementation Standards  
**Motto:** *Fortes-in-Fide*

---

## Table of Contents
1. [Paradigm Foundations: Procedural vs. Object-Oriented](#1-paradigm-foundations-procedural-vs-object-oriented)
2. [Core OOP Vocabulary and Mechanics](#2-core-oop-vocabulary-and-mechanics)
3. [Protecting Object State & Encapsulation](#3-protecting-object-state--encapsulation)
4. [Access Conventions & Python Name Mangling](#4-access-conventions--python-name-mangling)
5. [Validation & Class Invariants](#5-validation--class-invariants)
6. [Controlled Access: Property Getters and Setters](#6-controlled-access-property-getters-and-setters)
7. [Instance Attributes vs. Class Attributes](#7-instance-attributes-vs-class-attributes)
8. [Inheritance Hierarchies](#8-inheritance-hierarchies)
9. [The `super()` Function & Method Overriding](#9-the-super-function--method-overriding)
10. [Method Resolution Order (MRO) & Multiple Inheritance](#10-method-resolution-order-mro--multiple-inheritance)
11. [Abstract Classes & Interfaces (`abc` Module)](#11-abstract-classes--interfaces-abc-module)
12. [The Complete Refactored Case Study (`StudentRecord`)](#12-the-complete-refactored-case-study-studentrecord)
13. [Master OOP Design Rules Checklist](#13-master-oop-design-rules-checklist)

---

## 1. Paradigm Foundations: Procedural vs. Object-Oriented

Programming paradigms dictate how developers model real-world domains into software systems.

```
Procedural Architecture:
[ Input Data ] ---> [ Process Data ] ---> [ Call Functions ] ---> [ Produce Output ]
(Functions and data are distinct entities; state is passed around)

Object-Oriented Architecture:
+-------------------------------------------------------------+
|                      Student Object                         |
|  [State / Attributes]: ID, Name, Program, Scores            |
|  [Behavior / Methods]: add_score(), calculate_average(),    |
|                        get_status(), display_summary()      |
+-------------------------------------------------------------+
(Data and the actions that operate on that data are co-located)
```

### Procedural vs. Object-Oriented Comparison

| Dimension | Procedural Programming | Object-Oriented Programming (OOP) |
| :--- | :--- | :--- |
| **Core Organizing Unit** | Functions, procedures, step-by-step sequences. | Classes, objects, interacting entities. |
| **Guiding Question** | *"What operations should execute next?"* | *"Who is responsible for this operation?"* |
| **Data & Logic Coupling** | Separated; records are passed into routines. | Encapsulated; behavior and state live together. |
| **State Maintenance** | Vulnerable to unintended global/external mutation. | Guarded by internal invariants and methods. |
| **Scalability** | Becomes fragile as variable counts grow large. | Highly scalable through modular responsibility. |
| **Primary Use Cases** | Scripting, simple transformations, linear pipes. | Large-scale systems, domain entities, modeling. |

---

## 2. Core OOP Vocabulary and Mechanics

### The Core Terminology
- **Class:** A template, definition, or blueprint detailing the data an entity holds and the actions it can perform.
- **Object (Instance):** A concrete realization created from a class template.
- **Attribute:** A named data value associated with a class or instance.
- **State:** The current collection of attribute values held inside a specific object at a given moment.
- **Method:** A function defined inside a class that operates on the object's state.
- **Behavior:** The set of actions and calculations an object can execute.
- **Identity:** The distinct runtime reference differentiating an object from all other instances, even if their data attributes are identical (`id(obj_a) != id(obj_b)` or `obj_a is not obj_b`).
- **Constructor / Initializer (`__init__`):** A special initialization method that sets up initial state upon instantiation.

### The Role of `self`
In Python, `self` explicitly represents the **current instance** executing the method:
```python
class StudentRecord:
    def __init__(self, name: str):
        # 'name' is the parameter passed into the function
        # 'self.name' is the instance attribute stored on the object
        self.name = name

student_1 = StudentRecord("Anna")
student_2 = StudentRecord("Mark")

# During student_1 execution: self -> student_1
# During student_2 execution: self -> student_2
```

---

## 3. Protecting Object State & Encapsulation

### The Direct Attribute Access Flaw
Allowing outside code to modify an object's internal fields directly bypasses domain safety:

```python
# Unprotected Class
class StudentRecord:
    def __init__(self, student_id, name, program):
        self.student_id = student_id
        self.name = name
        self.program = program
        self.scores = []

student = StudentRecord("DS001", "Anna Cruz", "BS Data Science")

# External code can introduce logically invalid data silently:
student.name = ""
student.program = 12345
student.scores.append(-20)
student.scores.append(150)
```

> **The Execution Paradox:**  
> A program can exhibit zero syntax errors, zero indentation errors, and zero runtime exceptions while remaining **fundamentally incorrect** because its state is invalid.

### What Encapsulation Really Means
Encapsulation is not merely "data hiding." It is the practice of:
1. **Grouping** related data and behaviors together.
2. **Hiding** implementation details that callers do not need to know.
3. **Controlling access** so mutations only occur through validated channels.
4. **Preserving consistency** across multiple interrelated attributes.

---

## 4. Access Conventions & Python Name Mangling

Python does not enforce absolute access restriction keywords (such as `private` or `protected` in C++ or Java). Instead, it relies on standardized conventions:

| Convention | Syntax Pattern | Intended Visibility | Meaning & Developer Contract |
| :--- | :--- | :--- | :--- |
| **Public** | `self.name` | External & Internal | Freely accessible; intended for general use. |
| **Internal / Protected** | `self._name` | Internal by convention | Signals: *"Implementation detail. Do not touch directly."* |
| **Private / Mangled** | `self.__name` | Class-private | Activates runtime name mangling (`_ClassName__name`) to prevent name clashes in inheritance. |

```python
class Account:
    def __init__(self, owner: str, balance: float):
        self.owner = owner          # Public
        self._balance = balance     # Protected by convention
        self.__secret_key = "k99"   # Name-mangled

acc = Account("Maria", 1000.0)
print(acc.owner)        # Works: Maria
print(acc._balance)     # Works, but violates architectural convention
# print(acc.__secret_key) -> Raises AttributeError
print(acc._Account__secret_key)  # Accessible only via mangled identifier
```

---

## 5. Validation & Class Invariants

### Class Invariants Defined
> **Class Invariant:**  
> A condition or rule that must evaluate to `True` for every valid object across its entire lifecycle—immediately after creation and before/after any method execution.  
> *(Bisaya: "Mao ni ang rules nga dapat dili mabungkag para valid gihapon ang object.")*

### Implementation Rules for Invariants:
1. **Validate Before Modifying:** Never alter internal state before verifying input validity.
2. **Leave State Unchanged on Failure:** If incoming input violates a rule, reject it completely; do not allow partial or corrupted state updates.
3. **Type and Range Checks:** Verify data types first (`isinstance`), then verify domain boundaries (e.g., $0 \le \text{score} \le 100$).

```python
def add_score(self, score: float) -> bool:
    # 1. Type Validation
    if not isinstance(score, (int, float)):
        print("Score must be numeric.")
        return False
    
    # 2. Range Validation (Invariant: 0 <= score <= 100)
    if not (0 <= score <= 100):
        print("Score must be between 0 and 100.")
        return False
    
    # 3. State update only after all checks pass
    self._scores.append(score)
    return True
```

---

## 6. Controlled Access: Property Getters and Setters

The `@property` decorator allows methods to be accessed with attribute-style dot notation (`obj.attribute` instead of `obj.get_attribute()`), pairing clean syntax with strict validation.

```python
class StudentRecord:
    def __init__(self, name: str):
        self.name = name  # Invokes the setter below

    @property
    def name(self) -> str:
        """Getter: Controls how the attribute is read."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Setter: Controls how the attribute is updated."""
        if not isinstance(value, str):
            raise TypeError("Name must be a string.")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Name cannot be empty or whitespace.")
        self._name = cleaned
```

### Defensive Copying for Mutable Collections
Exposing internal lists or dictionaries directly allows external code to mutate them behind the class's back. Return **immutable copies** (e.g., `tuple`) instead:

```python
# VULNERABLE:
@property
def scores(self):
    return self._scores  # External code can call .append(9999) directly!

# DEFENSIVE / IMMUTABLE:
@property
def scores(self) -> tuple:
    return tuple(self._scores)  # Returns read-only tuple; prevents .append()
```

---

## 7. Instance Attributes vs. Class Attributes

```
+--------------------------------------------------------+
|                  Class: StudentRecord                  |
|  passing_score = 75  <--- (Shared Class Attribute)    |
+--------------------------------------------------------+
           /                                 \
          /                                   \
+-----------------------+           +-----------------------+
|  Instance: student_1  |           |  Instance: student_2  |
|  _name = "Anna Cruz"  |           |  _name = "Mark Reyes" |
|  _scores = [88, 91]   |           |  _scores = [70, 72]   |
+-----------------------+           +-----------------------+
```

| Dimension | Instance Attribute | Class Attribute |
| :--- | :--- | :--- |
| **Ownership** | Bound to a single, specific object instance. | Bound to the class definition itself. |
| **Declaration** | Defined inside methods using `self.variable`. | Defined directly inside the class body. |
| **Storage** | Stored in the instance dictionary (`__dict__`). | Stored in the class dictionary (`__dict__`). |
| **Purpose** | Captures individual entity state (e.g., student name). | Captures shared rules, constants, or defaults (e.g., passing mark). |
| **Access** | `self.attribute` or `instance.attribute`. | `ClassName.attribute` or `self.attribute`. |

---

## 8. Inheritance Hierarchies

Inheritance allows a new class (**derived/child class**) to inherit properties and behaviors from an existing class (**base/parent class**), promoting code reuse and establishing polymorphic relationships.

```python
class DerivedClassName(BaseClassName):
    """DerivedClassName inherits attributes and methods from BaseClassName."""
    pass
```

### The Four Major Forms of Inheritance

```
1. Single                  2. Multilevel
   [ Base ]                   [ Base ]
      |                          |
   [ Child ]                  [ Derived 1 ]
                                 |
                              [ Derived 2 ]

3. Hierarchical            4. Multiple
        [ Base ]               [ Base A ]   [ Base B ]
        /   |  \                   \             /
   [ C1 ] [ C2 ] [ C3 ]                 [ Child ]
```

1. **Single Inheritance:** One derived class inherits from one base class.
2. **Multilevel Inheritance:** A derived class serves as the base class for another derived class.
3. **Hierarchical Inheritance:** Multiple derived classes inherit from a single shared base class.
4. **Multiple Inheritance:** One derived class inherits simultaneously from multiple base classes.

---

## 9. The `super()` Function & Method Overriding

### Method Overriding
When a derived class defines a method with the exact same name as a method in its base class, the derived class's version executes first when invoked on an instance of the child class.

### The Purpose of `super()`
`super()` returns a proxy object delegating method calls to a parent or sibling class in the inheritance chain:
- Eliminates hardcoded references to base classes.
- Ensures base constructors initialize inherited attributes properly.
- **Rule:** Never pass `self` as an argument when invoking methods through `super()`.

```python
class Person:
    def __init__(self, first_name: str, last_name: str):
        self.first_name = first_name
        self.last_name = last_name

    def display(self) -> None:
        print(f"Name: {self.first_name} {self.last_name}")

class Student(Person):
    def __init__(self, first_name: str, last_name: str, student_id: str):
        # Delegate initialization of shared fields to the parent class
        super().__init__(first_name, last_name)
        self.student_id = student_id

    # Method Overriding
    def display(self) -> None:
        # Call the parent version to reuse formatting logic
        super().display()
        print(f"Student ID: {self.student_id}")
```

---

## 10. Method Resolution Order (MRO) & Multiple Inheritance

In multiple inheritance, Python resolves attribute and method lookups using the **C3 Linearization Algorithm** (Method Resolution Order).

### MRO Lookup Principles:
1. Python searches the class itself first.
2. Base classes are searched from left to right as specified in the class definition.
3. Classes are visited only after all of their subclasses have been searched (depth/breadth hybrid).
4. No class is visited more than once.
5. All classes ultimately terminate at the root `object` class.

To inspect the resolution path, call `Class.mro()` or read `Class.__mro__`.

### Cooperative Multiple Inheritance Example
```python
class A:
    def __init__(self, fname, mname):
        print("Init in A")
        super().__init__(mname=mname)
        self.fname = fname

    def feature_a(self):
        print("Feature A active")

class B:
    def __init__(self, mname):
        print("Init in B")
        super().__init__()
        self.mname = mname

    def feature_b(self):
        print("Feature B active")

class C(A, B):
    def __init__(self, fname, mname, lname):
        print("Init in C")
        # Calls the next class in MRO (which is A)
        super().__init__(fname=fname, mname=mname)
        self.lname = lname

# Execution:
obj = C("Maria", "Clara", "Santos")
# Resolution order: C -> A -> B -> object
print(C.mro())
```

---

## 11. Abstract Classes & Interfaces (`abc` Module)

Python provides the `abc` (Abstract Base Classes) module to define formal interfaces and enforce architectural contracts across a codebase.

### Core Rules for Abstract Classes:
1. **Inherit from `ABC`:** The class must inherit from `abc.ABC`.
2. **Use `@abstractmethod`:** Any method that subclasses must implement must be decorated with `@abstractmethod`.
3. **Cannot Be Instantiated:** Python prevents the instantiation of any class containing unimplemented abstract methods.
4. **Polymorphic Contracts:** Abstract base classes ensure derived classes supply concrete implementations while allowing shared helper logic to remain in the base class.

```python
from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    """Abstract Base Class defining a common preprocessing interface."""

    def __init__(self, name: str):
        self.name = name

    # Concrete method inherited by all child classes
    def log(self, message: str) -> None:
        print(f"[{self.name}] {message}")

    # Abstract method: Must be overridden by any concrete subclass
    @abstractmethod
    def fit(self, data: list) -> None:
        pass

    @abstractmethod
    def transform(self, data: list) -> list:
        pass

class StandardScaleTransformer(BasePreprocessor):
    """Concrete implementation of BasePreprocessor."""

    def __init__(self):
        super().__init__("StandardScaler")
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data: list) -> None:
        self.mean = sum(data) / len(data)
        variance = sum((x - self.mean) ** 2 for x in data) / len(data)
        self.std = variance ** 0.5
        self.log(f"Fit complete. Mean: {self.mean:.2f}, Std: {self.std:.2f}")

    def transform(self, data: list) -> list:
        return [(x - self.mean) / self.std for x in data]

# Attempting to instantiate the base class directly raises an error:
# base = BasePreprocessor("test") -> TypeError: Can't instantiate abstract class

scaler = StandardScaleTransformer()
scaler.fit([10.0, 20.0, 30.0, 40.0])
scaled_data = scaler.transform([10.0, 20.0, 30.0, 40.0])
```

---

## 12. The Complete Refactored Case Study (`StudentRecord`)

The following complete, production-ready class consolidates all discussed OOP rules: state protection, access conventions, validation, defensive copying, class attributes, and property accessors.

```python
class StudentRecord:
    # 1. Class Attribute: Shared business rule applied to all students
    passing_score: float = 75.0

    def __init__(self, student_id: str, name: str, program: str):
        # 2. Internal instance attributes initialized with validation
        self._student_id = self._validate_text(student_id, "Student ID")
        self._name = self._validate_text(name, "Name")
        self._program = self._validate_text(program, "Program")
        self._scores: list[float] = []

    # Helper method for internal string validation
    @staticmethod
    def _validate_text(value: str, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string.")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{field_name} cannot be empty.")
        return cleaned

    # 3. Read-Only Properties
    @property
    def student_id(self) -> str:
        return self._student_id

    @property
    def program(self) -> str:
        return self._program

    # 4. Read-and-Write Property with Setter Validation
    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = self._validate_text(value, "Name")

    # 5. Defensive Copy for Collection Access
    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(self._scores)

    # 6. Validated Action Method (Safety Gate)
    def add_score(self, score: float) -> bool:
        if not isinstance(score, (int, float)):
            print("Validation Error: Score must be numeric.")
            return False
        if not (0 <= score <= 100):
            print("Validation Error: Score must be between 0 and 100.")
            return False
        
        self._scores.append(float(score))
        return True

    # 7. Computations Relying on Validated Internal State
    def calculate_average(self) -> float:
        if not self._scores:
            return 0.0
        return sum(self._scores) / len(self._scores)

    def get_status(self) -> str:
        # Uses the shared class attribute rule
        if self.calculate_average() >= StudentRecord.passing_score:
            return "Passed"
        return "Failed"

    def display_summary(self) -> None:
        print(f"Student ID : {self.student_id}")
        print(f"Name       : {self.name}")
        print(f"Program    : {self.program}")
        print(f"Scores     : {self.scores}")
        print(f"Average    : {self.calculate_average():.2f}")
        print(f"Status     : {self.get_status()}")
```

---

## 13. Master OOP Design Rules Checklist

- [ ] **Single Responsibility:** Each class models one cohesive concept and its immediate business logic.
- [ ] **No Naked Mutable State:** Internal attributes storing state use a leading underscore (`_attribute`).
- [ ] **Validate Early:** All inputs are inspected for expected types and allowed ranges before assignment.
- [ ] **Preserve Invariants:** Failed modifications must leave the existing object state completely uncorrupted.
- [ ] **Use Properties for Access:** Read/write attributes expose `@property` getters and setters instead of manual `get_x()` and `set_x()` boilerplate.
- [ ] **Defend Collections:** Properties exposing lists or dictionaries return immutable copies (`tuple(...)`, `.copy()`).
- [ ] **Separate Instance and Class State:** Put shared configuration and fixed rules in class attributes; put per-entity data in instance attributes.
- [ ] **Cooperative `super()` Calls:** Always use `super()` in child class methods without passing `self`.
- [ ] **Respect MRO:** When designing multiple inheritance trees, verify the lookup order using `ClassName.mro()`.
- [ ] **Contract Enforcement via ABCs:** Define interfaces using `abc.ABC` and `@abstractmethod` whenever multiple subclasses must implement a uniform API.