import random

question_bank = [
    {"text": "What is Java?", "difficulty": 1},
    {"text": "Explain the concept of OOP in Java.", "difficulty": 1},
    {"text": "What are the different types of inheritance in Java?", "difficulty": 2},
    {"text": "Explain the difference between method overloading and overriding.", "difficulty": 2},
    {"text": "What is a constructor in Java?", "difficulty": 2},
    {"text": "What are access specifiers in Java?", "difficulty": 2},
    {"text": "What is the difference between an abstract class and an interface?", "difficulty": 3},
    {"text": "Explain the concept of exception handling in Java.", "difficulty": 3},
    {"text": "What is the significance of 'final' keyword in Java?", "difficulty": 3},
    {"text": "What is multithreading in Java and how is it implemented?", "difficulty": 4},
    {"text": "Explain garbage collection in Java.", "difficulty": 4},
    {"text": "What are Java streams and how do they work?", "difficulty": 4},
    {"text": "What is JVM, JRE, and JDK?", "difficulty": 1},
    {"text": "Explain the use of the 'super' keyword in Java.", "difficulty": 2},
    {"text": "Describe the lifecycle of a Java thread.", "difficulty": 4},
    {"text": "What are lambda expressions in Java?", "difficulty": 4},
    {"text": "Explain how Java achieves platform independence.", "difficulty": 3},
    {"text": "What is the significance of the 'this' keyword in Java?", "difficulty": 2},
    {"text": "What are functional interfaces in Java?", "difficulty": 3},
    {"text": "Explain the role of the Java Collections Framework.", "difficulty": 3},
    {"text": "What are the main differences between HashMap and TreeMap?", "difficulty": 4},
    {"text": "What is serialization in Java?", "difficulty": 4},
    {"text": "What is the difference between '=='' and '.equals()' in Java?", "difficulty": 2},
    {"text": "Explain how memory management works in Java.", "difficulty": 3},
    {"text": "What are annotations in Java?", "difficulty": 5},
    {"text": "What is the 'volatile' keyword in Java?", "difficulty": 5},
    {"text": "Explain the Producer-Consumer problem and its implementation in Java.", "difficulty": 5},
    {"text": "What are design patterns in Java? Name and explain one.", "difficulty": 4},
    {"text": "What is reflection in Java and how is it used?", "difficulty": 5}
]


def get_next_question(current_difficulty):

    possible_questions = [q for q in question_bank if q["difficulty"] == current_difficulty]
    

    if not possible_questions:
        possible_questions = [q for q in question_bank if abs(q["difficulty"] - current_difficulty) <= 1]

    return random.choice(possible_questions)

