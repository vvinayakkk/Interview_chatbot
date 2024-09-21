from difficulty_adaptation import DifficultyAdaptation
from langchains_integration import rag_processing
from feedback import generate_feedback
from question_bank import get_next_question
import json
import os
import os
from dotenv import load_dotenv
load_dotenv()

STUDENT_FILE = "./data/student_performance.json"

def load_student_data():
    if os.path.exists(STUDENT_FILE):
        with open(STUDENT_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_student_data(student_data):
    with open(STUDENT_FILE, 'w') as file:
        json.dump(student_data, file, indent=4)

def get_student_id(student_data):
    student_id = input("Enter your student ID (type 'new' if this is your first time): ")

    if student_id == "new":
        new_id = f"student_{len(student_data) + 1}"
        print(f"Welcome! Your new ID is {new_id}. Please remember it for future sessions.")
        student_data[new_id] = {"performance": [], "current_difficulty": 1}
        return new_id
    elif student_id in student_data:
        print(f"Welcome back, {student_id}!")
        return student_id
    else:
        print("Student ID not found. Please try again.")
        return get_student_id(student_data)

def start_interview(api_key, difficulty_curve, performance_tracker):
    student_data = load_student_data()
    student_id = get_student_id(student_data)

    current_difficulty = student_data[student_id]["current_difficulty"]

    difficulty_adapter = DifficultyAdaptation(difficulty_curve=difficulty_curve, student_performance_tracker=performance_tracker)

    while True:
        question = get_next_question(current_difficulty)
        print(f"Question: {question['text']}")

        answer = input("Your answer: ")

        try:
            evaluation = rag_processing([question], api_key=api_key, input_text=answer)
            print(f"Evaluation: {evaluation}")
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            continue

        current_difficulty = difficulty_adapter.calculate_new_difficulty(current_difficulty, evaluation)
        student_data[student_id]["current_difficulty"] = current_difficulty

        student_data[student_id]["performance"].append({
            "question": question["text"],
            "answer": answer,
            "evaluation": evaluation,
            "difficulty": current_difficulty
        })

        feedback = generate_feedback(evaluation)
        print(f"Feedback: {feedback}")

        continue_interview = input("Do you want to answer another question? (yes/no): ").lower()
        if continue_interview != "yes":
            break

    save_student_data(student_data)
    print("Thank you for completing the interview!")

if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")  
    difficulty_curve = None  
    performance_tracker = None  
    start_interview(api_key=api_key, difficulty_curve=difficulty_curve, performance_tracker=performance_tracker)
