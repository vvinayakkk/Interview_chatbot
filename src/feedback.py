import json
from langchain_integration import evaluate_answer, similarity_search


def load_student_performance(file_path="data/student_performance.json"):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_student_performance(performance_data, file_path="data/student_performance.json"):
    with open(file_path, 'w') as f:
        json.dump(performance_data, f, indent=4)

def generate_feedback(student_id, student_answer, question, api_key):

    evaluation = evaluate_answer([question], student_answer, api_key)
    

    similar_questions = similarity_search([question], student_answer)
    

    accuracy = calculate_accuracy(student_answer, question['answer'])
    coverage = calculate_coverage(evaluation, question['text'])
    difficulty_adjustment = calculate_difficulty_adjustment(accuracy, coverage, question['difficulty'])

    feedback = {
        "evaluation": evaluation,
        "accuracy": accuracy,
        "coverage": coverage,
        "difficulty_adjustment": difficulty_adjustment,
        "similar_questions": [q['text'] for q in similar_questions],
        "improvement_suggestions": generate_improvement_suggestions(accuracy, coverage),
    }

    update_student_performance(student_id, feedback)

    return feedback

def calculate_accuracy(student_answer, correct_answer):

    correct_tokens = set(correct_answer.lower().split())
    student_tokens = set(student_answer.lower().split())
    common_tokens = correct_tokens.intersection(student_tokens)
    accuracy = len(common_tokens) / len(correct_tokens) * 100
    return round(accuracy, 2)

def calculate_coverage(evaluation, question_text):
    
    return evaluation.get('coverage_score', 0) 

def calculate_difficulty_adjustment(accuracy, coverage, current_difficulty):
    if accuracy >= 80 and coverage >= 75:
        return current_difficulty + 1  
    elif accuracy < 50 or coverage < 50:
        return max(1, current_difficulty - 1)  
    else:
        return current_difficulty 
def generate_improvement_suggestions(accuracy, coverage):
    suggestions = []
    if accuracy < 80:
        suggestions.append("Focus on providing more accurate details in your answers.")
    if coverage < 75:
        suggestions.append("Try to cover more aspects of the question in your response.")
    return suggestions if suggestions else ["Great job! Keep up the good work."]


def update_student_performance(student_id, feedback):
    performance_data = load_student_performance()
    
    if student_id not in performance_data:
        performance_data[student_id] = []

    
    performance_data[student_id].append(feedback)

    
    save_student_performance(performance_data)


def get_performance_summary(student_id):
    performance_data = load_student_performance()
    if student_id not in performance_data:
        return "No performance data found for this student."
    
    
    student_history = performance_data[student_id]
    total_sessions = len(student_history)
    avg_accuracy = sum([session["accuracy"] for session in student_history]) / total_sessions
    avg_coverage = sum([session["coverage"] for session in student_history]) / total_sessions
    avg_difficulty = sum([session["difficulty_adjustment"] for session in student_history]) / total_sessions

    return {
        "total_sessions": total_sessions,
        "average_accuracy": round(avg_accuracy, 2),
        "average_coverage": round(avg_coverage, 2),
        "average_difficulty": round(avg_difficulty, 2),
        "overall_feedback": "You are doing well!" if avg_accuracy > 75 and avg_coverage > 70 else "Focus on improvement!"
    }
