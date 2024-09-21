class DifficultyAdaptation:
    def __init__(self, difficulty_curve, student_performance_tracker):
        self.difficulty_curve = difficulty_curve
        self.performance_tracker = student_performance_tracker  
    
    def calculate_new_difficulty(self, current_difficulty, evaluation_scores):
        
        performance_score = (
            0.3 * evaluation_scores['correctness'] +
            0.2 * evaluation_scores['depth'] +
            0.2 * evaluation_scores['concept_coverage'] +
            0.2 * evaluation_scores['confidence'] +
            0.1 * evaluation_scores['examples']
        )
   
        difficulty_change = self.difficulty_curve.adjust_difficulty(performance_score)
        new_difficulty = min(max(current_difficulty + difficulty_change, 0), 3)  

        return new_difficulty

    def track_performance(self, student_id, question_difficulty, performance_score):
      
        self.performance_tracker.log(student_id, question_difficulty, performance_score)
