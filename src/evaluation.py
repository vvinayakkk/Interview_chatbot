from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
import numpy as np

class AnswerEvaluator:
    def __init__(self):
        self.llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.7})

    def evaluate_answer(self, question, candidate_answer, expected_answer):
        evaluation = {
            'correctness': self.evaluate_correctness(candidate_answer, expected_answer),
            'depth': self.evaluate_depth(candidate_answer, expected_answer),
            'examples': self.check_example_usage(candidate_answer),
            'concept_coverage': self.evaluate_coverage(candidate_answer, expected_answer),
            'confidence': self.llm_confidence(question, candidate_answer)
        }
        return evaluation

    def evaluate_correctness(self, candidate_answer, expected_answer):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        candidate_embed = embeddings.embed_text(candidate_answer)
        expected_embed = embeddings.embed_text(expected_answer)
        similarity = np.dot(candidate_embed, expected_embed) / (np.linalg.norm(candidate_embed) * np.linalg.norm(expected_embed))
        return similarity

    def evaluate_depth(self, candidate_answer, expected_answer):
        generated_depth = self.llm(f"Explain this in detail: {expected_answer}")
        return self.evaluate_correctness(candidate_answer, generated_depth)

    def check_example_usage(self, candidate_answer):
        response = self.llm(f"Does the following answer contain an example? {candidate_answer}")
        return 'yes' in response.lower()

    def evaluate_coverage(self, candidate_answer, expected_answer):
        generated_coverage = self.llm(f"What key concepts should be covered for: {expected_answer}")
        return self.evaluate_correctness(candidate_answer, generated_coverage)

    def llm_confidence(self, question, candidate_answer):
        response = self.llm(f"How confident are you in evaluating this answer? Question: {question}, Answer: {candidate_answer}")
        try:
            confidence = float(response.split('confidence')[1].split('%')[0]) / 100.0
        except:
            confidence = 0.5
        return confidence
