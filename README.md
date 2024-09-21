# Interview Chatbot

## Overview

The Interview Chatbot is a comprehensive application designed to aid interview preparation by evaluating responses and adapting question difficulty based on student performance. Utilizing state-of-the-art AI models from Google and Hugging Face, this application offers a structured and interactive learning experience.

## Features

- **Dynamic Question Generation**: Questions are generated based on the student's current difficulty level.
- **AI-Powered Answer Evaluation**: Evaluates responses on correctness, depth, and conceptual coverage.
- **Adaptive Learning**: Adjusts question difficulty based on real-time performance data.
- **Performance Tracking**: Records and updates student performance for personalized learning.

## Detailed Workflow

1. **Environment Setup**:
   - Create a `.env` file with your API keys:
     ```
     GEMINI_API_KEY=your_gemini_api_key
     HUGGING_FACE_API_KEY=your_hugging_face_api_key
     ```
   - Install dependencies using `pip install -r requirements.txt`.

2. **Student Interaction**:
   - On starting the application, students are prompted to enter their ID. If it's their first time, a new ID is generated and stored.

3. **Question Selection**:
   - The application retrieves the current difficulty level for the student.
   - A question is fetched based on this difficulty level, which is determined by previous performance.

4. **Answer Submission**:
   - The student submits their answer to the displayed question.

5. **Embedding Generation**:
   - The submitted answer and expected answer are converted into embeddings using the `GoogleGenerativeAIEmbeddings` model. This involves:
     - Tokenization: The text is broken down into tokens suitable for model input.
     - Vector Representation: The model processes the tokens to generate dense vector representations (embeddings) capturing semantic meaning.

6. **Answer Evaluation**:
   - The evaluation process consists of multiple assessments:
     - **Correctness**: Computes cosine similarity between the candidate answer embedding and the expected answer embedding. A higher similarity indicates a more correct response.
     - **Depth Assessment**: The system generates an in-depth explanation of the expected answer using the `HuggingFaceHub` model and compares it with the candidate answer.
     - **Conceptual Coverage**: It assesses whether the candidate answer covers key concepts related to the expected answer, leveraging the LLM to identify essential topics.
     - **Example Usage**: The application checks if the candidate answer includes relevant examples through an LLM-generated query.

7. **Feedback Generation**:
   - Based on the evaluations, feedback is generated to guide the student. This feedback includes scores for correctness, depth, and coverage, along with suggestions for improvement.

8. **Performance Tracking**:
   - The student's performance data is updated, including the evaluation results and the difficulty level for the next question. This data is stored in a JSON file for persistence.

9. **Difficulty Adjustment**:
   - The system uses the `DifficultyAdaptation` class to determine whether to increase or decrease the difficulty level for the next question based on the student's performance.

10. **Session Continuation**:
    - After feedback is provided, the student is prompted to continue or end the interview. If they choose to continue, the process repeats from question selection.

11. **Session Termination**:
    - Upon completion, the updated performance data is saved, ensuring that student progress is retained for future sessions.
   
![image](https://github.com/user-attachments/assets/a702a2ad-9cf8-404a-a429-d97ce12ae5c1)


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/interview-chatbot.git
   cd interview-chatbot

2. Create a .env file:
   ```bash
    GEMINI_API_KEY=your_gemini_api_key
    HUGGING_FACE_API_KEY=your_hugging_face_api_key

3. Install dependencies:
   ```bash
    pip install -r requirements.txt

4. Run the code after going to src
   ```bash
   python main.py
   

