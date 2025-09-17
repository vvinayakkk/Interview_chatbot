import streamlit as st
import pandas as pd
import requests
import io
import re
import os
import tempfile
import PyPDF2
from groq import Groq
import urllib.parse
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class InterviewQuestionGenerator:
    def __init__(self):
        # Initialize Groq client with error handling
        try:
            api_key = ""
            
            if api_key:
                self.groq_client = Groq(api_key=api_key)
            else:
                st.error("No API key provided. LLM generation will be disabled.")
                self.groq_client = None
        
        except Exception as e:
            st.error(f"Error initializing Groq client: {e}")
            self.groq_client = None
        
        # DataFrame to store questions
        self.df = pd.DataFrame(columns=[
            'Topic', 'Sub-Topic', 'Difficulty', 
            'Passage', 'Question', 'Answer',
            'Complexity_Score', 'Bloom_Level'
        ])
        
        # Define Bloom's Taxonomy levels for classification
        self.bloom_taxonomy = {
            'Remember': ['define', 'recall', 'list', 'identify', 'name', 'describe', 'state'],
            'Understand': ['explain', 'interpret', 'summarize', 'classify', 'compare', 'discuss'],
            'Apply': ['implement', 'use', 'execute', 'solve', 'demonstrate', 'apply'],
            'Analyze': ['analyze', 'differentiate', 'distinguish', 'examine', 'investigate'],
            'Evaluate': ['evaluate', 'critique', 'judge', 'assess', 'recommend'],
            'Create': ['design', 'develop', 'create', 'construct', 'formulate', 'propose']
        }

    def parse_text(self, text):
        """
        Improved text parsing method
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate to a reasonable length
        return text[:2000]

    def extract_text_from_source(self, source, source_type='url'):
        """
        Extract and parse text from various sources
        """
        try:
            if source_type == 'url':
                # Validate URL
                if not urllib.parse.urlparse(source).scheme:
                    st.error("Invalid URL. Please provide a complete URL.")
                    return None

                # Fetch content with robust headers
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept-Language': 'en-US,en;q=0.9'
                }
                response = requests.get(source, headers=headers)
                
                # Parse Web Page
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove unnecessary elements
                for element in soup(['script', 'style', 'head', 'header', 'footer', 'nav']):
                    element.decompose()
                
                # Extract main content
                text = soup.get_text(separator=' ', strip=True)
                return self.parse_text(text)
            
            elif source_type == 'pdf':
                # Process the PDF file
                pdf_text = ""
                try:
                    pdf_reader = PyPDF2.PdfReader(source)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        pdf_text += page.extract_text() + " "
                    
                    return self.parse_text(pdf_text)
                except Exception as pdf_error:
                    st.error(f"Error reading PDF: {pdf_error}")
                    return None
            
            return None
        
        except Exception as e:
            st.error(f"Text extraction error: {e}")
            return None

    def generate_questions_with_chain_of_thought(self, topic, sub_topics, passage, difficulty_counts, generation_method="standard"):
        """
        Generate questions using different methods based on selection
        """
        if self.groq_client is None:
            st.warning("LLM generation is disabled.")
            return []

        generated_questions = []
        
        for sub_topic in sub_topics:
            for difficulty, count in difficulty_counts.items():
                if count <= 0:
                    continue
                    
                # Choose generation method
                if generation_method == "bloom":
                    questions = self._generate_with_blooms_taxonomy(topic, sub_topic, difficulty, count, passage)
                elif generation_method == "contextual":
                    questions = self._generate_contextual_questions(topic, sub_topic, difficulty, count, passage)
                else:  # standard
                    questions = self._generate_standard_questions(topic, sub_topic, difficulty, count, passage)
                
                for question_data in questions:
                    # Add to DataFrame with additional metrics
                    complexity_score = self._calculate_complexity_score(question_data["question"], question_data["answer"])
                    bloom_level = self._classify_bloom_level(question_data["question"])
                    
                    new_row = {
                        'Topic': topic,
                        'Sub-Topic': sub_topic,
                        'Difficulty': difficulty,
                        'Passage': passage[:500] if passage else 'Theoretical Question',
                        'Question': question_data["question"],
                        'Answer': question_data["answer"],
                        'Complexity_Score': complexity_score,
                        'Bloom_Level': bloom_level
                    }
                    self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
                    generated_questions.append(question_data["question"])
        
        return generated_questions

    def _generate_standard_questions(self, topic, sub_topic, difficulty, count, passage):
        """
        Standard question generation approach
        """
        cot_prompt = f"""
        Task: Generate high-quality technical interview questions about {topic} in {sub_topic}

        Difficulty Level Guidelines:
        {'Easy' if difficulty == 'Easy' else 'Medium' if difficulty == 'Medium' else 'Hard'} Difficulty Specifications:
        {{
            'Easy': '''
            - Focus on fundamental concepts
            - Require basic recall and simple explanations
            - Suitable for entry-level understanding
            - Test foundational knowledge
            - Use straightforward language
            - Avoid complex scenarios
            ''',
            'Medium': '''
            - Explore deeper conceptual relationships
            - Require moderate analytical thinking
            - Test understanding beyond basic recall
            - Involve some problem-solving
            - Introduce moderate complexity
            - Require explanation of underlying principles
            ''',
            'Hard': '''
            - Challenge with complex scenarios
            - Require advanced critical thinking
            - Test deep understanding and application
            - Involve multi-layered problem-solving
            - Explore edge cases and theoretical implications
            '''
        }}['{difficulty}']

        {'Reference passage for context: ' + passage[:1000] if passage else 'Generate theoretical questions without reference passage.'}

        Additional Constraints:
        - Generate {count} unique questions
        - Ensure no repetition
        - Align with {difficulty.lower()} difficulty guidelines
        - Focus on {sub_topic} within {topic}

        Output Format:
        Provide each question in JSON format:
        [
            {{"question": "Question text here", "answer": ""}},
            ...
        ]
        """

        try:
            # Generate questions
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"You are an expert technical interview question generator for {difficulty} level questions."},
                    {"role": "user", "content": cot_prompt}
                ],
                model="llama3-70b-8192"
            )
            
            response_text = chat_completion.choices[0].message.content.strip()
            
            # Extract questions and answers
            try:
                # Try to parse as JSON
                import json
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    questions_data = json.loads(json_str)
                else:
                    # Fallback to regex
                    questions_data = self._extract_questions_with_regex(response_text)
            except:
                # Fallback to regex
                questions_data = self._extract_questions_with_regex(response_text)
            
            # Process each question to add answers if missing
            result = []
            for q_data in questions_data[:count]:
                question = q_data.get("question", "").strip()
                answer = q_data.get("answer", "")
                
                if not question:
                    continue
                    
                if not answer:
                    answer = self.generate_detailed_answer(question, topic, sub_topic)
                
                result.append({"question": question, "answer": answer})
            
            return result
                
        except Exception as e:
            st.error(f"Question generation error: {e}")
            return []

    def _generate_with_blooms_taxonomy(self, topic, sub_topic, difficulty, count, passage):
        """
        Generate questions using Bloom's Taxonomy approach
        """
        # Map difficulty to Bloom's levels
        bloom_levels = {
            'Easy': ['Remember', 'Understand'],
            'Medium': ['Apply', 'Analyze'],
            'Hard': ['Evaluate', 'Create']
        }
        
        selected_levels = bloom_levels.get(difficulty, ['Understand', 'Apply'])
        
        bloom_prompt = f"""
        Task: Generate {count} technical interview questions about {topic} in {sub_topic} using Bloom's Taxonomy levels: {', '.join(selected_levels)}.
        
        {'Reference passage for context: ' + passage[:1000] if passage else 'Generate theoretical questions without reference passage.'}
        
        For each Bloom's level, create questions that require the following cognitive skills:
        
        - Remember: Recall facts and basic concepts
        - Understand: Explain ideas or concepts
        - Apply: Use information in new situations
        - Analyze: Draw connections among ideas
        - Evaluate: Justify a stand or decision
        - Create: Produce new or original work
        
        Output Format:
        Provide each question in JSON format:
        [
            {{"question": "Question text here", "answer": "", "bloom_level": "level_name"}},
            ...
        ]
        """
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert technical interview question generator specializing in Bloom's Taxonomy."},
                    {"role": "user", "content": bloom_prompt}
                ],
                model="llama3-70b-8192"
            )
            
            response_text = chat_completion.choices[0].message.content.strip()
            
            # Extract JSON data
            try:
                import json
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    questions_data = json.loads(json_str)
                else:
                    questions_data = self._extract_questions_with_regex(response_text)
            except:
                questions_data = self._extract_questions_with_regex(response_text)
            
            # Process questions
            result = []
            for q_data in questions_data[:count]:
                question = q_data.get("question", "").strip()
                answer = q_data.get("answer", "")
                
                if not question:
                    continue
                    
                if not answer:
                    answer = self.generate_detailed_answer(question, topic, sub_topic)
                
                result.append({"question": question, "answer": answer})
            
            return result
                
        except Exception as e:
            st.error(f"Bloom's Taxonomy question generation error: {e}")
            return []

    def _generate_contextual_questions(self, topic, sub_topic, difficulty, count, passage):
        """
        Generate questions tightly coupled with the provided passage
        """
        if not passage:
            # Fall back to standard generation if no passage
            return self._generate_standard_questions(topic, sub_topic, difficulty, count, passage)
        
        contextual_prompt = f"""
        Task: Generate {count} technical interview questions about {topic} in {sub_topic} based specifically on the following passage.
        
        Passage:
        {passage[:1500]}
        
        Difficulty: {difficulty}
        
        Guidelines:
        - Questions should directly reference concepts, examples, or information in the passage
        - For {difficulty} difficulty, focus on {'basic recall and understanding' if difficulty == 'Easy' else 'application and analysis' if difficulty == 'Medium' else 'evaluation and synthesis'}
        - Make questions that test both factual knowledge and conceptual understanding
        - Ensure questions require the candidate to have comprehended the passage
        
        Output Format:
        Provide each question in JSON format:
        [
            {{"question": "Question text here", "answer": ""}},
            ...
        ]
        """
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert technical interview question generator specializing in contextual comprehension."},
                    {"role": "user", "content": contextual_prompt}
                ],
                model="llama3-70b-8192"
            )
            
            response_text = chat_completion.choices[0].message.content.strip()
            
            # Extract JSON data
            try:
                import json
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    questions_data = json.loads(json_str)
                else:
                    questions_data = self._extract_questions_with_regex(response_text)
            except:
                questions_data = self._extract_questions_with_regex(response_text)
            
            # Process questions
            result = []
            for q_data in questions_data[:count]:
                question = q_data.get("question", "").strip()
                answer = q_data.get("answer", "")
                
                if not question:
                    continue
                    
                if not answer:
                    answer = self.generate_detailed_answer(question, topic, sub_topic, passage)
                
                result.append({"question": question, "answer": answer})
            
            return result
                
        except Exception as e:
            st.error(f"Contextual question generation error: {e}")
            return []
    
    def _extract_questions_with_regex(self, text):
        """
        Fallback method to extract questions when JSON parsing fails
        """
        questions = []
        
        # Find numbered questions
        matches = re.findall(r'(?:\d+\.|\*)\s*(.*?)(?=\d+\.|\*|$)', text, re.DOTALL)
        
        for match in matches:
            match = match.strip()
            if match:
                questions.append({"question": match, "answer": ""})
        
        return questions

    def generate_detailed_answer(self, question, topic, sub_topic, context=None):
        """
        Generate comprehensive answer using LLM
        """
        if self.groq_client is None:
            return "Answer generation disabled."

        try:
            context_part = f"\nContext:\n{context[:1000]}" if context else ""
            
            answer_prompt = f"""
            Provide a comprehensive technical answer to:
            Topic: {topic}
            Sub-Topic: {sub_topic}
            Question: {question}{context_part}

            Answer Guidelines:
            - Provide in-depth explanation
            - Include relevant technical details
            - Discuss underlying principles
            - Offer practical context
            - Include code examples if appropriate
            """

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a technical interview answer generator providing clear, accurate, and detailed responses."},
                    {"role": "user", "content": answer_prompt}
                ],
                model="llama3-70b-8192"
            )
            
            return chat_completion.choices[0].message.content.strip()
        
        except Exception as e:
            st.error(f"Answer generation error: {e}")
            return "Unable to generate answer."
    
    def _calculate_complexity_score(self, question, answer):
        """
        Calculate complexity score based on various metrics
        """
        # Initialize score
        score = 5.0  # Default medium score
        
        # Question length factor (longer questions often more complex)
        q_length = len(question.split())
        if q_length > 25:
            score += 1.5
        elif q_length < 12:
            score -= 1.0
            
        # Answer length factor
        a_length = len(answer.split())
        if a_length > 200:
            score += 1.5
        elif a_length < 100:
            score -= 1.0
            
        # Keyword complexity analysis
        complex_indicators = ['analyze', 'evaluate', 'synthesize', 'design', 'compare', 
                             'optimize', 'architecture', 'framework', 'implementation',
                             'advanced', 'complex', 'challenging']
        
        for word in complex_indicators:
            if word in question.lower() or word in answer.lower():
                score += 0.5
                
        # Technical term density
        tech_terms = ['algorithm', 'data structure', 'complexity', 'optimization', 
                      'pattern', 'framework', 'architecture', 'system', 'protocol',
                      'interface', 'implementation', 'design pattern']
        
        term_count = sum(1 for term in tech_terms if term in question.lower() or term in answer.lower())
        score += term_count * 0.3
        
        # Cap the score between 1-10
        return round(min(max(score, 1), 10), 1)
    
    def _classify_bloom_level(self, question):
        """
        Classify question according to Bloom's Taxonomy
        """
        question_lower = question.lower()
        
        # Check for each level, starting from highest
        for level, keywords in reversed(list(self.bloom_taxonomy.items())):
            for keyword in keywords:
                if keyword in question_lower:
                    return level
        
        # Default to "Understand" if no match found
        return "Understand"
    
    def sort_by_difficulty(self, method="complexity"):
        """
        Sort questions by difficulty using different methods
        """
        if method == "complexity":
            # Sort by computed complexity score
            self.df = self.df.sort_values(by=['Complexity_Score'], ascending=True)
        elif method == "bloom":
            # Sort by Bloom's Taxonomy level
            bloom_order = {
                'Remember': 1,
                'Understand': 2,
                'Apply': 3,#medium
                'Analyze': 4,#hard
                'Evaluate': 5,
                'Create': 6
            }
            self.df['BloomOrder'] = self.df['Bloom_Level'].map(bloom_order)
            self.df = self.df.sort_values(by=['BloomOrder'], ascending=True)
            self.df = self.df.drop('BloomOrder', axis=1)
        else:
            # Default sort by assigned difficulty
            difficulty_order = {'Easy': 1, 'Medium': 2, 'Hard': 3}
            self.df['DifficultyOrder'] = self.df['Difficulty'].map(difficulty_order)
            self.df = self.df.sort_values(by=['DifficultyOrder'], ascending=True)
            self.df = self.df.drop('DifficultyOrder', axis=1)
        
        return self.df
    
    def analyze_question_distribution(self):
        """
        Analyze the distribution of questions
        """
        summary = {
            'difficulty_counts': self.df['Difficulty'].value_counts().to_dict(),
            'bloom_level_counts': self.df['Bloom_Level'].value_counts().to_dict(),
            'avg_complexity': self.df['Complexity_Score'].mean(),
            'topic_distribution': self.df['Sub-Topic'].value_counts().to_dict()
        }
        
        return summary

def main():
    st.title("Advanced Technical Interview Question Generator")
    
    # Initialize the generator
    if 'generator' not in st.session_state:
        st.session_state.generator = InterviewQuestionGenerator()
    
    # Option to upload existing CSV
    st.sidebar.header("Data Import Options")
    uploaded_csv = st.sidebar.file_uploader("Upload Existing Questions CSV", type=['csv'])
    
    if uploaded_csv is not None:
        try:
            # Read the uploaded CSV
            existing_df = pd.read_csv(uploaded_csv)
            
            # Handle potential missing columns
            for col in ['Complexity_Score', 'Bloom_Level']:
                if col not in existing_df.columns:
                    existing_df[col] = None
            
            # Merge with existing dataframe
            st.session_state.generator.df = pd.concat([
                st.session_state.generator.df, 
                existing_df
            ], ignore_index=True)
            
            st.success(f"Uploaded {len(existing_df)} existing questions.")
        
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    
    # Input source selection
    st.sidebar.header("Content Source")
    source_type = st.sidebar.radio("Select Source Type", ["URL", "PDF", "Theoretical"])
    
    source = None
    source_file_content = None
    
    if source_type == "URL":
        source = st.sidebar.text_input("Enter URL")
        if source:
            with st.spinner("Extracting content from URL..."):
                source_file_content = st.session_state.generator.extract_text_from_source(source, 'url')
    
    elif source_type == "PDF":
        uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=['pdf'])
        
        if uploaded_pdf is not None:
            with st.spinner("Extracting content from PDF..."):
                # Save PDF to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(uploaded_pdf.getvalue())
                    temp_path = temp_file.name
                
                # Extract text
                with open(temp_path, 'rb') as pdf_file:
                    source_file_content = st.session_state.generator.extract_text_from_source(pdf_file, 'pdf')
                
                # Clean up
                os.unlink(temp_path)
    
    # Show content preview if available
    if source_file_content:
        with st.expander("Content Preview"):
            st.write(source_file_content[:500] + "...")
    
    # Topic and sub-topics inputs
    st.header("Question Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        topic = st.text_input("Enter Topic (e.g., Machine Learning, Java)")
    
    with col2:
        sub_topics_input = st.text_input("Enter Sub-Topics (comma-separated)")
    
    # Generation method selection
    generation_method = st.radio(
        "Generation Method", 
        ["Standard", "Bloom's Taxonomy", "Contextual"],
        help="Standard: Basic question generation. Bloom's: Uses educational taxonomy for structured questions. Contextual: Generates questions tightly coupled with provided content."
    )
    
    # Map selection to internal method name
    method_map = {
        "Standard": "standard",
        "Bloom's Taxonomy": "bloom",
        "Contextual": "contextual"
    }
    
    # Difficulty-based question generation
    st.subheader("Question Difficulty Distribution")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        easy_questions = st.number_input("Easy Questions", min_value=0, max_value=10, value=2)
    
    with col2:
        medium_questions = st.number_input("Medium Questions", min_value=0, max_value=10, value=3)
    
    with col3:
        hard_questions = st.number_input("Hard Questions", min_value=0, max_value=10, value=2)
    
    # Sorting options
    st.subheader("Advanced Options")
    col1, col2 = st.columns(2)
    
    with col1:
        sorting_method = st.selectbox(
            "Sort Questions By", 
            ["Assigned Difficulty", "Complexity Score", "Bloom's Taxonomy"],
            help="Choose how to sort the generated questions."
        )
    
    sorting_map = {
        "Assigned Difficulty": "default",
        "Complexity Score": "complexity",
        "Bloom's Taxonomy": "bloom"
    }
    
    # Generate button
    if st.button("Generate Questions"):
        if not topic or not sub_topics_input:
            st.error("Please enter a topic and sub-topics.")
            return
        
        # Split sub-topics
        sub_topic_list = [st.strip() for st in sub_topics_input.split(',') if st.strip()]
        
        # Prepare difficulty counts
        difficulty_counts = {
            'Easy': easy_questions,
            'Medium': medium_questions,
            'Hard': hard_questions
        }
        
        # Generate questions with progress indication
        with st.spinner("Generating questions... This may take a moment."):
            generated_questions = st.session_state.generator.generate_questions_with_chain_of_thought(
                topic, 
                sub_topic_list, 
                source_file_content,
                difficulty_counts,
                generation_method=method_map[generation_method]
            )
            
            # Sort questions
            sorted_df = st.session_state.generator.sort_by_difficulty(method=sorting_map[sorting_method])
        
        # Display analysis
        st.subheader("Question Analysis")
        analysis = st.session_state.generator.analyze_question_distribution()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Difficulty Distribution**")
            for diff, count in analysis['difficulty_counts'].items():
                st.write(f"- {diff}: {count}")
        
        with col2:
            st.write("**Bloom's Taxonomy Levels**")
            for level, count in analysis['bloom_level_counts'].items():
                st.write(f"- {level}: {count}")
        
        st.write(f"**Average complexity score:** {analysis['avg_complexity']:.2f}/10")
        
        # Display dataset
        st.subheader("Questions Dataset")
        st.dataframe(sorted_df)
        
        # Download option
        st.download_button(
            label="Download Questions Dataset",
            data=sorted_df.to_csv(index=False),
            file_name=f"{topic}_interview_questions.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()