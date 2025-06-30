# Dependencies: huggingface_hub>=0.29.0
import os
import json
from docx import Document
from huggingface_hub import InferenceClient
from typing import Dict

def count_tokens(text: str) -> int:
    # Token counting is not strictly necessary for HF API, but you can use tiktoken if desired
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except ImportError:
        return len(text.split())

class ResolutionReviewerHF:
    def __init__(self, api_key: str, model: str, provider: str = "together"):
        self.client = InferenceClient(provider=provider, api_key=api_key)
        self.model = model
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        return """You are an expert reviewer of CUNY Board of Trustees resolutions. Your task is to analyze resolutions for compliance with the template and rules below.

For each violation found, you must:
1. Identify the specific rule or template requirement that was violated
2. Explain the error clearly
3. Provide the exact line where the error occurs. If a WHEREAS clause doesn't address a specific point that it is supposed to, then provide the number of that clause.
4. Suggest how to fix it

TEMPLATE STRUCTURE:
1. The resolution must follow this exact structure:
   - First line: "Board of Trustees of The City University of New York"
   - Next line: "RESOLUTION TO"
   - Next line: "Establish a [Degree Level Program] in [Subject] at [College Name]"
   - Next line: [Date of Board of Trustees' committee meeting]. Date must be in format "Month DD, YYYY"
   - WHEREAS clauses
   - 'NOW THEREFORE BE IT'
   - RESOLVED clause
   - EXPLANATION part

2. WHEREAS Clauses Requirements:
   - Each WHEREAS clause, must address specific points, individually and in order:
     1. Why the CUNY and market needs this program
     2. How the program curriculum and credits are designed to meet the needs of the CUNY and market
     3. Current student interest in the program
     4. Transferability of the program courses
     5. Benefits to students served and benefits to the CUNY and market
     6. Projected enrollment, retention, and graduation in the first 3 to 5 years
     7. Financial sustainability of the program given demand, section size, and current staffing, and projected revenue through enrollment to cover all operating costs in the first year
     8. Program investments during the initial growth period; include any space/equipment needs, renovations, staff and faculty hiring (list number of Part and Full Time Faculty), as indicated, and identify the sources of funds for each.
   - If a clause doesn't address a specific point that it is supposed to in a clear manner, then it is a violation and you must provide the number of that clause, and the specific point it doesn't address, along with the suggested fix.
   - All the specific points must be addressed in a clear and concise manner. If not, then it is a violation and you must provide the point that is missing, and suggest/write a WHEREAS clause that addresses it.
   - Each clause must end with "; and" except the last one which ends with a period
   - Each clause must be a single statement without any full-stops. Internal periods, like commas, are allowed.

3. RESOLVED Clause Requirements:
   - The RESOLVED clause must state the aim of the resolution succinctly
   - The RESOLVED clause must be a single statement with only 1 full-stop. Internal periods, like commas, are allowed.

4. EXPLANATION Requirements:
   - The EXPLANATION part must briefly summarize the purpose and benefits of the resolution.

5. Resolution Structure Rules:
   - Must include "NOW, THEREFORE, BE IT" after the last WHEREAS clause
   - Must have exactly one RESOLVED clause
   - Must include EXPLANATION section after RESOLVED clause

FORMATTING RULES:
1. Clause Formatting:
   - WHEREAS clauses must start with "WHEREAS,"
   - Only one full-stop allowed across all WHEREAS clause (except the last WHEREAS clause)
   - "; and" required at end of all WHEREAS clauses except the last
   - RESOLVED clause must start with "RESOLVED,"
   - RESOLVED clause must come after "NOW, THEREFORE, BE IT"
   - EXPLANATION part must start with "EXPLANATION:"
   - EXPLANATION must come after RESOLVED clause

Provide your analysis in the following JSON format:
{
    "template_violations": [
        {
            "rule": "string - the specific rule violated",
            "location": "string - where in the document" OR "#clause that violates a rule - the number of the clause violated - the specific point it doesn't address",
            "description": "string - clear explanation of the violation",
            "suggestion": "string - how to fix it"
        }
    ],
    "formatting_violations": [
        {
            "rule": "string - the specific rule violated",
            "location": "string - where in the document",
            "description": "string - clear explanation of the violation",
            "suggestion": "string - how to fix it"
        }
    ],
    "overall_assessment": "string - brief summary of the resolution's compliance"
}
"""

    def read_document(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def review_resolution(self, file_path: str) -> Dict:
        resolution_text = self.read_document(file_path)
        ex_original, ex_modified, changes = self._get_examples()
        user_prompt = self._build_user_prompt(ex_original, ex_modified, changes, resolution_text)
        # Compose chat messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # Call the Together/HF Inference API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"}
        )
        print("DEBUG: response.choices[0].message.content =", repr(response.choices[0].message.content))
        return json.loads(response.choices[0].message.content)

    def _get_examples(self):
        # (Copy the same example code as in the original resolution_reviewer.py)
        ex_original = """
        Board of Trustees of the City University of New York
        ... (same as before) ...
        """
        ex_modified = """
        Board of Trustees of the City University of New York
        ... (same as before) ...
        """
        changes = """
        ... (same as before) ...
        """
        return ex_original, ex_modified, changes

    def _build_user_prompt(self, ex_original, ex_modified, changes, resolution_text):
        return f"""
        Your task is to analyze a draft resolution for compliance with the template and rules. 
        An example of a draft resolution is given below, and the suggested corrections. 
        The incorrect resolution is given below, followed by the updated version of that resolution. 
        The changes that were made and the rules violated are given after the updated version.
        <EXAMPLE START>
        ORIGINAL VERSION:-
        {ex_original}
        ----------------
        MODIFIED VERSION:-
        {ex_modified}
        ----------------
        ORIGINAL VERSION:-
        {changes}
        ----------------
        <EXAMPLE END>
        
        Based on the template provided, alonside the rules that may or may not be violated, please analyze the following resolution for compliance with the template and rules:
        {resolution_text}

        Provide a detailed analysis identifying any violations of the template or rules.""" 