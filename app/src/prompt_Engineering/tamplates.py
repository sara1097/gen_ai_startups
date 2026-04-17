from typing import List, Dict

# INTENTS DETECTION TEMPLATE
INTENTS_DETECTION_TEMPLATE = """You are a multilingual intent classifier. Detect intents regardless of the language used.
Always return all text fields in English only.

User input: "{user_message}"

Analyze this user input and detect ALL applicable intents.

CRITICAL DISTINCTIONS:
- problem_solving: User describes a SPECIFIC problem and wants a startup solution
  Examples: "I want to solve expensive education", "transportation in Cairo is bad"
  
- random_solution: User asks for ANY startup idea WITHOUT describing a problem
  Examples: "Give me a startup idea", "What's a good business"
  
- follow_up: User continues discussion on a PREVIOUS idea
  Examples: "Tell me more about that idea", "How can we improve it?"
  
- alternative_idea: User wants a DIFFERENT solution for the SAME problem
  Examples: "Another approach to education", "Different solution"
  
- details: User asks for more information/details
  Examples: "Explain more", "Give me details"
  
- feasibility: User asks about viability/feasibility
  Examples: "Is it feasible?", "Can we implement this?"
  
- novelty: User asks about innovation/uniqueness
  Examples: "Is it innovative?", "Is it unique?"
  
- general_chat: General conversation with no specific startup request
  Examples: "Hi how can you help me?", "How is the market?", "What's trending?"

RULES:
1. If user mentions a SPECIFIC problem → problem_solving
2. If user asks for ANY startup WITHOUT mentioning a problem → random_solution
3. If user references PREVIOUS discussion → follow_up or alternative_idea
4. If user asks for MORE about something already discussed → details
5. If user questions FEASIBILITY → feasibility
6. If user questions INNOVATION → novelty
7. If it's GENERAL conversation → general_chat

Return ONLY valid JSON (no explanations):
{{"detected_intents": [{{"intent": "intent_name", "confidence": "high/medium/low", "relevant_text": "the relevant part", "priority": 1}}], "primary_intent": "main_intent", "secondary_intents": ["other_intents"]}}

Examples:

Input: "Hi how can you help me"
Output: {{"detected_intents": [{{"intent": "general_chat", "confidence": "high", "relevant_text": "Hi how can you help me", "priority": 1}}], "primary_intent": "general_chat", "secondary_intents": []}}

Input: "I want to solve expensive education in Egypt"
Output: {{"detected_intents": [{{"intent": "problem_solving", "confidence": "high", "relevant_text": "solve expensive education", "priority": 1}}], "primary_intent": "problem_solving", "secondary_intents": []}}

Input: "Give me a startup idea"
Output: {{"detected_intents": [{{"intent": "random_solution", "confidence": "high", "relevant_text": "Give me a startup idea", "priority": 1}}], "primary_intent": "random_solution", "secondary_intents": []}}
"""

FULL_IDEA_TEMPLATE = """
    You are an expert in entrepreneurship and startup innovation focused on the MENA region.
    
     Your task is to generate a complete startup concept based on the given problem:
      {core_problem}
        

        Requirements:
        - Focus on realistic and practical solutions.
        - Adapt the idea for the Egypt or MENA market.
        - Use concise and clear text.
        - Provide multiple items for list fields when possible.

        Important Rules:
        - Return ONLY valid JSON.
        - Do NOT write any text outside JSON.
        - Do NOT add explanations or comments.
        - Follow the exact data types:
        - Text fields → string
        - Lists → array
        - Nested sections → object
        - novelty_score → number between 0 and 100
        - business_model MUST be an object (not a string).
        - feasibility MUST be an object.
        - market_analysis MUST be an object.
        - impact MUST be an object.
        - mvp_plan MUST be an object.

        Return the response using this exact structure:

        {{
        "problem_title": "",
        "problem_description": "",
        "root_cause": "",
        "target_users": "",
        "market_region": "Egypt or MENA",
        "why_now": "",
        "evidence_signals": [],

        "solution_name": "",
        "solution_description": "",
        "how_it_works": [],
        "key_features": [],
        "technology_stack": [],

        "business_model": {{
            "value_proposition": "",
            "revenue_streams": [],
            "pricing_model": "",
            "customer_acquisition": []
        }},

        "market_analysis": {{
            "market_size": "",
            "competitors": [],
            "competitive_advantage": ""
        }},

        "feasibility": {{
            "technical_feasibility": "Low",
            "market_feasibility": "Low",
            "risk_factors": []
        }},

        "novelty_score": 0,

        "impact": {{
            "economic_impact": "",
            "social_impact": ""
        }},

        "mvp_plan": {{
            "mvp_features": [],
            "first_steps": []
        }}
        }}
        Important:
        Return ONLY valid JSON.
        Do not repeat any section.
        Do not truncate the response.
        If you are unsure, return a shorter but complete JSON.
        If you cannot complete the JSON correctly, return a shorter but valid JSON.
        Never cut arrays or objects.
        Never leave fields incomplete.
"""

import json
from typing import List, Dict

def build_unified_prompt(
    detected_intents: List[Dict],
    extracted_data: Dict,
    context: str = None,
    primary_intent: str = None,
    idea_data: Dict = None,
    lang: str = "English"
) -> str:

    if not primary_intent:
        sorted_intents = sorted(detected_intents, key=lambda x: x.get("priority", 999))
        primary_intent = sorted_intents[0]["intent"]

    prompt = """
You are an expert in entrepreneurship and startup innovation focused on the MENA region.

Always answer clearly and practically.
Use the provided idea data as reference.
Base your response on the idea data provided.
"""

    # -------------------------
    # EXISTING IDEA CONTEXT
    # -------------------------

    if idea_data:
        idea_json = json.dumps(idea_data, indent=2)

        prompt += f"""

STARTUP IDEA DATA (REFERENCE):

{idea_json}

Important rules:
- Use this data as your source of truth when answering.
- Extract information from this data to answer user questions.
- Do NOT generate new ideas if this data exists.
- Base all your answers on this data.
"""

    prompt += f"\n\nPRIMARY REQUEST ({primary_intent}):\n"


    if primary_intent in ["problem_solving", "random_solution" , "alternative_idea"]:

        prompt += """
The user is asking for a startup idea solution.

Your task:
- Open with 1-2 sentences that reflect the user's pain point directly
- Then introduce the startup idea naturally (not as a title/header)
- Use bullet points ONLY for key features (max 4 bullets)
- Everything else must be flowing paragraphs
- Do NOT use repetitive prefixes like "هذه هي..." or "Here are the..."
- Do NOT start with self-introductions like "I am here to..." or "أنا هنا لـ..."
- The tone should feel like a founder pitching, not a report
- End with one strong closing sentence about the opportunity

CRITICAL:
- The JSON data is already saved separately for the backend
- Your ONLY job here is to write a compelling narrative for the user
- Do NOT reformat or list the JSON fields mechanically
- Use the data as inspiration to tell a story, not as a template to copy
- Respond in the SAME language the user wrote in (Arabic or English)
- Keep technical terms in English regardless of the response language (MVP, B2B, real-time tracking, etc.)
"""



    elif primary_intent == "follow_up":

        user_questions = extracted_data.get('questions', ['General questions about the idea'])
        questions_str = ', '.join(user_questions) if isinstance(user_questions, list) else user_questions

        prompt += f"""
The user is following up with questions about the existing idea.

User's questions/requests:
{questions_str}

Your task:
- Answer based on the idea data provided above
- Expand or clarify specific aspects
- Provide detailed explanations
- Return a clear narrative response (not JSON)

Focus on the aspects the user is asking about.
"""


    elif primary_intent == "details":

        prompt += """
The user wants more detailed information about the startup idea.

Your task:
- Provide comprehensive details based on the idea data
- Expand on implementation, business model, and execution
- Return a detailed narrative response (not JSON)
- Cover:
  * Detailed problem analysis
  * Complete solution description
  * Implementation steps and timeline
  * Business model breakdown
  * Target customer segments
  * Revenue streams and pricing
  * Required resources and team
  * Key success metrics

Format as detailed sections or bullet points.
Be specific and practical.
"""


    elif primary_intent == "general_chat":

        topic = extracted_data.get('core_problem', 'general startup topics')

        prompt += f"""
The user wants to have a general discussion about: {topic}

Your task:
- Provide thoughtful insights and analysis
- Use the idea data as context if available
- Return a conversational, informative response
- Be helpful and engaging

Format as clear narrative paragraphs.
"""


    secondary_intents = extracted_data.get("secondary_intents", [])

    if secondary_intents:

        prompt += "\n\nADDITIONAL ASPECTS TO ADDRESS:\n"

        for intent in secondary_intents:

            if intent == "details":

                prompt += """
- Include more detailed information about:
  * Implementation steps and timeline
  * Business model specifics
  * Target customers
  * Revenue streams and pricing strategy
  * Team and resources needed
"""

            elif intent == "feasibility":

                prompt += """
- Analyze and discuss feasibility:
  * Technical feasibility based on the idea data
  * Market feasibility in Egypt/MENA region
  * Risk factors and mitigation strategies
  * Resource requirements
  * Realistic timeline to MVP
  * Success probability
"""

            elif intent == "novelty":

                prompt += """
- Evaluate innovation and uniqueness:
  * What's new and innovative about this solution
  * Competitive advantages over existing solutions
  * Unique value proposition
  * Market differentiation factors
  * Why customers would choose this
"""

    context_text = context or "Startup discussion focused on solving real problems in Egypt and the MENA region."

    prompt += f"""

CONTEXT:
{context_text}

USER REQUIREMENTS:
{', '.join(extracted_data.get('requirements', ['comprehensive analysis']))}

CONSTRAINTS:
{', '.join(extracted_data.get('constraints', ['Egypt/MENA market focus']))}

IMPORTANT INSTRUCTIONS:
- Return a clear, practical, narrative response (NOT JSON or code)
- Base everything on the idea data provided
- Keep language simple and actionable
- Focus on Egypt/MENA market realities
- Be specific with examples where possible
- Do NOT return the raw JSON data
- Format response as readable text or bullet points
- Make it engaging and professional
- Do NOT start with "نعم" or any filler opener
- Do NOT repeat or summarize at the end
- Start directly with the idea or the problem hook
"""
    prompt += f"""

LANGUAGE INSTRUCTIONS:
- The user wrote in {lang}.
- You MUST respond in {lang}.
- Keep all technical and startup terms in English (e.g., MVP, B2B, SaaS, revenue streams, market fit).
- All explanations, descriptions, and narrative text must be in {lang}.
- Keep technical ENGLISH terms as-is without adding Arabic suffixes
- BAD: "transparentة", "efficientة"
- GOOD: "شفافة (transparent)", "فعّالة (efficient)"
"""

    return prompt