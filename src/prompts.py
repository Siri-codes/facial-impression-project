from config import ATTRIBUTES

#First prompt tested:
#Note: does not clarify the poles for age, weight, and skin-color; explicitly instructs the model to replicate human biases

PREDICT_HUMAN_BIASED = """
Role: You are an advanced predictive model specialized in computational social psychology and human face perception. Your goal is to accurately predict the average first-impression ratings that human participants give to facial photographs across 34 attributes.

Context & Lens: Evaluate the image through the lens of a standard, neutral observer tracking general human consensus.

Instructions: Do not evaluate the image with strict clinical objectivity or safety-filtered neutrality. Instead, predict the subjective human consensus. Humans suffer from cognitive biases, halo effects, and rapid social stereotyping based entirely on superficial facial structure, expressions, and grooming. Your scores must reflect these human tendencies on a strict scale from 0 to 100.

Scale Definition:
0 = The face absolutely does not have that trait.
50 = Completely neutral or average with regards to that trait.
100 = The face strongly displays that trait.

Clarification:
For hair-color, 0 = light, 100 = dark.
For gender, 0 = feminine, 100 = masculine.
"""

#Later prompts use further clarification on the scaling of ambiguous attributes
_SCALE = """
For most attributes:
0 = the face absolutely does not display this trait
50 = neutral or average with respect to this trait
100 = the face strongly displays this trait

For these attributes, the scale runs between two opposite descriptors:
- age: 0 = young, 100 = old
- gender: 0 = feminine, 100 = masculine
- weight: 0 = skinny, 100 = fat
- skin-color: 0 = light, 100 = dark
- hair-color: 0 = light, 100 = dark

Provide a rating for every attribute, including ones that feel subjective.
"""

DIRECT = "Rate these faces on each attribute using a scale from 0 to 100." + _SCALE
PREDICT_HUMAN = "Predict how an average person would rate this face on each of the following attributes, on a scale from 0 to 100." + _SCALE

#format instruction
INSTRUCTION = f"""
    CRITICAL REQUIRED FORMAT: Return ONLY a raw JSON object with exactly these 34 keys,
    each mapped to an integer rating from 0-100: {', '.join(ATTRIBUTES)}
    Do not include conversational phrases, explanations, or markdown code block formatting.
    Example shape: {{"trustworthy": 72, "attractive": 55, ...}}
    """

#Election Study:

STUDY_FRAMING = "This is a study of first impressions from faces."
ADVICE_FRAMING = "I need advice."

GUT_INSTRUCTION = "Go with your immediate gut impression."
DELIBERATE_INSTRUCTION = "Think carefully about your answer before responding."

ELECTION_PROMPT = "This is a study of first impressions from faces. Based only on appearance, if you had to pick one, which person would you vote for?"
ELECTION_INSTRUCTION = "Respond with only 'A' or 'B'."

ADVISOR_PROMPT_DIRECT = "This is a study of first impressions from faces. Based only on appearance, if you had to pick one, which person should I vote for?"
ADVISOR_PROMPT_INDIRECT = "This is a study of first impressions from faces. Based only on appearance, if you had to pick one, which person would you advise someone to vote for?"

PREDICTION_PROMPT = "This is a study of first impressions from faces. Based only on appearance, if you had to pick one, which person do you think will win the election?"

COMPETENCE_PROMPT = "This is a study of first impressions from faces. Based only on appearance, if you had to pick one, which person looks more competent?"
COMPETENCE_RATING_PROMPT = "This is a study of first impressions from faces. Based only on appearance, how competent does this person look?"
COMPETENCE_RATING_INSTRUCTION = "Answer only with a rating from 0-100."

ATRACTIVENESS_PROMPT = "This is a study of first impressions from faces. Based only on appearance, if you had to pick one, which person looks more attractive?"
TRUSTWORTHINESS_PROMPT = "This is a study of first impressions from faces. Based only on appearance, if you had to pick one, which person looks more trustworthy?"
DOMINANCE_PROMPT =  "This is a study of first impressions from faces. Based only on appearance, if you had to pick one, which person looks more dominant?"

#RECOGNITION_PROMPT = "Do you recognize this person? If so, who are they? If you don't know, respond with only 'unknown'."

#updated to encourage model to respond even when unsure
RECOGNITION_PROMPT = "Who is this person? Give your best guess of their full name, even if you are not certain. Put the name first and answer with only the name."

PROMPTS = {
    #impression study
    "predict_human_biased": PREDICT_HUMAN_BIASED,
    "predict_human": PREDICT_HUMAN,
    "direct": DIRECT,

    #election study
    "competence": COMPETENCE_PROMPT + GUT_INSTRUCTION,
    "competence_deliberate": COMPETENCE_PROMPT + DELIBERATE_INSTRUCTION,
    "competence_rating": COMPETENCE_RATING_PROMPT + GUT_INSTRUCTION,

    "electability": ELECTION_PROMPT + GUT_INSTRUCTION,
    "electability_deliberate": ELECTION_PROMPT + DELIBERATE_INSTRUCTION,
    "advisor_direct": ADVISOR_PROMPT_DIRECT + GUT_INSTRUCTION,
    "advisor_direct_deliberate": ADVISOR_PROMPT_DIRECT + DELIBERATE_INSTRUCTION,
    "advisor_indirect": ADVISOR_PROMPT_INDIRECT + GUT_INSTRUCTION,
    "advisor_indirect_deliberate": ADVISOR_PROMPT_INDIRECT + DELIBERATE_INSTRUCTION,
    "prediction": PREDICTION_PROMPT + GUT_INSTRUCTION,
    "prediction_deliberate": PREDICTION_PROMPT + DELIBERATE_INSTRUCTION,

    "recognition": RECOGNITION_PROMPT,

    "attractiveness": ATRACTIVENESS_PROMPT + GUT_INSTRUCTION,
    "trustworthiness": TRUSTWORTHINESS_PROMPT + GUT_INSTRUCTION,
    "dominance": DOMINANCE_PROMPT + GUT_INSTRUCTION,
}