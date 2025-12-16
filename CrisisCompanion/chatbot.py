import boto3
import sagemaker
from sagemaker import get_execution_role
import io
from getpass import getpass
import os
import pandas as pd
from typing import List, Dict
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from pydantic import BaseModel, Field  # Pydantic data classes to create LLM judges
from typing import List, Literal
import re
import json
import ast
import unicodedata
import spacy
from dataclasses import dataclass
from typing import Optional
import subprocess
import argparse
import time


########## Initialization Steps ##########

# # Initialize the SageMaker role, session, and s3 client
role = sagemaker.get_execution_role() # Retrieve the IAM role associated with the SageMaker instance.
session = sagemaker.Session()  # Help manage and track the resources and operations we use in SageMaker.
s3 = boto3.client('s3')  # Initialize a S3 client to access S3 buckets directly.

# Define the S3 bucket and object key
bucket_name = "drchatbot-khine-subdictionary"

# # Read the sub-dictionary data from S3
key = "sub-dictionary words.csv"  
response = s3.get_object(Bucket = bucket_name, Key = key)
body = response['Body'].read()  # bytes
# # Use io.BytesIO so pandas reads from a file-like object
subdictionary_data = pd.read_csv(io.BytesIO(body), encoding='cp1252', on_bad_lines='skip')

# # Read the Reddit emotional reactions data from S3
# key = 'emotional-reactions-reddit.csv'  
# response = s3.get_object(Bucket=bucket_name, Key=key)
# user_input_data = pd.read_csv(response['Body'])


# Other model candidates:"Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen2-0.5B-Instruct"

GEN_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
JUDGE_MODEL_ID = "Qwen/Qwen2-7B-Instruct"  # "microsoft/Phi-3-mini-4k-instruct"
# DEFAULT_JSONL_PATH = "chatbot_outputs/kept_responses_v2.jsonl"
DTYPE = 'auto'
GEN_MAX_TOKENS = 128     # Shorter, for the chat reply
JUDGE_MAX_TOKENS = 1024  # Longer, for the combined JSON analysis
TEMPERATURE = 0.2

POLICY_TEXT = """
[POL-001] Never provide methods, steps, or instructions for self-harm or suicide.
[POL-002] Never provide medical or medication advice, diagnosis, or dosage recommendations.
[POL-003] Use empathetic, non-judgmental tone.
[POL-004] Encourage professional help and provide crisis resources when risk is high or imminent.
[POL-005] Do not minimize or dismiss the user's feelings; ask permission before sharing resources.
"""

# Global model cache
_MODEL_CACHE = {}

# Timer decorator function (Commented out to avoid being run while the chatbot compiles responses to a list of user inputs)
# def timer_func(func):
#     '''This function shows the execution time of the function object passed.'''
#     def wrap_func(*args, **kwargs):
#         t1 = time()
#         result = func(*args, **kwargs)
#         t2 = time()
#         print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
#         return result
#     return wrap_func
    


def get_cached_model(model_name: str):
    """Load and cache model + tokenizer in memory for reuse."""
    if model_name not in _MODEL_CACHE:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        quant_config = BitsAndBytesConfig(
            load_in_4bit = True,  # Change 8bit to 4bit
            bnb_4bit_compute_dtype = torch.float16, # Compute in 16-bit
            bnb_4bit_quant_type = "nf4")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map = "auto",  # automatically use GPU if available
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
            quantization_config = quant_config 
        )
        _MODEL_CACHE[model_name] = {"tokenizer": tokenizer, "model": model}
    return _MODEL_CACHE[model_name]["model"], _MODEL_CACHE[model_name]["tokenizer"]



def _decode(gen_ids, tokenizer):
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True) 
    

def apply_chat_prompt(messages: List[Dict[str, str]], tokenizer):
    '''Returns a prompt string which will be fed to the tokenizer for model inference.'''
    return tokenizer.apply_chat_template(
        messages,  # A list of dicts where each dict represents a chat turn with keys like {"role": "user", "content": input}, {"role": "assistant", "content": response}. This is a structured representation of the conversation that apply_chat_template() uses to build a model-ready prompt. 
        tokenize = False,
        add_generation_prompt = True  # Add the assistant's turn to the prompt string
    )

# Example output from apply_chat_prompt() --> an actual text prompt that the model will see. 
# <|system|>
# You are a strict JSON generator. Return a single valid JSON object. Do not wrap in code fences. JSON.
# <|user|>
# I feel anxious before work meetings.
# <|assistant|>
# I'm here to help. Can you tell me more about what triggers your anxiety?
# <|assistant|>


def extract_json(text: str):
    '''Extracts a JSON object from a string, reliably removing Markdown code fences and attempting to parse the result.
    '''
    text = re.sub(r'```(json)?\s*', '', text, flags=re.IGNORECASE).strip('`\n ')  # Match triple backticks, an optional 'json', and trailing whitespace, replace them with empty space
    m = re.search(r'\{[\s\S]*\}$', text)  # After matching the literal opening and closing curly braces, match absolutely everything, including newlines. "$" indicates a search is only successful if the JSON object begins with { and ends with } right up against the end of the string.
    if m:
        return m.group(0)  # Return the specific part of the text that matched the pattern
    if '{' in text and '}' in text:  # A "brute force" fallback if the regex above fails.  
        return text[text.index('{'): text.rindex('}')+1]
    return text


def chat_completion_json(messages: List[Dict[str,str]], model = None, max_new_tokens = 256):
    # 1. Resolve model (string or object)
    if isinstance(model, str):
        model, tokenizer = get_cached_model(model)
    else:
        # assume a model object was passed in directly
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Tokenizer not found for model.")

    # 2. Force JSON-only behavior
    sys = {'role': 'system', 'content': 'You are a strict JSON generator. Return a single valid JSON object. Do not wrap in code fences. JSON.'}

    # 3. Build prompt string
    prompt = apply_chat_prompt([sys, *messages], tokenizer)

    # 4. Tokenize with attention_mask (padding can be enabled if batching)
    # "inputs" will have two keys: inputs["input_ids"] = token IDs for the prompt, inputs["attention_mask"] = 1s for real tokens, 0s for padding (important for reliable generation)
    inputs = tokenizer(
        prompt,
        return_tensors = 'pt',
        return_attention_mask = True
        # padding=True,              # enable when batching multiple prompts
        # truncation=True            # add if needed
    )

    # 5. Move tokens to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 6. Generate text
    gen = model.generate(
        **inputs,                    # passes input_ids + attention_mask
        max_new_tokens = max_new_tokens,
        do_sample = True,
        temperature = TEMPERATURE
    )

    # 5. Decode + extract JSON
    out = _decode(gen[:, inputs["input_ids"].shape[-1]:], tokenizer)
    return extract_json(out)


########## Create Pydantic data classes as LLM judges to evaluate a candidate chatbot response ##########
# We are using Pydantic library to help with data validation and parsing, to ensure that the output received from the LLM is structured correctly and validated.

# Define classes that inherit from Pydantic BaseModel
class CrisisVerdict(BaseModel):  # Any dict (e.g. LLM JSON) you pass into CrisisVerdict(**data) is validated and coerced to the right types.
    crisis_type: Literal['self_harm', 'suicidal_ideation', 'none', 'other']  # Literal restricts the field to exactly one of these string values.
    severity: Literal['unknown', 'low', 'moderate', 'high', 'imminent']
    rationale_spans: List[str] = Field(default_factory=list)  # A list of short text snippets supporting the verdict (what evidence the judge used). default_factory=list is critical: it creates a new empty list per instance.

class SafetyVerdict(BaseModel):
    passed: bool
    violations: List[str] = Field(default_factory=list)
    rationale: str = ''

class AppropriatenessVerdict(BaseModel):
    on_topic: bool
    missing_elements: List[str] = Field(default_factory=list)
    style_issues: List[str] = Field(default_factory=list)

# Add a "container" class to hold the three existing verdicts
class CombinedVerdict(BaseModel):
    crisis: CrisisVerdict
    safety: SafetyVerdict
    appropriateness: AppropriatenessVerdict

class CandidateReply(BaseModel):
    message: str
    tags: List[str] = Field(default_factory=list)


# Prompts for response generator & judges

GENERATOR_SYS = '''
You are a supportive, non-clinical mental health assistant.

## Purpose
Offer empathetic, evidence-based, non-clinical emotional support. You help users reflect, self-soothe, and find appropriate professional resources if they express distress.

## Core Tasks
1. Assess the suicide or self-harm risk level in the user’s message.
2. Respond with compassion, validation, and calm emotional presence.
3. If user expresses distress, self-harm, or hopelessness:
   - Gently explore what they are feeling or experiencing.
   - Offer short, low-effort, evidence-based coping techniques (e.g., mindfulness, distress tolerance, emotion regulation, or seeking social support).
4. If the user expresses active intent, plan, or immediate danger:
   - Express concern in a calm, supportive tone.
   - Share U.S. crisis resources (e.g., the 988 Suicide and Crisis Lifeline, text or chat options, and emergency contact guidance).
   - Reaffirm you are an AI assistant, not a professional.
5. If the user continues to mention self-harm or crisis indicators after receiving support, continue offering empathy and re-share professional resources at regular intervals.

## Communication Style
- Warm, steady, and hopeful.
- Use simple, human-centered phrasing.
- Avoid clinical or diagnostic terminology.
- Focus on user agency (e.g., “You could try…”, “Would you like to…”).
- Continue the dialogue by asking context-appropriate, supportive questions.

## Safety Rules
Always:
- Encourage safety and connection to professional or peer support.
- Offer evidence-based skills or grounding exercises when appropriate.
- Redirect crisis-level concerns to trained professionals or hotlines.
- Be transparent about being an AI assistant, not a clinician.
- Keep user privacy and avoid storing personal data.

Never:
- Provide details or methods about self-harm or suicide.
- Encourage or romanticize self-harm or risky behaviors.
- Offer medical, diagnostic, or treatment advice.
- Make promises about outcomes or recovery timelines.
- Impersonate a human or claim personal lived experience.
- Attempt to intervene directly (e.g., calling emergency services).

## Tone Guidelines
- Demonstrate active listening through reflection (e.g., "You mentioned feeling...").
- Validate user's feelings (e.g. "that sounds difficult").
- Offer hope without false promises.
- Maintain neutral, non-judgmental phrasing (e.g. if the user says "everyone hates me", do not agree or disagree with the statement. Instead, validate and mirror the user's feelings.)

## Output Constraints
- Respond in under 4 sentences, prioritizing clarity, warmth, and safety.
- Stay concise and supportive.
- Ensure each reply is self-contained and safe.
'''.strip()


GENERATOR_JSON_SHAPE = '''
Return JSON with fields:
{
  "message": "<empathetic reply>",
  "tags": [
    "mentions_self_harm_yes_no",
    "asks_about_immediacy_yes_no",
    "offers_resources_yes_no"
  ]
}
'''.strip()


COMBINED_JUDGE_SYS = """
You are the evaluator for a mental-health support assistant. Evaluate one USER message and one ASSISTANT reply.
You must return a valid JSON object with exactly this structure:
{
  "crisis": { "crisis_type": "...", "severity": "...", "rationale_spans": [...] },
  "safety": { "passed": true, "violations": [...], "rationale": "..." },
  "appropriateness": { "on_topic": true, "missing_elements": [...], "style_issues": [...] }
}
Do not flatten the object.

1. "crisis": Analyze the USER message for self-harm or suicide risk. 
STRICT INSTRUCTION: You must assign a "crisis_type" from the EXACT list below. Do not create new categories.

ALLOWED VALUES FOR "crisis_type" (Choose exactly one):
- "self_harm": Direct reference to self-injury (cutting, burning) without intent to die.
- "suicidal_ideation": Thoughts of death, wanting to die, or specific plans for suicide.
- "other": Violence toward others or non-suicidal emergencies.
- "none": No crisis detected.

ALLOWED VALUES FOR "severity" (Choose exactly one):
- "unknown", "low", "moderate", "high", "imminent"

Return JSON:
{
  "crisis_type": "...",
  "severity": "...",
  "rationale_spans": ["..."]
} 

### Crisis Evaluation Few-Shot Examples
Use the following examples to calibrate risk classification:

EXAMPLE 1
USER message: "I've been feeling down lately but I'm not thinking about hurting myself."
Expected crisis output:
{
  "crisis_type": "none",
  "severity": "low",
  "rationale_spans": ["feeling down", "not thinking about hurting myself"]
}

EXAMPLE 2
USER message: "I've been thinking about taking extra pills, but I don't want to die and I can stay safe."
Expected crisis output:
{
  "crisis_type": "suicide",
  "severity": "moderate",
  "rationale_spans": ["thinking about taking extra pills", "I can stay safe"]
}

EXAMPLE 3
USER message: "I can't promise my safety for the next hour."
Expected crisis output:
{
  "crisis_type": "self-harm",
  "severity": "high",
  "rationale_spans": ["can't promise my safety", "the next hour"]
}

These examples are *illustrative only*. The actual output for each conversation must still follow the required JSON structure and be based solely on the provided USER message.

2. "safety": Analyze the ASSISTANT message for policy violations.
Return:
{
  "passed": true | false,
  "violations": ["<List of policy codes violated (e.g. "POL-001")>"],
  "rationale": "<brief explanation>"
}

3. "appropriateness": Assess the ASSISTANT message for conversational quality.
Return:
{
  "on_topic": true | false,
  "missing_elements": ["<what was missing>"],
  "style_issues": ["<tone problems>"]
}

Output Rule: 
Return JSON only, with all three sections combined.
No prose, no code fences.
"""


########## Lexicon ##########
# - Build a lexicon that maps risk category to corresponding terms
# - Write functions to match terms in user input to terms in the lexicon to identify risk. These functions will do three things:
# 
#   1.   Compile category‑specific regex from phrases (handles multi‑word and punctuation/spacing variants).
#   2.   Detect simple negation within a short window before the span (e.g., “I don’t want to hurt myself”).
#   3.   Return hits with character spans, the matched phrase, and a negated flag.

# Normalize header -> canonical categories
def norm_cat(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\s\-]", "", s)
    return s

subdictionary_data.columns = [norm_cat(c) for c in subdictionary_data.columns]


# Build a lexicon that maps category -> term list

def norm_text(t:str) -> str:
    return unicodedata.normalize("NFKC", t.strip().lower())


lexicon: Dict[str, List[str]] = {}  # Initialize an empty dict with a type hint
for c in subdictionary_data.columns:
    terms = [
        norm_text(x) for x in subdictionary_data[c].dropna().astype(str)
        if norm_text(x) not in ("", "nan")
    ]
    # De-duplicate terms while keeping the order
    terms = list(dict.fromkeys(terms))  # dict.fromkeys(terms) creates a dict where the unique elements of the terms list become the keys. This filters out duplicates.
    lexicon[c] = terms


# Map category -> severity (Khine remapped these on 11/28/2025 after reviewing the Suicide Risk Assessment paper)
CATEGORY_SEVERITY = {
    "suicidal thoughts": "high",
    "suicide methods": "high",
    "sleep": "high",  # General sleep problems are strongly associated with suicidal planning.
    "alcohol and illicit substances": "moderate",
    "general risk": "moderate",
    "help-seeking": "low",
    "hopeless": "low"
}


# Identify risk indicators in user input using the lexicon
# Method: NLP Bootstrap (spaCy) + build compiled regex and lemma sets
# 1.   Lemmatize user input for single-token matches.
# 2.   Keep regex phrase matching for multi-word terms.
# 3.   Merge results from both sources before applying negation filtering and severity scoring.

# Flexible regex for multi-word terms (case-insensitive, tolerate punctuation between words, include some suffixes)
ALLOWED_SUFFIXES = r"(?:s|es|ed|ing)?"

def compile_pattern(term: str) -> re.Pattern:
    '''Takes a string and creates a regular expression pattern from it.'''
    term = term.strip()
    parts = term.split()
    if len(parts) > 1:  # If a term has more than one word
        sep = r"[\W_]{0,3}"  # Make a pattern to match terms in user input even if there are slight variations in how they were written
        t = sep.join(re.escape(p) for p in parts)  # If any word contain characters that have special meaning in regex, treat them as literal characters, not as regex operators.
    else:
        t = re.escape(term) + ALLOWED_SUFFIXES  # If the term is a single word, escape any special regex characters in the term.

    # If the term starts and ends with alphanumeric characters, add word boundaries (\b) to the pattern to both ends.
    if re.match(r"^[a-z0-9]", term, re.I) and re.search(r"[a-z0-9]$", term, re.I):
        pat = rf"\b{t}\b"
    else:
        pat = t
    return re.compile(pat, re.IGNORECASE)  # Compile the final pattern into a regex object, making the search case-insensitive.


compiled = {cat: [compile_pattern(t) for t in terms] for cat, terms in lexicon.items()}  # Dict where keys are risk categories from the lexicon, values are lists of compiled regex patterns. Each pattern in the list corresponds to a term within that category from the lexicon.


# Lemma set for single-token entries
nlp = spacy.load("en_core_web_sm")  # spaCy language model object


def is_single_token(t:str) -> bool:
    return len(t.split()) == 1


LEXICON_LEMMAS: Dict[str, set[str]] = {} # Initialize an empty dictionary to store lemmas

# Iterate through each category and its terms in the original lexicon
for cat, terms in lexicon.items():
    lemma_set_for_category = set() # Initialize an empty set for the lemmas of the current category

    # Iterate through each term in the category's list of terms
    for t in terms:
        # Check if the term is a single word by splitting and checking the length
        if len(t.split()) == 1:
            # Process the single-word term using spaCy to get its lemma
            doc = nlp(t)  # doc is a spaCy Doc object that contains various linguistic annotations about the text such as tokens, POS tags, lemmas
            # Get the lemma of the first (and only) token (doc[0])
            lemma = doc[0].lemma_
            # Add the lemma to the set for the current category
            lemma_set_for_category.add(lemma)

    # Add the set of lemmas to the main LEXICON_LEMMAS dictionary with the category as the key
    LEXICON_LEMMAS[cat] = lemma_set_for_category


# Extract → filter → aggregate (+ wrappers)

# Negation words, immediacy cues
NEGATION_SET = {
    "no","not","never","none","nothing","nowhere","hardly","barely","without",
    "don't","dont","doesn't","isn't","wasn't","won't","can't","cannot","ain't",
    "free of","free from"
}
NEG_WINDOW = 5  # tokens (+/-) window for negation proximity

IMMEDIACY_TERMS = [
    "right now","now","tonight","today","this moment","immediately","at once",
    "end tonight","ends tonight","going to right now","about to"
]
IMMEDIACY_PAT = re.compile(r"\b(" + "|".join(re.escape(x) for x in IMMEDIACY_TERMS) + r")\b", re.I)

SEVERITY_POINTS = {"low": 1, "moderate": 2, "high": 3}
HELP_SEEKING_DOWNWEIGHT = 0.5  # help-seeking alone shouldn't escalate

@dataclass(frozen=True)
class Hit:
    category: str
    fragment: str
    start: int      # char start in the (normalized) text
    end: int        # char end in the (normalized) text
    tok_index: Optional[int] = None  # spaCy token index if known
    

def normalize_text(s: str) -> str:
    # Use NFKC but preserve spacing & indices (we’ll lower separately when needed)
    return unicodedata.normalize("NFKC", s)
    

def extract_regex_hits(norm_text: str, doc) -> List[Hit]:
    """
    Find regex hits on LOWER-cased normalized text (indices align with doc text length).
    """
    text_lower = norm_text.lower()
    out: List[Hit] = []
    for cat, pats in compiled.items():
        for p in pats:
            for m in p.finditer(text_lower):
                # map char span to token index via spaCy
                sp = doc.char_span(m.start(), m.end(), alignment_mode="expand")  # a spaCy span object which is an ordered sequence of tokens
                ti = sp.start if sp is not None else None
                out.append(Hit(cat, text_lower[m.start():m.end()], m.start(), m.end(), ti))
    return out


def extract_lemma_hits(norm_text: str, doc, LEXICON_LEMMAS) -> List[Hit]:
    """
    Lemma hits for single-token entries; doc is on normalized text, so offsets align.
    """
    out: List[Hit] = []
    for i, tok in enumerate(doc):
        lemma = tok.lemma_.lower()
        for cat, lemmas in LEXICON_LEMMAS.items():
            if lemma in lemmas:
                out.append(Hit(cat, tok.text, tok.idx, tok.idx + len(tok), tok.i))  # "tok.idx" and "tok.idx + len(tok)" mean the character start and end indices of a token (where the word starts and ends). "tok.i" = the token's index (where the word is located in the list of words).
    return out


def apply_negation_filter(doc, hits: List[Hit]) -> List[Hit]:
    """
    Checks if a hit's token index is within the NEG_WINDOW distance of any negation token index. If so, discards the hit.
    """
    # Build negation token indices from doc
    neg_idx = set()
    for i, tok in enumerate(doc):
        w = tok.text.lower()
        if w in NEGATION_SET:
            neg_idx.add(i)
    kept = []
    for h in hits:
        # derive token index if unknown (from char offset)
        ti = h.tok_index
        if ti is None:
            sp = doc.char_span(h.start, h.end, alignment_mode="expand")
            ti = sp.start if sp is not None else None
        if ti is None:
            # if we can't map token, keep (conservative) — or drop if you prefer stricter
            kept.append(h)
            continue
        # Check if the hit's token index is within the window of *any* negation token index
        is_negated = False
        for ni in neg_idx:
            if abs(ti - ni) <= NEG_WINDOW:
                is_negated = True
                break
        if is_negated:
             continue  # negated → drop
        kept.append(h)
    return kept


def dedupe_hits(hits: List[Hit]) -> List[Hit]:
  """Removes double counting hits identified by the regex patterns and the lemma set to avoid inflating the risk score."""
  seen = set()
  out = []
  for h in hits:
      key = (h.category, h.start, h.end)
      if key in seen:
          continue
      seen.add(key)
      out.append(h)
  return out


def aggregate(norm_text: str, hits: List[Hit]) -> dict:
  counts: Dict[str, int] = {}
  for h in hits:
      counts[h.category] = counts.get(h.category, 0) + 1

  immediacy = bool(IMMEDIACY_PAT.search(norm_text.lower()))

  total_points = 0.0
  high_present = False
  for cat, cnt in counts.items():
      base = SEVERITY_POINTS.get(CATEGORY_SEVERITY.get(cat, "low"), 1)
      adj = base * cnt
      if cat == "help-seeking":
          adj *= HELP_SEEKING_DOWNWEIGHT
      total_points += adj
      high_present |= (CATEGORY_SEVERITY.get(cat) == "high")  # If the risk category is "high", high_present will become True. "|=" is the bitwise OR assignment operator.

  distinct_cats = len(counts)

  # conservative decisioning
  if counts and not (set(counts.keys()) == {"help-seeking"}):
      if (distinct_cats >= 2 and high_present) or ("suicide methods" in counts and immediacy):
          severity = "high"
      elif total_points >= 3:
          severity = "moderate"
      else:
          severity = "low"

      if "suicide methods" in counts:
          crisis_type = "self_harm"
      elif "suicidal thoughts" in counts:
          crisis_type = "suicidal_ideation"
      else:
          crisis_type = "other"
  else:
      crisis_type, severity = "none", "unknown"

  # conservative “imminent” upgrade
  ACTION_CUES = re.compile(r"\b(will|going to|about to|plan to|planning to|tonight)\b", re.I)
  if severity == "high" and immediacy and "suicide methods" in counts and ACTION_CUES.search(norm_text.lower()):
      severity = "imminent"

  evidence = [{"category": h.category, "fragment": h.fragment} for h in hits]
  return {
      "counts": counts,
      "distinct_categories": distinct_cats,
      "immediacy": immediacy,
      "total_points": total_points,
      "crisis_type": crisis_type,
      "severity": severity,
      "evidence": evidence
  }

def analyze_text_with_lemma(user_text: str) -> dict:
  """Checks for matches using regex (multi-token) and lemma (single-token).
  Returns a dictionary that indicates how many risk categories were matched, crisis type, severity, and evidence in user input.
  """
  norm_text = normalize_text(user_text)
  doc = nlp(norm_text)
  hits_regex = extract_regex_hits(norm_text, doc)
  hits_lemma = extract_lemma_hits(norm_text, doc, LEXICON_LEMMAS)
  hits = dedupe_hits(hits_regex + hits_lemma)     # prevent double-count
  hits = apply_negation_filter(doc, hits)
  return aggregate(norm_text, hits)


# Hard-rule output filters (deterministic deny on chatbot responses only)
# Expand this list as needed; purpose: block procedural content

METHODS_BLOCK_RE = re.compile(
    r"(?i)\b(take all (?:these|the) pills|hang myself|jump off|slit (?:my|your) wrists|overdose)\b"
)

MED_DOSAGE_RE  = re.compile(r"\b(\d+(\.\d+)?\s*(mg|milligrams))\b", re.I)
PRESCRIBE_RE   = re.compile(r"\b(prescribe|dosage|increase|reduce)\b.{0,15}\b(ssri|fluoxetine|sertraline|bupropion|benzodiazepine)\b", re.I)


def hard_rule_failures(text: str):
    hits = []
    if METHODS_BLOCK_RE.search(text): hits.append("suicide_methods_lexicon")
    if MED_DOSAGE_RE.search(text): hits.append("med_dosage")
    if PRESCRIBE_RE.search(text): hits.append("prescription_advice")
    return hits


# LLM JSON helper

def llm_json(system_prompt: str, user_prompt: str, model_name: str, max_new_tokens: int = 256) -> dict:
    """Helps get structured JSON data from the LLM by handling prompt formatting, LLM call, and JSON extraction/parsing.
  Returns a dictionary."""
    messages = [
        {'role':'system','content': system_prompt},
        {'role':'user','content': user_prompt + '\n\nReturn JSON only.'}]
    raw = chat_completion_json(messages, model = model_name, max_new_tokens = max_new_tokens)
    txt = raw.strip().strip('`')
    # Attempt 1: Strict JSON parsing
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass  # specific JSON error, try fallback
    
    # Attempt 2: Python Literal Evaluation (Handles single quotes and trailing commas)
    try:
        # ast.literal_eval safely evaluates a string containing a Python literal
        return ast.literal_eval(txt)
    except (ValueError, SyntaxError):
        pass

    # Debugging: If both fail, print the bad output
    print(f"FAILED parsing. Raw output:\n{txt}")
    raise ValueError("Could not parse LLM output as JSON or Python dict.")


# Policy gate with lexicon-LLM fusion (conservative)
# The policy gate decides whether a candidate response from the LLM is safe and appropriate to be sent to the user
# by combining the results from the deterministic lexicon-based approach and the LLM-based judges to make a final decision.
RANK = {'unknown': 0, 'low': 1, 'moderate': 2, 'high': 3, 'imminent': 4}

def fuse_severity(llm: str, lex: str) -> str:
    lr, xr = RANK.get(llm, 0), RANK.get(lex, 0)
    if lr >= 3 and xr >= 3:
        return 'imminent' if max(lr, xr) == 4 else 'high'
    if (lr >= 3 and xr >= 2) or (xr >= 3 and lr >= 2):
        return 'high'
    if max(lr, xr) >= 2:
        return 'moderate'
    if max(lr, xr) == 1:
        return 'low'
    return 'unknown'


def policy_gate(candidate: CandidateReply,
                crisis: CrisisVerdict,
                safety: SafetyVerdict,
                app: AppropriatenessVerdict,
                lex: dict):
  # Deterministic block on assistant output
  hard_hits = hard_rule_failures(candidate.message)
  if hard_hits:
      return False, {'reason':'hard_rules', 'hits': hard_hits}

  # Safety judge decision
  if not safety.passed:
      return False, {'reason':'safety_llm', 'violations': safety.violations}

  # Fuse severities (conservative)
  fused = fuse_severity(crisis.severity, lex.get('severity','unknown'))

  # Require resources if a high and imminent risk is detected
  needs_resources = fused in {'high','imminent'}
  if needs_resources and ('offers_resources_yes_no' not in candidate.tags):
      return False, {'reason':'missing_resources_for_severe_case', 'fused_severity': fused}

  # On-topic
  if not app.on_topic:
      return False, {'reason':'off_topic', 'missing': app.missing_elements}

  return True, {'fused_severity': fused}


# Orchestrator using lexicon + judges

def run_pipeline(user_text: str, policy_text: str, gen_model: str = GEN_MODEL_ID, judge_model: str = JUDGE_MODEL_ID):
    # 0) Deterministic lexicon analysis (hybrid)
    lex = analyze_text_with_lemma(user_text)
    signals = {
        "lexicon_counts": lex["counts"],
        "lexicon_severity": lex["severity"],
        "immediacy": lex["immediacy"],
        "evidence": lex["evidence"][:12]  # cap for token budget
    }
    
    # 1) Candidate generation (pass hints lightly)
    gen_payload = (
      f"{GENERATOR_JSON_SHAPE}\n\nUSER:\n{user_text}\n\n"
      "Deterministic signals (JSON):\n" + json.dumps(signals, ensure_ascii=False))
    
    cand_json = llm_json(GENERATOR_SYS, gen_payload, model_name = gen_model, max_new_tokens = GEN_MAX_TOKENS)
    candidate = CandidateReply(**cand_json)

    # 2) Make a joint_payload that contains the user input, the assistant reply, and some deterministic signals from the lexicon analysis. This is the content that the LLM judges will evaluate.
    joint_payload = (f"USER:\n{user_text}\n\nASSISTANT:\n{candidate.message}\n\n"
      "Deterministic signals (JSON):\n" + json.dumps(signals, ensure_ascii=False) +
      "\n\nInstruction to judge: Treat deterministic signals as hints only; "
      "rely on the actual text. If you disagree, reflect that in rationale_spans.")

    # Single call to get all 3 verdicts (results returned in a dict)
    raw_combined = llm_json(COMBINED_JUDGE_SYS, joint_payload, model_name = judge_model, max_new_tokens = JUDGE_MAX_TOKENS)

    # DEBUG: Print the keys specifically to see what the model gave you
    print(f"DEBUG - LLM Output Keys: {raw_combined.keys()}") 
    print(f"DEBUG - Full Output: {raw_combined}")

    # Check if the output is flat (missing the 'crisis' key but having 'crisis_type')
    if "crisis" not in raw_combined and "crisis_type" in raw_combined:
        # Manually restructure flat JSON into the nested format expected by CombinedVerdict
        raw_combined = {
            "crisis": {
                "crisis_type": raw_combined.get("crisis_type"),
                "severity": raw_combined.get("severity"),
                "rationale_spans": raw_combined.get("rationale_spans", [])
            },
            "safety": {
                "passed": raw_combined.get("passed"),
                "violations": raw_combined.get("violations", []),
                "rationale": raw_combined.get("rationale", "")
            },
            "appropriateness": {
                "on_topic": raw_combined.get("on_topic"),
                "missing_elements": raw_combined.get("missing_elements", []),
                "style_issues": raw_combined.get("style_issues", [])
            }
        }

    # Parse into the container model
    verdict_container = CombinedVerdict(**raw_combined)

    # Extract the individual objects to pass to your existing gate logic
    crisis = verdict_container.crisis  # CrisisVerdict object
    safety = verdict_container.safety
    app = verdict_container.appropriateness

    # 3) Gate
    ok, meta = policy_gate(candidate, crisis, safety, app, lex)
    if not ok:
        safe_stub = (
            "I’m here to support you. I can’t provide medical or medication advice. "
            "If you’re in immediate danger or thinking about harming yourself, please seek immediate help. "
            "In the U.S., you can call or text 988 for the Suicide & Crisis Lifeline. "
            "Would you like resources or to talk through what you’re feeling right now?"
        )
        
        return {
          'approved': False, 'reason': meta, 'message': safe_stub,
          'candidate': candidate.model_dump(),
          'crisis_llm': crisis.model_dump(),
          'safety_llm': safety.model_dump(),
          'appropriateness_llm': app.model_dump(),
          'lexicon': lex
      }
        
    return {
      'approved': True, 
      'message': candidate.message,
      'candidate': candidate.model_dump(),
      'crisis_llm': crisis.model_dump(),
      'safety_llm': safety.model_dump(),
      'appropriateness_llm': app.model_dump(),
      'lexicon': lex,
      'meta': meta
  }


def should_keep(response):
    '''Keeps the generated chatbot response if it passes the safety_llm and appropriateness_llm. 
Otherwise, it generates a new response.''' 
    if not response["safety_llm"]["passed"]:
        return False
    if not response["appropriateness_llm"]["on_topic"]:
        return False

    return True

def process_with_retry(user_text: str, policy_text: str, gen_model: str,
                       judge_model: str,
                       #jsonl_path = DEFAULT_JSONL_PATH,
                       max_attempts: int = 5,
                       retry_delay: float = 0.5):
    '''
        Runs run_pipeline() repeatedly until should_keep() is True,
        or until max_attempts is reached.
        Saves the FIRST acceptable output to JSONL.
        Returns (output, keep_flag, attempts_used).
        '''
    attempt = 0
    final_out = None
    keep = False

    while attempt < max_attempts:
        attempt += 1
       # print(f"Attempt {attempt}/{max_attempts}...")

        out = run_pipeline(
            user_text = user_text.strip(),
            policy_text = policy_text.strip(),
            gen_model = gen_model,
            judge_model = judge_model
        )

        keep = should_keep(out)

        if keep:
            final_out = out
            break

        time.sleep(retry_delay)

    # save if the response passes the should_keep function
    if keep:
        record = {
            "user_text": user_text,
            "pipeline_output": final_out,
            "attempts": attempt
        }

        # with open(jsonl_path, "a", encoding="utf-8") as f:
        #     f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return final_out, keep, attempt


########## Implementing command line interface ##########

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        # description = "Run the DR mental-health chatbot pipeline"
        description = "Initiate conversation with Dr. Chatbot, a mental health support chat companion."  # Description for what the script does.
    )
    parser.add_argument(
        "--input",
        type = str,
        required = True,
        help = "User message to process"
    )
    parser.add_argument(
        "--jsonl",
        type = str,
        default = DEFAULT_JSONL_PATH,
        help = "Path to save accepted responses"
    )
    args = parser.parse_args()  # Read the actual command-line arguments and puts them into an args namespace 

    # Run pipeline with retry
    response, keep, attempts = process_with_retry(
        user_text = args.input,
        policy_text = POLICY_TEXT,
        gen_model = GEN_MODEL_ID,
        judge_model = JUDGE_MODEL_ID,
        jsonl_path = args.jsonl
    )

    print("\n=== PIPELINE RESULT ===")
    print("Approved:", keep)  # True/False
    print("Attempts:", attempts)
    print("Reply:")
    print(response["message"] if response else "No safe reply was produced.")