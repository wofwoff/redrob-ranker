"""Curated, data-grounded vocabularies for the Redrob pool.

Every set below was derived by enumerating the *actual* distinct values in the
100K-candidate pool (see the data-coverage audit in Redrob_Solution_Design_v3.md
sec 2). The pool is a closed world: 63 companies, 48 job titles, 24 industries,
16 fields of study, 133 skill names. Because the world is closed we can classify
each value explicitly rather than guessing with fuzzy string matching -- which is
both more accurate and directly defensible at the Stage-5 interview.

Keep this module free of scoring logic; it is pure taxonomy.
"""

# --------------------------------------------------------------------------- #
# Companies (63 distinct across current + career_history)
# --------------------------------------------------------------------------- #

# The JD's explicit "worked only at consulting firms" knockdown list, plus the
# close services/BPO firms the JD's "etc." invites. Industry == IT Services /
# Consulting / AI Services is the corroborating tell for anything unlisted.
CONSULTANCY_COMPANIES = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "HCL", "Tech Mahindra", "Mindtree", "Mphasis", "Genpact AI",
}

# Real product companies whose engineers build software for their own users.
# Global product/tech giants:
_GLOBAL_PRODUCT = {
    "Google", "Meta", "Microsoft", "Amazon", "Apple", "Netflix", "LinkedIn",
    "Salesforce", "Adobe", "Uber",
}
# Indian product / consumer-internet companies:
_INDIA_PRODUCT = {
    "Swiggy", "Zomato", "Flipkart", "CRED", "Razorpay", "PhonePe", "Paytm",
    "Meesho", "Nykaa", "PolicyBazaar", "PharmEasy", "Dream11", "InMobi",
    "Glance", "Unacademy", "Vedantu", "upGrad", "BYJU'S", "Zoho", "Freshworks",
    "Ola",
}
# AI-native product companies (strong positive: exactly the "product ML" shape).
AI_PRODUCT_COMPANIES = {
    "Krutrim", "Sarvam AI", "Haptik", "Yellow.ai", "Verloop.io", "Observe.AI",
    "Saarthi.ai", "Rephrase.ai", "Wysa", "Mad Street Den", "Niramai",
    "Aganitha", "Locobuzz",
}
PRODUCT_COMPANIES = _GLOBAL_PRODUCT | _INDIA_PRODUCT | AI_PRODUCT_COMPANIES

# Placeholder / fictional employers (each reused ~31K times). They carry no
# real product-vs-services signal on their own, so they are treated as neutral
# and classified via company_size + industry instead.
GENERIC_COMPANIES = {
    "Acme Corp", "Globex Inc", "Hooli", "Initech", "Pied Piper",
    "Stark Industries", "Wayne Enterprises", "Dunder Mifflin",
}

# --------------------------------------------------------------------------- #
# Job titles (48 distinct)
# --------------------------------------------------------------------------- #

# Strong applied-ML / AI product-engineering titles -- the JD's bullseye.
TITLE_STRONG_ML = {
    "Senior AI Engineer", "AI Engineer", "Lead AI Engineer", "AI Specialist",
    "ML Engineer", "Machine Learning Engineer", "Applied ML Engineer",
    "Junior ML Engineer", "Senior ML Engineer — Search & Ranking",
    "Senior Machine Learning Engineer", "Staff Machine Learning Engineer",
    "Senior Software Engineer (ML)", "NLP Engineer", "Senior NLP Engineer",
    "Recommendation Systems Engineer", "Search Engineer",
    "Senior Applied Scientist", "Data Scientist", "Senior Data Scientist",
}

# Research-leaning: the JD explicitly knocks out *pure* research. "Applied
# Scientist" is fine (kept above); a bare "Research Engineer" is a caution.
TITLE_RESEARCH = {"AI Research Engineer"}

# Computer-vision-primary title: JD down-weights CV/speech/robotics without
# NLP/IR. Treated as a soft caution, resolved against skills/description.
TITLE_CV_SPEECH = {"Computer Vision Engineer"}

# Adjacent engineering: real software/data work, transferable but not the
# target ML profile. Partial credit; the career prose decides how close.
TITLE_ADJACENT_TECH = {
    "Software Engineer", "Senior Software Engineer", "Backend Engineer",
    "Full Stack Developer", "Data Engineer", "Senior Data Engineer",
    "Analytics Engineer", "Data Analyst", "Cloud Engineer", "DevOps Engineer",
}

# Weakly-related tech: mostly wrong specialty for this role.
TITLE_WEAK_TECH = {
    "Frontend Engineer", "Mobile Developer", "QA Engineer",
    "Java Developer", ".NET Developer",
}

# Non-technical titles: the keyword-stuffer trap surface. A "Marketing Manager
# with 9 AI skills" lives here and must be crushed by the title gate.
TITLE_NON_TECH = {
    "Business Analyst", "HR Manager", "Mechanical Engineer", "Accountant",
    "Project Manager", "Customer Support", "Operations Manager",
    "Content Writer", "Sales Executive", "Civil Engineer", "Graphic Designer",
    "Marketing Manager",
}

# --------------------------------------------------------------------------- #
# Industries (24 distinct)
# --------------------------------------------------------------------------- #

SERVICES_INDUSTRIES = {"IT Services", "Consulting", "AI Services"}
NON_TECH_INDUSTRIES = {"Manufacturing", "Paper Products", "Conglomerate"}
# Everything else (Fintech, E-commerce, SaaS, AI/ML, Conversational AI, ...) is
# a product industry; enumerated as the complement in role_match.

# --------------------------------------------------------------------------- #
# Skills (133 distinct) -> JD must-have buckets
# --------------------------------------------------------------------------- #
# The JD's absolute needs: (1) embeddings-based retrieval in production,
# (2) vector DB / hybrid search infra, (3) strong Python, (4) ranking-eval
# frameworks (NDCG/MRR/MAP). These buckets score the "must-have evidence"
# component and also flag CV/speech anti-signal.

SKILL_RETRIEVAL_EMBEDDINGS = {
    "Embeddings", "Sentence Transformers", "Semantic Search", "RAG",
    "Text Encoders", "Vector Representations", "Hugging Face Transformers",
    "Haystack", "LlamaIndex",
}
SKILL_VECTOR_DB = {
    "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus", "pgvector",
    "Elasticsearch", "OpenSearch", "Vector Search", "Search Infrastructure",
    "Search Backend", "Indexing Algorithms", "BM25",
}
SKILL_RANKING_IR = {
    "Learning to Rank", "Ranking Systems", "Recommendation Systems",
    "Information Retrieval", "Information Retrieval Systems",
    "Search & Discovery", "Content Matching",
}
SKILL_CORE_ML = {
    "Machine Learning", "Deep Learning", "NLP", "Natural Language Processing",
    "LLMs", "PyTorch", "TensorFlow", "scikit-learn", "Feature Engineering",
    "Statistical Modeling", "MLflow", "MLOps", "Fine-tuning LLMs", "PEFT",
    "LoRA", "QLoRA", "Model Adaptation", "Prompt Engineering", "LangChain",
    "Weights & Biases", "BentoML", "Kubeflow",
}
# The union of the three "must-have" buckets = the JD's non-negotiables. Used
# both for scoring and for the "N AI core skills" count in reasoning.
SKILL_MUST_HAVE = SKILL_RETRIEVAL_EMBEDDINGS | SKILL_VECTOR_DB | SKILL_RANKING_IR

# CV / speech / robotics: JD down-weights these as a *primary* specialty.
SKILL_CV_SPEECH = {
    "Computer Vision", "Object Detection", "YOLO", "OpenCV",
    "Image Classification", "CNN", "Diffusion Models", "GANs",
    "Speech Recognition", "ASR", "TTS",
}

# Non-technical skills: signal of a keyword-stuffer / wrong-domain profile when
# they dominate a supposedly-AI candidate.
SKILL_NON_TECH = {
    "Content Writing", "Marketing", "SEO", "Sales", "Accounting", "Excel",
    "SAP", "Tally", "Six Sigma", "Figma", "Photoshop", "Illustrator",
    "PowerPoint", "Salesforce CRM", "Project Management", "Scrum", "Agile",
}

PYTHON_SKILL = "Python"

# --------------------------------------------------------------------------- #
# Fields of study (16 distinct)
# --------------------------------------------------------------------------- #
FIELD_HIGHLY_RELEVANT = {
    "Artificial Intelligence", "Machine Learning", "Data Science",
    "Computer Science",
}
FIELD_RELEVANT = {
    "Computer Engineering", "Information Technology", "Statistics",
    "Mathematics", "Electronics", "Electrical Engineering", "Physics",
}
# The rest (Mechanical/Civil/Chemical Engineering, Commerce, MBA) are off-field.

# --------------------------------------------------------------------------- #
# Geography (JD: Pune/Noida hybrid; Tier-1 Indian cities welcome)
# --------------------------------------------------------------------------- #
# JD names Noida & Pune offices and welcomes Hyderabad, Pune, Mumbai, Delhi NCR.
PREFERRED_INDIA_CITIES = {
    "noida", "pune", "hyderabad", "mumbai", "delhi", "gurgaon", "bangalore",
}


def is_india(country: str) -> bool:
    return str(country or "").strip().lower() in {"india", "in"}
