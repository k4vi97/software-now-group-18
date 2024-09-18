import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import Counter

# Function to extract entities (diseases, drugs) using SciSpacy models
def extract_entities_sci(file_path, spacy_model, entity_types=['DISEASE', 'CHEMICAL']):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    doc = spacy_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in entity_types]
    return entities

# Function to extract entities (diseases, drugs) using BioBERT
def extract_entities_biobert(file_path, biobert_pipeline):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    entities = biobert_pipeline(text)
    # BioBERT typically uses "B-DISEASE", "I-DISEASE", "B-DRUG", "I-DRUG" labels, so we filter by those
    filtered_entities = [(entity['word'], entity['entity']) for entity in entities if 'DISEASE' in entity['entity'] or 'DRUG' in entity['entity']]
    return filtered_entities

# Function to compare entities between SciSpacy and BioBERT
def compare_entities(entities_sci, entities_biobert):
    # Normalize entities to lower case for comparison
    sci_entities_set = set([(ent[0].lower(), ent[1]) for ent in entities_sci])
    biobert_entities_set = set([(ent[0].lower(), ent[1]) for ent in entities_biobert])

    common_entities = sci_entities_set.intersection(biobert_entities_set)
    sci_only_entities = sci_entities_set - biobert_entities_set
    biobert_only_entities = biobert_entities_set - sci_entities_set

    print(f"Total common entities: {len(common_entities)}")
    print(f"Entities only in SciSpacy: {len(sci_only_entities)}")
    print(f"Entities only in BioBERT: {len(biobert_only_entities)}")

    return common_entities, sci_only_entities, biobert_only_entities

# Load SciSpacy models
nlp_sci_sm = spacy.load('en_core_sci_sm')  # Small SciSpacy model
nlp_sci_ner = spacy.load('en_ner_bc5cdr_md')  # SciSpacy model for disease and chemical NER

# Load BioBERT pipeline using Hugging Face
biobert_model = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(biobert_model)
model = AutoModelForTokenClassification.from_pretrained(biobert_model)
nlp_biobert = pipeline("ner", model=model, tokenizer=tokenizer)

# File path to the combined text
file_path = 'combined_texts.txt'

# Task 4.1: Extract entities using SciSpacy small model
entities_sci_sm = extract_entities_sci(file_path, nlp_sci_sm)

# Task 4.2: Extract entities using SciSpacy NER model
entities_sci_ner = extract_entities_sci(file_path, nlp_sci_ner)

# Task 4.3: Extract entities using BioBERT model
entities_biobert = extract_entities_biobert(file_path, nlp_biobert)

# Output the extracted entities for both models
print("SciSpacy small model (en_core_sci_sm) entities:", entities_sci_sm)
print("SciSpacy NER model (en_ner_bc5cdr_md) entities:", entities_sci_ner)
print("BioBERT model entities:", entities_biobert)

# Compare the extracted entities between SciSpacy NER and BioBERT
common_entities, sci_only_entities, biobert_only_entities = compare_entities(entities_sci_ner, entities_biobert)

# Print comparison details
print(f"Common entities between SciSpacy and BioBERT: {common_entities}")
print(f"Entities only found by SciSpacy: {sci_only_entities}")
print(f"Entities only found by BioBERT: {biobert_only_entities}")