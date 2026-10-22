# CivicLens Data Documentation & Annotation Guidelines

## Overview
CivicLens utilizes three curated datasets designed to benchmark complaint classification, semantic duplicate detection, and linguistic robustness across English and code-mixed Indian languages.

### 1. Dataset Files & Schema

#### `sample_complaints.csv`
Primary corpus used for training and evaluating complaint classification models.
- **`complaint_id`**: Unique alphanumeric identifier (e.g., `CMP-001`).
- **`text`**: The raw complaint text submitted by the user.
- **`category`**: Ground-truth service category (one of 6 predefined domains):
  1. `Water supply`: Contaminated water, low pressure, pipeline leaks, no supply.
  2. `Electricity`: Blackouts, transformer breakdown, high voltage fluctuations, fallen wires.
  3. `Sanitation`: Garbage piles, open drains, delayed trash pickup, unhygienic public restrooms.
  4. `Roads`: Potholes, damaged footpaths, non-functional streetlights, traffic signal damage.
  5. `Transport`: Public bus delays, rude crew behavior, route cancellations, overcrowding.
  6. `Campus facilities`: Broken hostel elevator, non-working AC in library, classroom projector issues.
- **`language_tag`**: Linguistic variation tag:
  - `en_formal`: Standard grammatical English.
  - `en_informal`: Informal English with colloquialisms, typos, abbreviations.
  - `code_mixed_hinglish`: Hindi-English code-mixed in Latin script.
  - `code_mixed_tanglish`: Tamil-English code-mixed in Latin script.
- **`incident_group_id`**: Incident cluster ID. Rephrased reports of the exact same event share this ID to enable group-based splitting and prevent data leakage between train and test splits.

#### `complaint_pairs.csv`
Pairwise labeled dataset for evaluating duplicate detection systems.
- **`pair_id`**: Unique identifier for the pair.
- **`complaint_1`**: Text of the first complaint.
- **`complaint_2`**: Text of the second complaint.
- **`is_duplicate`**: Binary indicator (1 = duplicate/same underlying issue, 0 = distinct issue).
- **`pair_type`**:
  - `exact_duplicate`: Near-identical wording.
  - `lexical_paraphrase`: Same incident described with synonymous vocabulary in English.
  - `semantic_paraphrase`: Rephrased using different sentence structure and perspective.
  - `code_mixed_pair`: One English complaint and one code-mixed counterpart describing the same incident.
  - `hard_negative_same_category`: Complaints in the same domain but describing distinct incidents (e.g., two different streetlights on different avenues).
  - `cross_category_negative`: Distinct complaints across different service categories.

#### `code_mixed_eval.csv`
Evaluation set containing parallel complaint pairs:
- **`eval_id`**: Alphanumeric ID.
- **`standard_english`**: Standard English reference complaint.
- **`code_mixed_variant`**: Code-mixed rephrasing of the same incident.
- **`category`**: Service category.
- **`language`**: Underlying code-mixed language (`Hinglish`, `Tanglish`, etc.).

---

## Ethical & Privacy Safeguards
- **Data Anonymization**: All names, personal phone numbers, vehicle license plates, and private home addresses are synthetic or masked.
- **Fair Representation**: Categories are balanced across municipal and campus public utilities.
- **Code-Mixing Attribution**: Reference patterns and colloquial lexical items drawn from standard Indian social media linguistic patterns (respecting DravidianCodeMix reference styles without misrepresenting unassociated comment datasets).
