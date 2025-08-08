import logging
import os
import pickle
from generations.cache import PATH as GENERATIONS_CACHE_PATH
from evaluations.cache import PATH as EVAL_CACHE_PATH
from evaluations.syntax import PATH as SYNTAX_PATH

from evaluations.readability import PATH as READABILITY_PATH
import json
import csv
from italian_ats_evaluator import TextAnalyzer
from llama_index.llms.google_genai import GoogleGenAI
analyzer = TextAnalyzer()
def load_pickle_in_folder(folder):
    """
    Load all the pickle files in a folder.
    """

    # get all the files in the folder
    files = os.listdir(folder)
    # filter for only the pickle files
    pickle_files = [f for f in files if f.endswith(".pkl")]
    # load all the pickle files
    data = {}
    for file in pickle_files:
        with open(os.path.join(folder, file), "rb") as f:
            data[file] = pickle.load(f)
    return data

prompt = """
# Linee Guida per Analisi di Leggibilità

Agisci come un esperto analista linguistico specializzato nella lingua italiana. Il tuo compito è valutare la leggibilità del testo fornito e restituire la tua analisi in un formato JSON specifico.

## Criteri di Analisi

Analizza il testo basandoti su metriche standard di leggibilità per l'italiano, considerando i seguenti fattori:

- **Lunghezza delle frasi**: Frasi più brevi sono generalmente più facili da comprendere
- **Complessità lessicale**: Frequenza di parole comuni vs. parole tecniche o rare
- **Struttura sintattica**: Costruzioni sintattiche semplici vs. complesse (subordinate, periodi articolati)
- **Complessità morfologica**: Presenza di forme verbali complesse, costruzioni passive
- **Registro linguistico**: Linguaggio colloquiale, formale o specialistico

## Formato di Output

Il tuo output deve essere un singolo oggetto JSON contenente esclusivamente la seguente chiave:

**`readability_level`**: Un numero intero su una scala da 0 a 100, dove un punteggio più alto indica una leggibilità maggiore.

### Scala di Valutazione:

- **90-100**: **Molto facile da leggere**
  - Comprensibile da bambini di 10-11 anni
  - Frasi brevi e semplici
  - Vocabolario di base
  - Strutture sintattiche elementari

- **80-89**: **Facile**
  - Adatto a studenti di scuola media (11-14 anni)
  - Linguaggio chiaro e diretto
  - Poche subordinate

- **70-79**: **Abbastanza facile**
  - Comprensibile da studenti di scuola superiore (14-18 anni)
  - Linguaggio standard con occasionali termini specifici

- **60-69**: **Standard**
  - Livello di italiano medio-alto
  - Adatto ad adulti con educazione secondaria completa

- **50-59**: **Abbastanza difficile**
  - Richiede educazione superiore
  - Linguaggio formale o tecnico

- **30-49**: **Difficile**
  - Comprensibile principalmente da laureati
  - Terminologia specialistica
  - Strutture sintattiche complesse

- **0-29**: **Molto difficile**
  - Livello accademico o altamente specialistico
  - Linguaggio tecnico-scientifico
  - Sintassi articolata e periodi lunghi

## Istruzioni per l'Output

- **NON** includere spiegazioni, commenti o formattazione markdown nella risposta
- Restituisci **SOLO** l'oggetto JSON
- Il JSON deve essere valido e pronto per il parsing automatico

## Esempio di Output Corretto

```json
{
  "readability_level": 75
}
```

---

**Testo da Analizzare:**
[Inserire qui il testo da valutare]

"""

to_eval = load_pickle_in_folder(GENERATIONS_CACHE_PATH)
judge_llama_index = GoogleGenAI(model="gemini-2.5-flash")
for key, value in to_eval.items():
    key = key.replace(".pkl", "")
    logging.info(f"Evaluating {key}")
    #evaluations = []
    #for current_value in value["responses"]:
    #    evaluations.append(analyzer.analyze(current_value["response"].__str__()))
#
    ## Prepare data for CSV: one row per evaluation
    #csv_data = [
    #    [
    #        key,
    #        eval.readability_evaluation.ttr,
    #        eval.readability_evaluation.gulpease,
    #        eval.readability_evaluation.flesch_vacca,
    #        eval.readability_evaluation.lexical_density
    #    ]
    #    for eval in evaluations
    #]
    #os.makedirs(SYNTAX_PATH, exist_ok=True)
    #csv_path = os.path.join(SYNTAX_PATH, f"{key}.csv")
    #with open(csv_path, "w", newline="") as csvfile:
    #    writer = csv.writer(csvfile)
    #    writer.writerow(["key", "ttr", "gulpease", "flesch_vacca", "lexical_density"])
    #    writer.writerows(csv_data)

    scores = []
    for current_value in value["responses"]:
        text_to_analyze = current_value["response"].__str__()
        response = judge_llama_index.complete(prompt + text_to_analyze)
        print(f"Response: {response}")

        try:
            readability_data = str(response).strip()
            readability_data = readability_data.replace("```json", "").replace("```", "").strip()
            readability_json = json.loads(readability_data)
            readability_level = readability_json.get("readability_level", None)
            if readability_level is not None:
                scores.append(readability_level)
                print(f"Readability Level: {readability_level}")
            else:
                print("No readability level found in the response.")
        except Exception as e:
            print(f"Error parsing response: {e}")
            readability_level = None
    # Save the scores to a CSV file
    os.makedirs(READABILITY_PATH, exist_ok=True)
    csv_path = os.path.join(READABILITY_PATH, f"{key}.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["key", "readability_level"])
        for score in scores:
            writer.writerow([key, score])