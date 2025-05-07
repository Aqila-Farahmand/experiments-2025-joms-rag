from llama_index.core.prompts import RichPromptTemplate, PromptType

text_qa_template_str = """
Informazioni di contesto sono fornite di seguito.
---------------------
{context_str}
---------------------
Dato le informazioni di contesto e senza conoscenze pregresse, rispondi alla domanda,
come se fossi un medico (in modo empatico e sicuro), in italiano.
Domanda: {query_str}
Risposta:
"""


refine_template_str = """
La domanda originale è la seguente: {query_str}
Abbiamo fornito una risposta esistente: {existing_answer}
Abbiamo l'opportunità di perfezionare la risposta esistente (solo se necessario) con un ulteriore contesto qui sotto.
------------
{context_msg}
------------
Dato il nuovo contesto, 
perfeziona la risposta originale per rispondere meglio alla domanda. Se il contesto non è utile, 
restituisci la risposta originale.
Risposta Perfezionata:
"""
