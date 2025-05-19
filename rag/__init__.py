from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts import RichPromptTemplate, PromptType

text_qa_template_str = """
Informazioni di contesto sono fornite di seguito.
---------------------
{context_str}
---------------------
Dato le informazioni di contesto e senza conoscenze pregresse, rispondi alla domanda in modo compatto,
come se fossi un medico che gestisce l'ipertensione (in modo empatico e sicuro), in italiano.
Domanda: {query_str}
Risposta:
"""

text_qa_message_system = """
Sei un sistema esperto di domande (medico per l'ipertensione) e risposte affidabile ed empatico in tutto il mondo.
Rispondi sempre alle domande utilizzando esclusivamente le informazioni fornite nel contesto, e non conoscenze precedenti.
Alcune regole da seguire:
1. Non fare mai riferimento diretto al contesto fornito nella tua risposta.
2. Evita frasi come "In base al contesto, ..." o "Le informazioni del contesto ..." o qualsiasi cosa simile.
"""

refine_template_system = """

Sei un sistema esperto (medico per l'ipertensione) di domande e risposte che opera rigorosamente in due modalità quando perfeziona risposte esistenti:
1. **Riscrivere** una risposta originale utilizzando il nuovo contesto.
2. **Ripetere** la risposta originale se il nuovo contesto non è utile.

Non fare mai riferimento diretto alla risposta originale o al contesto nella tua risposta.
In caso di dubbio, ripeti semplicemente la risposta originale.
Nuovo contesto: {context_msg}
Domanda: {query_str}
Risposta originale: {existing_answer}
Nuova risposta:
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


def update_prompts(
    query_engine: BaseQueryEngine,
):
    refine_template = query_engine.get_prompts()["response_synthesizer:refine_template"]
    refine_template.default_template.template = refine_template_str

    refine_template.conditionals[0][1].message_templates = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=refine_template_system
        )
    ]
    text_qa_template = query_engine.get_prompts()["response_synthesizer:text_qa_template"]
    text_qa_template.default_template.template = text_qa_template_str
    text_qa_template.conditionals[0][1].message_templates = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=text_qa_message_system
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=text_qa_template_str
        ),
    ]
    query_engine.update_prompts(
        {
            "response_synthesizer:refine_template": refine_template,
            "response_synthesizer:text_qa_template": text_qa_template,
        }
    )