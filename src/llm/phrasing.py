"""
Saying things back to the patient, in the language they wrote in.

The rule engine decides what to ask. This module only decides how to word it,
and the separation matters: a question's *meaning* comes from
`data/clinical/questions.json` and is fixed, reviewable, and identical for every
patient. Translation is presentation applied on top of it. A model is never
asked what to ask.

Three layers, in order of preference:

* **Canned text.** Fixed strings for English, Malayalam and Hindi, written into
  this file. No call, no latency, and they work with no key at all.
* **Cached translation.** For a language with no canned text, Gemini translates
  once and the result is kept for the life of the process. The first Malayalam
  patient of the day pays for a question; nobody after them does.
* **English.** If translation fails, the English wording is used rather than
  nothing. A question the patient may have to puzzle over still beats a
  conversation that stops.

The closing message is deliberately careful. It reports an urgency and a
department and says a clinician will review. It never names a condition, never
reassures, and never tells a patient they are fine - VITA does not know that,
and a triage assistant that offers comfort it cannot support is doing the one
thing this system is built not to do.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.knowledge import Question
from ..core.schema import Urgency
from .gemini import GeminiClient

logger = logging.getLogger(__name__)

#: Languages with hand-written text. Everything else is translated on demand.
CANNED_LANGUAGES = {"en", "ml", "hi"}

_MESSAGES: dict[str, dict[str, str]] = {
    "empty_message": {
        "en": "I did not catch that. Could you tell me what is troubling you?",
        "ml": "എനിക്ക് അത് മനസ്സിലായില്ല. നിങ്ങൾക്ക് എന്താണ് ബുദ്ധിമുട്ട് എന്ന് പറയാമോ?",
        "hi": "मुझे वह समझ नहीं आया। कृपया बताइए आपको क्या तकलीफ़ है?",
    },
    "opening": {
        "en": "Tell me what is troubling you today, in your own words.",
        "ml": "ഇന്ന് നിങ്ങൾക്ക് എന്താണ് ബുദ്ധിമുട്ട് എന്ന് നിങ്ങളുടെ വാക്കുകളിൽ പറയൂ.",
        "hi": "आज आपको क्या तकलीफ़ है, अपने शब्दों में बताइए।",
    },
    "review_required": {
        "en": "A clinician will review your case before you are seen.",
        "ml": "നിങ്ങളെ കാണുന്നതിന് മുൻപ് ഒരു ഡോക്ടർ നിങ്ങളുടെ കേസ് പരിശോധിക്കും.",
        "hi": "आपको देखने से पहले एक चिकित्सक आपके मामले की समीक्षा करेंगे।",
    },
    "seek_immediate": {
        "en": "This meets our criteria for immediate attention. Please tell a member of staff now.",
        "ml": "ഇത് അടിയന്തര ശ്രദ്ധ ആവശ്യമുള്ള അവസ്ഥയാണ്. ദയവായി ഇപ്പോൾ തന്നെ ഒരു ജീവനക്കാരനോട് പറയുക.",
        "hi": "इसमें तत्काल ध्यान देने की आवश्यकता है। कृपया अभी किसी कर्मचारी को बताएं।",
    },
    "out_of_scope": {
        "en": (
            "What you have described is outside what I am able to assess. "
            "I have passed your case to a clinician, who will see you directly."
        ),
        "ml": (
            "നിങ്ങൾ പറഞ്ഞ കാര്യം എനിക്ക് വിലയിരുത്താൻ കഴിയുന്നതിന് പുറത്താണ്. "
            "നിങ്ങളുടെ കേസ് ഒരു ഡോക്ടർക്ക് കൈമാറിയിട്ടുണ്ട്."
        ),
        "hi": (
            "आपने जो बताया है वह मेरे आकलन के दायरे से बाहर है। "
            "मैंने आपका मामला एक चिकित्सक को भेज दिया है।"
        ),
    },
    "routed": {
        "en": "I have recorded your details and routed you to {department}.",
        "ml": "നിങ്ങളുടെ വിവരങ്ങൾ രേഖപ്പെടുത്തി {department} വിഭാഗത്തിലേക്ക് അയച്ചിട്ടുണ്ട്.",
        "hi": "मैंने आपका विवरण दर्ज कर लिया है और आपको {department} भेज दिया है।",
    },
}

_TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {
            "type": "string",
            "description": "The sentence, translated, with its meaning unchanged.",
        }
    },
}

_TRANSLATION_INSTRUCTION = (
    "You translate questions asked by a hospital intake system. Preserve the "
    "clinical meaning exactly. Do not soften a question, do not add reassurance, "
    "do not add or remove any detail, and do not answer it. Return only the "
    "translated sentence."
)


class Phraser:
    """Produces patient-facing text, translating only when it has to."""

    def __init__(self, llm: GeminiClient) -> None:
        self.llm = llm
        self._cache: dict[tuple[str, str], str] = {}

    # -- canned ----------------------------------------------------------

    def say(self, key: str, language: str, **fields: Any) -> str:
        template = _MESSAGES.get(key, {})
        text = template.get(language) or template.get("en", "")
        return text.format(**fields) if fields else text

    # -- questions -------------------------------------------------------

    def question(self, question: Question, language: str) -> str:
        """The wording for one follow-up question, in the patient's language."""
        if not question:
            return ""
        if language == "en" or not language:
            return question.text

        cached = self._cache.get((question.fact, language))
        if cached:
            return cached

        translated = self._translate(question.text, language)
        if translated:
            self._cache[(question.fact, language)] = translated
            return translated

        # Falling back to English keeps the conversation moving. The meaning is
        # unchanged; only the convenience is lost.
        logger.info("no translation for %s into %s; using English", question.fact, language)
        return question.text

    def _translate(self, text: str, language: str) -> str:
        if not self.llm.available:
            return ""
        outcome = self.llm.generate_json(
            f"Translate this into the language with BCP-47 code '{language}':\n\n{text}",
            _TRANSLATION_SCHEMA,
            system_instruction=_TRANSLATION_INSTRUCTION,
        )
        if not outcome.ok or not isinstance(outcome.data, dict):
            return ""
        return str(outcome.data.get("translation", "")).strip()

    # -- outcome ---------------------------------------------------------

    def outcome(
        self,
        urgency: Urgency,
        department: str,
        requires_review: bool,
        language: str,
        *,
        out_of_scope: bool = False,
    ) -> str:
        """The closing message.

        Reports what was decided and what happens next. Never names a condition,
        never reassures, and never tells the patient they are fine.
        """
        if out_of_scope:
            return self.say("out_of_scope", language)

        parts = [self.say("routed", language, department=department)]

        if urgency in (Urgency.HIGH, Urgency.CRITICAL):
            parts.append(self.say("seek_immediate", language))
        if requires_review:
            parts.append(self.say("review_required", language))

        return " ".join(p for p in parts if p)
