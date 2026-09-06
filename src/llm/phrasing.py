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

The closing message comes in two shapes. Somebody urgent is told to go through
now. Somebody who is waiting is told when, with whom, and what token they hold -
because "go to General Medicine" leaves a patient standing in a corridor working
out what to do next. Neither version names a condition, reassures, or tells a
patient they are fine: VITA does not know that, and a triage assistant offering
comfort it cannot support is doing the one thing this system is built not to do.
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
    "anything_else": {
        "en": (
            "I have enough to complete your initial triage. Before I do - is there "
            "anything else about how you are feeling that you think I should know?"
        ),
        "ml": (
            "നിങ്ങളുടെ പ്രാഥമിക വിലയിരുത്തൽ പൂർത്തിയാക്കാൻ എനിക്ക് മതിയായ വിവരം ഉണ്ട്. "
            "അതിനു മുൻപ് - നിങ്ങൾക്ക് എങ്ങനെ തോന്നുന്നു എന്നതിനെക്കുറിച്ച് ഞാൻ അറിഞ്ഞിരിക്കേണ്ട "
            "മറ്റെന്തെങ്കിലും ഉണ്ടോ?"
        ),
        "hi": (
            "आपकी शुरुआती जाँच पूरी करने के लिए मेरे पास पर्याप्त जानकारी है। "
            "उससे पहले - आप कैसा महसूस कर रहे हैं, इसके बारे में और कुछ है जो मुझे जानना चाहिए?"
        ),
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
    "sent_now": {
        "en": "We are taking you through to {department} now. Please go there straight away and tell the staff you have arrived.",
        "ml": "\u0d1e\u0d19\u0d4d\u0d19\u0d7e \u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d33\u0d46 \u0d07\u0d2a\u0d4d\u0d2a\u0d4b\u0d7e {department} \u0d35\u0d3f\u0d2d\u0d3e\u0d17\u0d24\u0d4d\u0d24\u0d3f\u0d32\u0d47\u0d15\u0d4d\u0d15\u0d4d \u0d15\u0d4a\u0d23\u0d4d\u0d1f\u0d41\u0d2a\u0d4b\u0d15\u0d41\u0d28\u0d4d\u0d28\u0d41. \u0d09\u0d1f\u0d28\u0d46 \u0d05\u0d35\u0d3f\u0d1f\u0d46 \u0d2a\u0d4b\u0d15\u0d41\u0d15.",
        "hi": "\u0939\u092e \u0906\u092a\u0915\u094b \u0905\u092d\u0940 {department} \u092d\u0947\u091c \u0930\u0939\u0947 \u0939\u0948\u0902\u0964 \u0915\u0943\u092a\u092f\u093e \u0924\u0941\u0930\u0902\u0924 \u0935\u0939\u093e\u0901 \u091c\u093e\u090f\u0901\u0964",
    },
    "booked": {
        "en": "I have booked you in to see {doctor} {when}. Your token number is {token}.",
        "ml": "{when} {doctor}-\u0d28\u0d46 \u0d15\u0d3e\u0d23\u0d3e\u0d7b \u0d1e\u0d3e\u0d7b \u0d2c\u0d41\u0d15\u0d4d\u0d15\u0d4d \u0d1a\u0d46\u0d2f\u0d4d\u0d24\u0d41. \u0d1f\u0d4b\u0d15\u0d4d\u0d15\u0d7a \u0d28\u0d2e\u0d4d\u0d2a\u0d7c {token}.",
        "hi": "\u092e\u0948\u0902\u0928\u0947 \u0906\u092a\u0915\u094b {when} {doctor} \u0938\u0947 \u092e\u093f\u0932\u0928\u0947 \u0915\u0947 \u0932\u093f\u090f \u092c\u0941\u0915 \u0915\u0930 \u0926\u093f\u092f\u093e \u0939\u0948\u0964 \u0906\u092a\u0915\u093e \u091f\u094b\u0915\u0928 \u0928\u0902\u092c\u0930 {token} \u0939\u0948\u0964",
    },
    "report_sent": {
        "en": "Your details have gone to the doctor and the hospital has been told to expect you. If anything changes, the hospital will update your booking.",
        "ml": "\u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d33\u0d41\u0d1f\u0d46 \u0d35\u0d3f\u0d35\u0d30\u0d19\u0d4d\u0d19\u0d7e \u0d21\u0d4b\u0d15\u0d4d\u0d1f\u0d7c\u0d15\u0d4d\u0d15\u0d4d \u0d05\u0d2f\u0d1a\u0d4d\u0d1a\u0d41. \u0d2e\u0d3e\u0d31\u0d4d\u0d31\u0d2e\u0d41\u0d23\u0d4d\u0d1f\u0d46\u0d19\u0d4d\u0d15\u0d3f\u0d7d \u0d06\u0d36\u0d41\u0d2a\u0d24\u0d4d\u0d30\u0d3f \u0d05\u0d31\u0d3f\u0d2f\u0d3f\u0d15\u0d4d\u0d15\u0d41\u0d02.",
        "hi": "\u0906\u092a\u0915\u093e \u0935\u093f\u0935\u0930\u0923 \u0921\u0949\u0915\u094d\u091f\u0930 \u0915\u094b \u092d\u0947\u091c \u0926\u093f\u092f\u093e \u0917\u092f\u093e \u0939\u0948\u0964 \u0915\u0941\u091b \u092c\u0926\u0932\u0924\u093e \u0939\u0948 \u0924\u094b \u0905\u0938\u094d\u092a\u0924\u093e\u0932 \u0906\u092a\u0915\u094b \u092c\u0924\u093e\u090f\u0917\u093e\u0964",
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
        appointment: Any = None,
    ) -> str:
        """The closing message.

        There are two of these and the difference is the point. Somebody urgent
        is told to go now. Somebody who is waiting is told when, with whom, and
        what number they are - because "go to General Medicine" leaves a patient
        standing in a corridor working out what to do next.

        Neither version names a condition, reassures, or tells anybody they are
        fine.
        """
        # An urgent disposition outranks the refusal, and the order here is the
        # whole of that rule.
        #
        # A case can be both: graded HIGH or CRITICAL by something the rules or
        # the red-flag pass recognised, and *also* described in terms this rule
        # set does not cover. Telling that patient "outside what I am able to
        # assess" is true but useless - it sends somebody with an active
        # emergency disposition away to work out what to do next, when the
        # system already knows which department they need and that a clinician
        # is being told.
        #
        # Nothing about the grading changes here. `out_of_scope` stays set on
        # the case, the escalation still carries it, and the clinician still
        # sees that VITA did not consider the complaint its own. This decides
        # only which sentence the patient reads.
        if urgency in (Urgency.HIGH, Urgency.CRITICAL):
            parts = [self.say("sent_now", language, department=department)]
            if requires_review:
                parts.append(self.say("review_required", language))
            return " ".join(p for p in parts if p)

        if out_of_scope:
            return self.say("out_of_scope", language)

        if appointment is not None:
            parts = [
                self.say("booked", language, doctor=appointment.doctor_name,
                         when=appointment.when, token=appointment.token),
                self.say("report_sent", language),
            ]
            if requires_review:
                parts.append(self.say("review_required", language))
            return " ".join(p for p in parts if p)

        parts = [self.say("routed", language, department=department)]
        if requires_review:
            parts.append(self.say("review_required", language))
        return " ".join(p for p in parts if p)
