# -*- coding: utf-8 -*-

# This sample demonstrates handling intents from an Alexa skill using the Alexa Skills Kit SDK for Python.
# Please visit https://alexa.design/cookbook for additional examples on implementing slots, dialog management,
# session persistence, api calls, and more.
# This sample is built using the handler classes approach in skill builder.
import ast
import logging
import requests
import ask_sdk_core.utils as ask_utils

from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput

from ask_sdk_model import Response

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BASE_URL = "http://128.46.53.152:7500/"

EXPL = None
ALTPUN = None
EXIT = None
RECENT_ALT_PUN = None

class PunRequestIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("PunRequestIntent")(handler_input)
    
    def handle(self, handler_input):
        puncat = ask_utils.request_util.get_slot(handler_input, "puncategory").value
        URL = BASE_URL + "getapun"
        PARAMS = {'puncat': puncat}
        r = requests.get(url = URL, params = PARAMS)    # Sending get request and saving the response as response object
        data = ast.literal_eval(r.text)
        
        sentence = data[0]
        max_score = float(data[1])
        kword = data[2]
        global EXPL
        global ALTPUN
        global EXIT
        global RECENT_ALT_PUN
        
        if max_score >= 0.85:
            speak_output = sentence + "   do you like to hear the pun explanation?"
            EXPL = 1
            ALTPUN = 0
            EXIT = 0
        
        else:
            speak_output = "Sorry, I could not find a pun from the category you mentioned. The closest category I found is " + kword + ". Do you like to hear a pun on that?"
            EXPL = 0
            ALTPUN = 1
            EXIT = 0
            RECENT_ALT_PUN = sentence
        
        return handler_input.response_builder.speak(speak_output).ask(speak_output).response
        
# Yes Intent Handler
class YesIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.YesIntent")(handler_input)
    
    def handle(self, handler_input):
        global EXPL
        global ALTPUN
        global EXIT
        
        if EXPL == 1 and ALTPUN == 0 and EXIT == 0:         # If 'yes' is for an explanation
            URL = BASE_URL + "getpunexp"
            PARAMS = {}
            r = requests.get(url = URL, params = PARAMS)    # Sending get request and saving the response as response object
            data = r.text
            speak_output = data
            EXPL = 0
            ALTPUN = 0
            EXIT = 1
            return handler_input.response_builder.speak(speak_output).ask(speak_output).response
            
        if EXPL == 0 and ALTPUN == 1 and EXIT == 0:         # If 'yes' is for getting the alternative pun
            speak_output = RECENT_ALT_PUN + "   do you like to hear the pun explanation?"
            EXPL = 1
            ALTPUN = 0
            EXIT = 0
            return handler_input.response_builder.speak(speak_output).ask(speak_output).response
            
        if EXPL == 0 and ALTPUN == 0 and EXIT == 1:         # If 'yes' is for exiting the system
            speak_output = "Exiting the system. Have a nice day."
            return handler_input.response_builder.speak(speak_output).response
        
# No Intent Handler
class NoIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.NoIntent")(handler_input)
    
    def handle(self, handler_input):
        global EXPL
        global ALTPUN
        global EXIT
        
        if EXPL == 0 and ALTPUN == 0 and EXIT == 1:                                                     # If 'no' is for exiting the system
            speak_output = "Suggest a category to get started."

        if (EXPL == 0 and ALTPUN == 1 and EXIT == 0) or (EXPL == 1 and ALTPUN == 0 and EXIT == 0):     # If 'no' is for getting the alternative pun or explanation
            speak_output = "Do you want to exit?"
            EXPL = 0
            ALTPUN = 0
            EXIT = 1
            
        return handler_input.response_builder.speak(speak_output).ask(speak_output).response
    
class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Welcome to the pun recommendation system. What kind of puns you like to hear?"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


class HelloWorldIntentHandler(AbstractRequestHandler):
    """Handler for Hello World Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("HelloWorldIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Hello World!"

        return (
            handler_input.response_builder
                .speak(speak_output)
                # .ask("add a reprompt if you want to keep the session open for the user to respond")
                .response
        )


class HelpIntentHandler(AbstractRequestHandler):
    """Handler for Help Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "You can say hello to me! How can I help?"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Single handler for Cancel and Stop Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Goodbye!"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .response
        )

class FallbackIntentHandler(AbstractRequestHandler):
    """Single handler for Fallback Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        logger.info("In FallbackIntentHandler")
        speech = "Hmm, I'm not sure. You can say Hello or Help. What would you like to do?"
        reprompt = "I didn't catch that. What can I help you with?"

        return handler_input.response_builder.speak(speech).ask(reprompt).response

class SessionEndedRequestHandler(AbstractRequestHandler):
    """Handler for Session End."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response

        # Any cleanup logic goes here.

        return handler_input.response_builder.response


class IntentReflectorHandler(AbstractRequestHandler):
    """The intent reflector is used for interaction model testing and debugging.
    It will simply repeat the intent the user said. You can create custom handlers
    for your intents by defining them above, then also adding them to the request
    handler chain below.
    """
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("IntentRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        intent_name = ask_utils.get_intent_name(handler_input)
        speak_output = "You just triggered " + intent_name + "."

        return (
            handler_input.response_builder
                .speak(speak_output)
                # .ask("add a reprompt if you want to keep the session open for the user to respond")
                .response
        )


class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors. If you receive an error
    stating the request handler chain is not found, you have not implemented a handler for
    the intent being invoked or included it in the skill builder below.
    """
    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        logger.error(exception, exc_info=True)

        speak_output = "Sorry, I had trouble doing what you asked. Please try again."

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

# The SkillBuilder object acts as the entry point for your skill, routing all request and response
# payloads to the handlers above. Make sure any new handlers or interceptors you've
# defined are included below. The order matters - they're processed top to bottom.


sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(HelloWorldIntentHandler())
sb.add_request_handler(PunRequestIntentHandler())
sb.add_request_handler(YesIntentHandler())
sb.add_request_handler(NoIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_request_handler(IntentReflectorHandler()) # make sure IntentReflectorHandler is last so it doesn't override your custom intent handlers

sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()