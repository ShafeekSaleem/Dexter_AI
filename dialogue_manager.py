import os
from sklearn.metrics.pairwise import pairwise_distances_argmin
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot import ChatBot
from utils import *
import random

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)
        best_thread = pairwise_distances_argmin(question_vec.reshape((1,-1)),thread_embeddings)

        return thread_ids[best_thread[0]]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

        self.chatbot = self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals 
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param
        
        # Create a new ChatBot with name Dexter.to
        chatbot = ChatBot(
            'Dexter.to',
            storage_adapter='chatterbot.storage.SQLStorageAdapter',
            logic_adapters=[
                'chatterbot.logic.MathematicalEvaluation',
                'chatterbot.logic.BestMatch'
            ],
            database_uri='sqlite:///database.sqlite3'
        )

        # Create a new trainer for the chatbot
        trainer = ChatterBotCorpusTrainer(chatbot)

        # Train the chatbot based on the english corpus
        trainer.train("chatterbot.corpus.english")
        return chatbot
        
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = text_prepare(question, remove_stop_words=True)
        features = self.tfidf_vectorizer.transform(np.array([prepared_question],dtype=object))
        intent = str(self.intent_recognizer.predict(features)[0])

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chatbot.get_response(question)
            return response

        # Say name
        if intent == 'say_name':
            response = random.choice(say_name_resp)
            return response

        # ask age
        if intent == 'ask_age':
            response = random.choice(ask_age_resp)
            return response

        # bot challenge
        if intent == 'bot_challenge':
            response = random.choice(bot_challenge_resp)
            return response

        # who_built
        if intent == 'ask_builder':
            response = random.choice(who_built_resp)
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = str(self.tag_classifier.predict(features)[0])
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)
           
            return self.ANSWER_TEMPLATE % (tag, thread_id)

