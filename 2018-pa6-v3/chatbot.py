#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA6, CS124, Stanford, Winter 2018
# v.1.0.2
# Original Python code by Ignacio Cases (@cases)


# TODO STARTER:
#Ensure runs in less than 5 seconds
# Don't repeat same output (e.g. use getPositiveResponse)



######################################################################
import csv
import math
import re
import numpy as np

from movielens import ratings
import random
from random import randint
from PorterStemmer import PorterStemmer

import pdb

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    #############################################################################
    # `moviebot` is the default chatbot. Change it to your chatbot's name       #
    #############################################################################
    def __init__(self, is_turbo=False):
      ### BASIC PARAMETERS ###
      self.name = 'Seb'
      self.is_turbo = is_turbo
      
      ### CONSTANTS ###
      self.NUM_MOVIES_MIN = 5
      self.BINARY_THRESH = 2.5

      ### TOKEN SETS ###
      self.eng_articles = {'The', 'A', 'An'}
      self.negationTokens = {"nt", "not", "no", "never"}
      self.punctuation = ".,!?;:"
      self.alphanum = re.compile('[^a-zA-Z0-9]')

      ### UTILS ###
      self.p = PorterStemmer()

      ### DATA ###
      self.read_data() # self.titles, self.ratings, self.sentiment
      self.userMovies = {}
      self.userMovies = {(idx, movieDetails[0]):1 for idx, movieDetails in list(enumerate(self.titles))[:4]}
      self.returnedMovies = set()
      
      ### RESPONSE SETS ###
      self.positiveResponses = ["You liked %s. Me too!",
                                "Yeah, %s was a great movie.",
                                "I loved %s! Glad you enjoyed it too.",
                                "I'm a huge fan of %s. I'm glad you liked it.",
                                "Ooh %s is a good one.",
                                "You're absolutely right! %s was a great movie!"
                                "Right on! %s is definitely one of my favorites!!"]

      self.negativeResponses = ["I agree, %s was pretty bad.",
                                "I didn't like %s either.",
                                "Yeah %s wasn't very good.",
                                "I hear you -- I was very disappointed by %s, too.",
                                "I know what you mean. It's such a shame, I had such high hopes for %s."]
      
      self.requestAnotherResponses = ['Tell me about another movie you have seen.', 
                                      "What's another movie you remember?", 
                                      "Tell me another one.", 
                                      "Any other ones?",
                                      "Tell me about another!",
                                      "What's another movie you remember?",
                                      "Tell me about another movie!",
                                      "Any other movies?",
                                      "Are there any other movies you remember?"]
      
      

    #############################################################################
    # 1. WARM UP REPL
    #############################################################################

    def greeting(self):
      """chatbot greeting message"""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = "Hi! I'm " + self.name + "! I'm going to recommend a movie to you.\nFirst I will ask you about your taste in movies. Tell me about a movie that you have seen."

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return greeting_message

    def goodbye(self):
      """chatbot goodbye message"""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = "Thank you for hanging out with me! Stay in touch! Goodbye!"

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return goodbye_message


    #############################################################################
    # 2. Modules 2 and 3: extraction and transformation                         #
    #############################################################################

    def process(self, input):
      """Takes the input string from the REPL and call delegated functions
      that
        1) extract the relevant information and
        2) transform the information into a response to the user
      """
      #############################################################################
      # TODO: Implement the extraction and transformation in this method, possibly#
      # calling other functions. Although modular code is not graded, it is       #
      # highly recommended                                                        #
      #############################################################################
      
      if input == ":quit":
        return goodbye_message

        '''
        TO-DO:
          - What kind of film do you want to watch (options: highly-rated, recently released, genre, year) e.g. I like action moves from 2000 on?
          - Ask them about the most statistically significant movies?
          - Give them a list of top 10 e.g. ask which theyve seen and how they liked them
          - Give the real reccccc
        '''
      if self.is_turbo:
        return self.processCreative(input)
      else:
        return self.processStarter(input)
        
    
    ### Processes input for starter version ###
    def processStarter(self, input):
      ### If we already have the minimum number of movies, give them another recommendation ###
      if (len(self.userMovies) >= self.NUM_MOVIES_MIN):
        recommendation = self.recommend()
        response =  'I suggest you watch "%s".\n' % recommendation + \
                    "Would you like to hear another recommendation? (Or enter :quit if you're done.)"
        return response

      getMovies = '\"(.*?)\"'
      matches = re.findall(getMovies, input)

      ### If user tries to give more than one movie ###
      if len(matches) > 1:
        return "Please tell me about one movie at a time. Go ahead."

      ### If no movie is specified in quotation marks ###
      elif len(matches) == 0:
        return "Sorry, I don't understand. Tell me about a movie that you have seen e.g. 'I liked \"Toy Story (1995)\"'"

      else: 
        userMovie = matches[0]
        movieDetails = self.getMovieDetailsStarter(userMovie)

        ### The movie is not in the database ###
        if (not movieDetails):
          return "I'm sorry, I haven't seen that one yet! Please tell me about another movie."

        ### The user has already given us this movie ###
        if movieDetails in self.userMovies:
          return 'You mentioned that one already. Please tell me another.'

        ### Remove extraneous whitespace ###
        withoutTitle = input.replace("\"" + userMovie + "\"", '').strip()
      
        ### Must include a sentiment, e.g. "I like..." ###
        if not withoutTitle:
          return 'How did you like "%s"?' % userMovie

        ### Classify sentiment ###
        sentiment = self.classifySentiment(withoutTitle)
        if (not sentiment):
          return 'I\'m sorry, I\'m not quite sure if you liked "%s".\nTell me more about "%s".' % (userMovie, userMovie)

        idx, movie = movieDetails
        self.userMovies[movieDetails] = sentiment
        if (sentiment > 0):
          response = random.choice(self.positiveResponses) % userMovie
        else:
          response = random.choice(self.negativeResponses) % userMovie

        ### We have collected a sufficient number of movies ###
        if len(self.userMovies) >= self.NUM_MOVIES_MIN:
          recommendation = self.recommend()
          response += "\nThank you for your patience, That's enough for me to make a recommendation.\n" + \
                      'I suggest you watch "%s".\n' % recommendation + \
                      "Would you like to hear another recommendation? (Or enter :quit if you're done.)"
        else:
          response += ' ' + random.choice(self.requestAnotherResponses)

        return response  

    ### Processes input for creative version ###
    def processCreative(self, input):
      ### TO-DO ###
      return response

    def checkForMovie(self, input):
        #Takes in e.g. "Ladybird" or "Ladybird (2017)"
        #Checks database for Ladybird or Ladybird (2017)
        getMovies = '\"(.*?)\"'
        matches = re.findall(getMovies, input)
        if (not matches):
          words = input.split(' ')
          capitalized_word_idx = None
          for i, w in enumerate(words):
            #Ignore if the first letter is capitalized
            if (w.istitle()) and i > 0:
              capitalized_word_idx = i 
              break
          for i in range(capitalized_word_idx + 1, len(words) + 1):
            potential_title = " ".join(words[capitalized_word_idx:i])
            movieTitle = self.getMovieDetails(potential_title)
            if movieTitle:
              return movieTitle
        else:
          potential_title = matches[0]
          movieTitle = self.getMovieDetails(potential_title)
          if movieTitle:
            return movieTitle

    ### Returns the idx into self.titles and the movie title of 'movie' ###
    ### 'movie' must match the title in self.titles exactly.            ###
    def getMovieDetailsStarter(self, movie):
      movieWordTokens = movie.split()
      if movieWordTokens and movieWordTokens[0] in self.eng_articles: # check not empty and first word is an article
        ### Transforms movie title of the form 'An American in Paris (1951)' to 'American in Paris, An (1951)'
        movie = ' '.join(movieWordTokens[1:-1]) + ', ' + movieWordTokens[0] + ' ' + movieWordTokens[-1]

      for idx, movieDetails in enumerate(self.titles):
        title, genre = movieDetails
        if title == movie:
          return idx, title

      return None

    ### Returns an array of stemmed words ###
    def stem(self, line):
      # make sure everything is lower case
      line = line.lower()
      # insert a space before every punctuation mark
      punc_regex = '[' + self.punctuation + ']'
      line = re.sub(punc_regex, lambda x: " " + x.group(0), line)
      # split on whitespace
      line = [xx.strip() for xx in line.split()]
      # remove non alphanumeric characters contained in words, solitary punctuation is left untouched
      line = [xx if xx in self.punctuation else self.alphanum.sub('', xx) for xx in line]
      # remove any words that are now empty
      line = [xx for xx in line if xx != '']
      # stem words
      line = [self.p.stem(xx) for xx in line]
      return line
    
    ### Classifies the sentiment of an input string ###
    ### TO-DO: Test upgraded sentiment classifier   ###
    def classifySentiment(self, input):
      stemmedInputArr = self.stem(input)
      countPos = 0
      countNeg = 0
      neg_state = False
      for word in stemmedInputArr:
        if word in self.punctuation:
          neg_state = False

        elif any(neg in word for neg in self.negationTokens):
          neg_state = not neg_state

        elif word in self.sentiment:
          if self.sentiment[word] == "pos":
            if neg_state:
              countNeg += 1
            else:
              countPos += 1
          else:
            if neg_state:
              countPos += 1
            else:
              countNeg += 1


      bestClass = 1 if countPos >= countNeg else -1
      return bestClass



    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################


    def read_data(self):
      """Reads the ratings matrix from file"""
      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, self.ratings = ratings()
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      self.sentiment = dict(reader)
      self.sentiment = {self.p.stem(k):v for k,v in self.sentiment.iteritems()}
      self.binarize() # will be removed later

    ### Binarize the ratings matrix ###
    def binarize(self):
      ratings = np.where(self.ratings > self.BINARY_THRESH, 1, np.where(self.ratings > 0, -1, 0))

    ### Returns the cosine similarity between the ratings vector at indices movieIdx1 and movieIdx2 ###
    def findSimilarity(self, movieIdx1, movieIdx2):
      ratingVector1 = np.array(self.ratings[movieIdx1])
      ratingVector2 = np.array(self.ratings[movieIdx2])
      denom = (np.linalg.norm(ratingVector1) * np.linalg.norm(ratingVector2))
      similarity = 0 if denom == 0 else ratingVector1.dot(ratingVector2) / denom
      return similarity

    ### Generates a list of movies based on the movies in self.userMovies using collaborative filtering ###
    def recommend(self):
      ratingSums = {}
      for titlesIdx, movieDetails in enumerate(self.titles):
        title, _ = movieDetails # (title, genre)
        if (titlesIdx, title) in self.userMovies: # Don't want to return movies they've already told us
          continue
        for userMovieDetails in self.userMovies:
          userMovieIdx, userMovie = userMovieDetails # (index into self.titles, movie title)
          similarity = self.findSimilarity(titlesIdx, userMovieIdx)
          rating = self.userMovies[userMovieDetails]
          ratingSums[title] = ratingSums.get(title,0) + (similarity * rating)

      values = np.array(ratingSums.values())
      keys = np.array(ratingSums.keys())
      bestMovieIndices = np.argsort(-values) # Sort in decreasing order
      for idx in bestMovieIndices:
        movie = keys[idx]
        ### We do not want to return the same movie multiple times ###
        if movie not in self.returnedMovies:
          self.returnedMovies.add(movie)
          return movie

      ### We have already recommended every single movie (This should never happen) ###
      bestMovie = keys[bestMovieIndices[0]]
      ### Reset the set of returned movies ###
      self.returnedMovies = {bestMovie}
      return bestMovie


    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, input):
      """Returns debug information as a string for the input string from the REPL"""
      # Pass the debug information that you may think is important for your
      # evaluators
      debug_info = 'debug info'
      return debug_info


    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
      return """
      Your task is to implement the chatbot as detailed in the PA6 instructions.
      Remember: in the starter mode, movie names will come in quotation marks and
      expressions of sentiment will be simple!
      Write here the description for your own chatbot!
      """


    #############################################################################
    # Auxiliary methods for the chatbot.                                        #
    #                                                                           #
    # DO NOT CHANGE THE CODE BELOW!                                             #
    #                                                                           #
    #############################################################################

    def bot_name(self):
      return self.name


if __name__ == '__main__':
    Chatbot()
