#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA6, CS124, Stanford, Winter 2018
# v.1.0.2
# Original Python code by Ignacio Cases (@cases)
######################################################################
import csv
import math
import re
import numpy as np

from movielens import ratings
from random import randint

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    #############################################################################
    # `moviebot` is the default chatbot. Change it to your chatbot's name       #
    #############################################################################
    def __init__(self, is_turbo=False):
      self.name = 'moviebot'
      self.is_turbo = is_turbo
      self.userMovies = {}
      self.NUMBER_MOVIES = 5
      self.read_data()
      

    #############################################################################
    # 1. WARM UP REPL
    #############################################################################

    def greeting(self):
      """chatbot greeting message"""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = 'Hi! Tell me about a movie you like'

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return greeting_message

    def goodbye(self):
      """chatbot goodbye message"""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = 'Have a nice day!'

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
      

      if self.is_turbo == True:
        response = 'processed %s in creative mode!!' % input
      else:
        getMovies = '\"(.*?)\"'
        matches = re.findall(getMovies, input)

        #If user tries to give more than one movie
        if len(matches) > 1:
          return 'I didn\'t quite understand you. Please only mention one movie per response.'
        elif len(matches) == 0:
          return 'u no good'
        else: 
          #Movie details of form ['Toy Story(1996), 'Adventure|Comedy']
          movieDetails = self.getMovieDetails(matches[0])
          sentiment = self.classifySentimet(input, matches[0])
          if not movieDetails:
            return 'Try again lol'
          else:
            #Store sentiment for movies - string representation of movieDetails
            if repr(movieDetails) in self.userMovies:
              return 'You already have %s. Try another one!' % matches[0]
            self.userMovies[repr(movieDetails)] = sentiment

          if len(self.userMovies) == self.NUMBER_MOVIES:
            #When we have all the movies we need
            print 'Thank you for your patience, imma tell u a movie'
            self.recommend(self.userMovies)
          else:
            print self.userMovies
            if sentiment == "pos":
              return 'You really liked %s. Me too! What\'s another one?' % matches[0]
            else:
              return 'You did not like %s. It sucks huh. What\'s another one?' % matches[0]
            



    def getMovieDetails(self, movie):
      getName = '(.*?)(?:\s\()'
      #If insufficient: Create bag of words and compare length
      for i in range(0, len(self.titles)):
        potentialMovies = re.findall(getName, self.titles[i][0])
        if (potentialMovies):
          currMovie = potentialMovies[0].lower()
          if (currMovie == movie.lower()):
            return self.titles[i]
      return None

    
    def classifySentimet(self, response, title):
      withoutTitle = response.replace(title, '')
      countPos = 0
      countNeg = 0
      for word in response:
        if word in self.sentiment:
          if self.sentiment[word] == "pos":
            countPos += 1
          else:
            countNeg += 1
      bestClass = 'pos' if countPos >= countNeg else 'neg'
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


    def binarize(self):
      """Modifies the ratings matrix to make all of the ratings binary"""

      pass


    def distance(self, u, v):
      """Calculates a given distance function between vectors u and v"""
      # TODO: Implement the distance function between vectors u and v]
      # Note: you can also think of this as computing a similarity measure

      pass


    def recommend(self, u):
      """Generates a list of movies based on the input vector u using
      collaborative filtering"""
      # TODO: Implement a recommendation function that takes a user vector u
      # and outputs a list of movies recommended by the chatbot

      print "I'm in reccommend!!!!!!"
      return ["Me", "Seb"]


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
