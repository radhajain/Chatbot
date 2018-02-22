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
from random import randint
from PorterStemmer import PorterStemmer

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    #############################################################################
    # `moviebot` is the default chatbot. Change it to your chatbot's name       #
    #############################################################################
    def __init__(self, is_turbo=False):
      self.name = 'seb'
      self.is_turbo = is_turbo
      self.userMovies = {}
      self.NUMBER_MOVIES = 5
      self.p = PorterStemmer()
      self.alphanum = re.compile('[^a-zA-Z0-9]')
      self.negativeWords = {"nt", "not", "no", "never"}
      self.punctation = ".,!?;:"
      self.read_data()
      
      

    #############################################################################
    # 1. WARM UP REPL
    #############################################################################

    def greeting(self):
      """chatbot greeting message"""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = 'Hi! I\'m MovieBot! I\'m going to recommend a movie to you. First I will ask you about your taste in movies. Tell me about a movie that you have seen.'

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return greeting_message

    def goodbye(self):
      """chatbot goodbye message"""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = 'Thank you! Have a nice day!'

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

      if self.is_turbo == True:
        response = 'processed %s in creative mode!!' % input
      else:
        getMovies = '\"(.*?)\"'
        matches = re.findall(getMovies, input)
        print matches

        #If user tries to give more than one movie
        if len(matches) > 1:
          return 'I didn\'t quite understand you. Please only mention one movie per response.'
        elif len(matches) == 0:
          return 'Sorry, I don\'t understand. Please enter a movie in "quotation marks"'
        else: 
          stemmed = self.stem(input)
          print "After stemming: %s" % stemmed
          withoutTitle = stemmed.replace("\"" + matches[0] + "\"", '')
          #Must include a sentiment, e.g. "I like..."
          if not withoutTitle:
            return 'Tell me more about "%s"' % matches[0]

          #Movie details of form (['Toy Story(1996), 'Adventure|Comedy'], 74)
          movieDetails = self.getMovieDetails(matches[0])
          #Sentiment is either pos or neg
          sentiment = self.classifySentimet(withoutTitle)
          if not movieDetails:
            return 'Sorry, I don\'t know that movie. Try another'
          else:
            #Store sentiment for movies - string representation of movieDetails
            if (movieDetails) in self.userMovies:
              return 'You already have %s. Try another one!' % matches[0]
            self.userMovies[movieDetails] = sentiment

          if len(self.userMovies) == self.NUMBER_MOVIES:
            #When we have all the movies we need
            recedMovies = self.recommend(self.userMovies)
            return 'Thank you for your patience, That\'s enough for me to make a recommendation. These are the movies you like: %s' % str(recedMovies)
          else:
            if sentiment == 1:
              return 'You liked %s. Me too! Tell me about another movie you have seen.' % movieDetails[0][0]
            else:
              return 'You did not like %s. It sucks huh. Tell me about another movie you have seen.' % movieDetails[0][0]
            


    def getPositiveResponses(self, movie):
      #Remeber responses seen, and pick a new one
      pass


    def getNegativeResponses(self,movie):
      pass



    def stem(self, line):
      # make sure everything is lower case
      line = line.lower()
      # split on whitespace
      line = [xx.strip() for xx in line.split()]
      # remove non alphanumeric characters
      line = [self.alphanum.sub('', xx) for xx in line]
      # remove any words that are now empty
      line = [xx for xx in line if xx != '']
      # stem words
      line = [self.p.stem(xx) for xx in line]
      # add to the document's conents
      return " ".join(line)



    def getMovieDetails(self, movie):
      print movie
      # getName = '(.*?)(?:\s\()'
      #If insufficient: Create bag of words and compare length
      for i in range(0, len(self.titles)):
        # potentialMovies = re.findall(getName, self.titles[i][0])
        if (self.titles[i][0].lower() == movie.lower()):
          return (tuple(self.titles[i]),i)
      return None

    
    def classifySentimet(self, withoutTitle):
      #Remove the movie title from the response
      countPos = 0
      countNeg = 0
      neg_state = False
      for word in withoutTitle.split(' '):
        #need to stem word
        if any(punc in word for punc in self.punctation):
          neg_state = False

        if any(neg in word for neg in self.negativeWords):
          neg_state = not neg_state

        if word in self.sentiment:
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
      print self.sentiment.keys()[:10]



    def binarize(self):
      """Modifies the ratings matrix to make all of the ratings binary"""
      for i in range(0, len(self.ratings)):
        for j in range (0, len(self.ratings[i])):
          if (self.ratings[i][j] >= 2.5):
            self.ratings[i][j] = 1
          elif (self.ratings[i][j] == 0):
             self.ratings[i][j] = 0
          else:
            self.ratings[i][j] = -1




    def findSimilarity(self, movie, userMovie):
      #Find the cosine similarity between the binarized rating vector for movie and for userMovie
      userRating = np.array(self.ratings[userMovie[1]])
      movieRating = np.array(self.ratings[movie[1]])
      denom = (np.linalg.norm(userRating) * np.linalg.norm(movieRating))
      if denom == 0: 
        return 0
      similarity = userRating.dot(movieRating) / denom
      return similarity



      

    def recommend(self, u):
      """Generates a list of movies based on the input vector u using
      collaborative filtering"""

      self.binarize()
      print "done binasing"
      #Stores the rxi's
      ratingSums = {}
      for idx, movie in enumerate(self.titles):
        movie = tuple(movie)
        for userMovie in self.userMovies:
          if userMovie[0] == movie:
            continue
          similarity = self.findSimilarity((movie, idx), userMovie)
          rating = self.userMovies[userMovie]
          ratingSums[movie] = ratingSums.get(movie,0) + (similarity * rating)

      values = np.array(ratingSums.values())
      keys = np.array(ratingSums.keys())
      bestMovieIndices = np.argsort(values)
      bestMovies = keys[bestMovieIndices][::-1][:10]
      return bestMovies


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
