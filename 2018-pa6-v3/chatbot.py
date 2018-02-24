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
import string
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
      self.engArticles = {'The', 'A', 'An'}
      self.engArticlesLower = {'the', 'a', 'an'}
      self.negationTokens = {"nt", "not", "no", "never"}
      self.punctuation = ".,!?"
      self.alphanum = re.compile('[^a-zA-Z0-9]')

      ### UTILS ###
      self.p = PorterStemmer()

      ### DATA ###
      self.read_data() # self.titles, self.ratings, self.sentiment
      self.userMovies = {}
      self.userMovies = {(idx, movieDetails[0]):1 for idx, movieDetails in list(enumerate(self.titles))[:4]} # will be removed, just for testing
      self.returnedMovies = set()
      
      ### RESPONSE SETS ###
      self.positiveResponses = ["You liked %s. Me too!",
                                "Yeah, %s was a great movie.",
                                "I loved %s! Glad you enjoyed it too.",
                                "I'm a huge fan of %s. I'm glad you liked it.",
                                "Ooh %s is a good one.",
                                "You're absolutely right! %s was a great movie!",
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

        return self.processMovieDetails(input, movieDetails, userMovie)

    ### Processes input for creative version ###
    def processCreative(self, input):
      ### If we already have the minimum number of movies, give them another recommendation ###
      if (len(self.userMovies) >= self.NUM_MOVIES_MIN):
        recommendation = self.recommend()
        response =  'I suggest you watch "%s".\n' % recommendation + \
                    "Would you like to hear another recommendation? (Or enter :quit if you're done.)"
        return response

      ### Try to extract movie title in quotation marks ###
      getMoviesBasic = '\"(.*?)\"'
      matches = re.findall(getMoviesBasic, input)
      movieDetails = None
      userMovie = None


      ### If user tries to give more than one movie ###
      if len(matches) > 1:
        return "Please tell me about one movie at a time. Go ahead."

      if (len(matches) == 1):
        userMovie = matches[0]
        movieDetails = self.getMovieDetailsStarter(userMovie)
        ## If our basic search doesn't work, check for misspelling
        if not movieDetails:
          movieDetails = self.checkMispellingMovie(userMovie)

      # If we have been unsuccessful check the potential splits of the sentence
      if movieDetails:
          return self.processMovieDetails(input, movieDetails, userMovie)
      else:
        ### Unable to extract movie title in quotation marks. ###
        ### Try to extract movie title with alternative methods ###
          potentialTitles = self.extractPotentialMovieTitles(input)
          movieDetails = self.getMovieDetailsCreative(potentialTitles)

        ### Still unable to extract movie title. Give up ###
          if (not movieDetails):
            return "Sorry, I don't understand. Tell me about a movie that you have seen e.g. 'I liked Toy Story'"

          return self.processMovieDetails(input, movieDetails)

    # Check if all the movies are the same title, in which case ask for date
    def allSame(self, currSeries):
      first = currSeries[0][1]
      for i in range(1, len(currSeries)):
        if first != currSeries[i][1]:
          return False
      return True



    def processMovieDetails(self, input, movieDetails, userMovie=None):
      if (not userMovie):
        userMovie = movieDetails[1]

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


    ### Extracts potential movie titles. Assumes the first word of the title is capitalized ###
    def extractPotentialMovieTitles(self, input):
      punc_regex = '[' + self.punctuation + ']'
      words = re.sub(punc_regex, lambda x: " " + x.group(0), input)
      words = [xx.strip() for xx in words.split()]

      cap_word_indices = []
      punc_indices = []
      title_state = False
      cap_word_idx = None
      ### Find the bounds of a capitalized set of words. Stops at punctuation ###
      for i, w in enumerate(words):
        if self.alphanum.sub('', w).istitle():
          if not title_state:
            title_state = True
            cap_word_idx = i
        elif title_state:
          title_state = False
          cap_word_indices.append((cap_word_idx, i))
        if (w in self.punctuation):
          punc_indices.append(i)
      if (title_state):
        cap_word_indices.append((cap_word_idx, len(words)))

      title_state = False
      cap_word_idx = None
      cap_word_indices_with_punc = []
      ## Same as above but allow for punctuation in between a set of capitalized words ###
      for i, w in enumerate(words):
        if self.alphanum.sub('', w).istitle():
          if not title_state:
            title_state = True
            cap_word_idx = i
        elif not w in self.punctuation and title_state:
          title_state = False
          cap_word_indices_with_punc.append((cap_word_idx, i))
      if (title_state):
        cap_word_indices_with_punc.append((cap_word_idx, len(words)))

      ### Titles near the end of the input are more likey to be true titles, so we reverse ###
      ### cap_word_indices i.e. 'I like The Lion King' will return the potential titles    ###
      ### 'The Lion King' and 'I'                                                          ###
      cap_word_indices.reverse()
      cap_word_indices_with_punc.reverse()

      potentialTitles = []
      for cap_word_idx, _ in cap_word_indices:
        next_punc_indices = [idx for idx in punc_indices if idx > cap_word_idx]
        ### Assume the next punctuation mark after the capitalized word delimits the ###
        ### movie title or that movie title goes until the end of the line e.g.      ###
        ### 'I liked Snow White, but not very much' -> 'Snow White' or               ###
        ### 'I loved The Lion King' -> 'The Lion King'                               ###
        next_punc_idx = next_punc_indices[0] if next_punc_indices else len(words)
        potentialTitles.append(' '.join(words[cap_word_idx:next_punc_idx]))
      for cap_word_idx, cap_word_idx_f in cap_word_indices:
        potentialTitles.append(' '.join(words[cap_word_idx:cap_word_idx_f]))
      for cap_word_idx, cap_word_idx_f in cap_word_indices_with_punc:
        potentialTitles.append(' '.join(words[cap_word_idx:cap_word_idx_f]))

      for cap_word_idx, _ in cap_word_indices:
        for j in xrange(len(words), cap_word_idx, -1):
          joinedWords = ' '.join(words[cap_word_idx:j])
          potentialTitles.append(joinedWords.translate(None, string.punctuation).strip())
          potentialTitles.append(joinedWords.strip())

      ### Remove duplicates, while maintaining the order ###
      alreadySeen = set()
      potentialTitlesFinal = []
      for title in potentialTitles:
        if (not title in alreadySeen):
          alreadySeen.add(title)
          potentialTitlesFinal.append(title)

      print "Potential Titles: " + str(potentialTitlesFinal)

      return potentialTitles


    ### Tries to return movie details, assuming the movie title is one of ###
    ### the entries in potentialMovieTitles                               ###
    def getMovieDetailsCreative(self, potentialMovieTitles):
      for movieTitle in potentialMovieTitles:
        movieDetails = self.getMovieDetailsCreativeHelper(movieTitle)
        if (movieDetails):
          return movieDetails
      return None


    ##Check if the movie is in a series
    def checkForMovieInSeries(self, moviesInSeries, userMovie):
      if self.allSame(moviesInSeries):
          date = raw_input("We have multiple films by the name of " + userMovie + ". What year was yours released? (e.g 1945)")
          return self.getMovieDetailsStarter(userMovie + "(" + date + ")")
      else:
          failedLastTime = False
          while (len(moviesInSeries) > 1): #Whilst the movies found
            second_part = "These are the movies I know that match " + userMovie + ":\n " + moviesInSeries + "\nWhich " + userMovie + " did you mean?\n"
            if failedLastTime: #The user entered something that didn't filter movies found
              second_part = "Sorry, we don't have any films that match " + userMovie + " and " + specific_thing + ". Please try again. " + second_part
            specific_thing = raw_input(second_part) #Filter word
            newMoviesInSeries = [m for m in moviesInSeries if specific_thing in m[1]]
            if (len(newMoviesInSeries) > 0): #Filtered list
              failedLastTime = False
              moviesInSeries = newMoviesInSeries
            else:
              failedLastTime = True

          return self.getMovieDetailsCreativeHelper(moviesInSeries[0][1])


          #TODO: check for either an integer or a title
           

    ### Identical to getMovieDetailsStarter, but doesn't require the date ###
    ### to be entered and ignores case.                                   ###
    def getMovieDetailsCreativeHelper(self, movie):
      movie = movie.lower()
      movieWordTokens = movie.split()
      if (not movieWordTokens):
        return None

      if (re.search(r'\(\d{4}\)', movieWordTokens[-1])): # check if last word is a date
        movieWordTokens = movieWordTokens[:-1]

      if movieWordTokens[0] in self.engArticlesLower: # check not empty and first word is an article e.g. American in Paris, An
        movie = ' '.join(movieWordTokens[1:]) + ', ' + movieWordTokens[0]

      movie = " ".join(movieWordTokens).strip()

      moviesInSeries = []

      for idx, movieDetails in enumerate(self.titles):
        title, genre = movieDetails
        titleWithoutDate = title[:-7].lower() # (1995) -> 6 characters + 1 space character
        if titleWithoutDate == movie:
          print idx, title
          return idx, title

        # TODO: Add a check to make sure that we don't accept words like "I" or whatever
        elif titleWithoutDate.__contains__(movie) and len(movie) >= 4: #e.g. user enters Star Wars, check for "Star Wars I"
          moviesInSeries.append((idx, titleWithoutDate))

      # If we have some sort of a series here, we should check that
      if len(moviesInSeries) > 1:
        potential_movie = self.checkForMovieInSeries(moviesInSeries, movie)
        if potential_movie:
          return potential_movie

      return None

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


    ### If the standard check fails we check to see if the spelling may have been incorrect, ###
    ### and then return this as an option, or None if there are no convincing options. ###
    # TODO: I just need to read the slides on min edit distance, need to use some DP here I think
    def checkMispellingMovie(self, movie):
      movie = self.checkReorder(movie)
      for idx, movieDetails in enumerate(self.titles):
        title, genre = movieDetails
        if title == movie:
          return idx, title
      return None


    ### Returns the idx into self.titles and the movie title of 'movie' ###
    ### 'movie' must match the title in self.titles exactly.            ###
    def getMovieDetailsStarter(self, movie):
      movie = self.checkReorder(movie)
      for idx, movieDetails in enumerate(self.titles):
        title, genre = movieDetails
        if title == movie:
          return idx, title

      return None

    ## Check to see if the movie title needs to be re-ordered
    def checkReorder(self, movie):
      movieWordTokens = movie.split()
      if movieWordTokens and movieWordTokens[0] in self.engArticles: # check not empty and first word is an article
        ### Transforms movie title of the form 'An American in Paris (1951)' to 'American in Paris, An (1951)'
        movie = ' '.join(movieWordTokens[1:-1]) + ', ' + movieWordTokens[0] + ' ' + movieWordTokens[-1]
      return movie


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
