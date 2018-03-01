#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA6, Cw124, Stanford, Winter 2018
# v.1.0.2
# Original Python code by Ignacio Cases (@cases)
# Contributors: Sebastien Goddijn, Radha Jain, George Preudhomme


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
      self.MAX_EDIT_DIST = 2
      self.DATE_REGEX = r'\(?(?P<date>\d{4})\)?'

      ### TOKEN SETS ###
      self.engArticles = {'The', 'A', 'An'}
      self.engArticlesLower = {'the', 'a', 'an'}
      self.negationTokens = {"nt", "not", "no", "never"}
      self.punctuation = ".,!?"
      self.alphanum = re.compile('[^a-zA-Z0-9]')
      self.ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

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

      self.canYouAnswers = ["I most certainly can %s, but I'm a bit busy at the moment.",
                            "As much as I would like to %s right now, I'm just too excited about movies!",
                            'I could, but can you %s?']

      self.whatIsAnswers = ["I'm unsure what %s is, but you're smart! I'm sure you can figure it out!",
                            "Once upon a time, a wise man from a forgotten realm spoke to me... In his infinite wisdom " + \
                            "he told me precisely what %s is. But, alas, I forgot..."]

      self.catchAllAnswers = ["Cool, that's great.",
                              'Ok, got it.',
                              'Nice!',
                              'Thanks for sharing!',
                              'You really should talk to someone about that.']

      self.returnToMovies = ["Now let's get back to movies. Tell me about a movie that you have seen e.g. 'I liked Toy Story'",
                              "Ok enough chitter chatter. What's another movie you can remember?",
                              "Soo, back to movies?",
                              "Please can we talk about movies?!",
                              "Talk movies to me.",
                              "I like talking about movies. Can we do that again?",
                              "Remember when we were talking about movies? I liked that",
                              "You keep getting distracted. I just wanna talk about movies.",
                              "Please stop distracting me, my boss is looking. Let's chat about movies."]

    #############################################################################
    # 1. WARM UP REPL
    #############################################################################

    def greeting(self):
      """chatbot greeting message"""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = "Let's get going!\nTell me about a movie that you have seen, I promise I won't judge."

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
      input = input.strip()
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
        
    #############################################################################
    # STARTER MODE                                                              #
    #############################################################################

    ### Processes input for starter version ###
    def processStarter(self, input):

      getMovies = '\"(.*?)\"'
      matches = re.findall(getMovies, input)

      ### If user tries to give more than one movie ###
      if len(matches) > 1:
        return "Please tell me about one movie at a time. Go ahead."

      ### If no movie is specified in quotation marks ###
      elif len(matches) == 0:
        return "Sorry, I don't understand." + random.choice(self.requestAnotherResponses)

      else: 
        userMovie = matches[0]
        formattedUserMovie = self.processMovieTitle(userMovie)
        movieDetails = self.getMovieDetailsStarter(formattedUserMovie)

        ### The movie is not in the database ###
        if (not movieDetails):
          return "I'm sorry, I haven't seen that one yet! Please tell me about another movie."

        withoutTitle = input.replace("\"" + userMovie + "\"", '')
        return self.processMovieDetails(withoutTitle, movieDetails)

    #############################################################################
    # CREATIVE MODE                                                             #
    #############################################################################

    ### Processes input for creative version ###
    def processCreative(self, input):

      ### Try to extract movie title in quotation marks ###
      getMoviesBasic = '\"(.*?)\"'
      matches = re.findall(getMoviesBasic, input)
      movieDetails = None
      userMovie = None

      ### If user tries to give more than one movie ###
      if len(matches) > 1:
        return "Please tell me about one movie at a time. Go ahead."

      if (len(matches) == 1): #User gave movie "in quotes"
        userMovie = matches[0]
        formattedUserMovie = self.processMovieTitle(userMovie)
        movieDetails = self.getMovieDetailsStarter(formattedUserMovie)
        ### If our basic search doesn't work ###
        if not movieDetails: 
          movieDetails = self.getMovieDetailsCreative([userMovie])

      ### We've found the correct movie ###
      if movieDetails:
        withoutTitle = input.replace("\"" + userMovie + "\"", '')
        return self.processMovieDetails(withoutTitle, movieDetails)

      else:
        ### Unable to extract movie title in quotation marks. ###
        ### Try to extract movie title with alternative methods ###
        potentialTitles = self.extractPotentialMovieTitles(input)
        movieDetails = self.getMovieDetailsCreative(potentialTitles)

        ### Still unable to extract movie title. User must be talking about something arbitrary ###
        if (not movieDetails):
          inputTokens = input.split()
          ### Assume that if a word was capitalized (other than the first word) they meant to give us a movie ###
          if any([w.istitle() for w in inputTokens[1:]]):
            return "Sorry I couldn't understand that last movie title you gave me!\nLet's try again." + random.choice(self.requestAnotherResponses)

          return self.checkArbitraryInput(input) + random.choice(self.returnToMovies)

        withoutTitle = self.removeTitle(input, movieDetails[1])
        return self.processMovieDetails(withoutTitle, movieDetails)

    ### Removes the movie title from the input string ###
    ### TO-DO: Update this function to work for I love Love is Strange ###
    def removeTitle(self, input, userMovie):
      ### Split user movie into a bag of words, and replace common occurrences with input ###
      bag_of_words_title = userMovie.translate(None, string.punctuation).lower().strip().split(' ')
      input = input.translate(None, string.punctuation).lower().strip()
      for word in bag_of_words_title:
        if input.__contains__(word):
          input = input.replace(word, '')
      return input.strip()

    #############################################################################
    # BOTH                                                                      #
    #############################################################################

    ### Makes a recommendation/asks for more movies ###
    def processMovieDetails(self, withoutTitle, movieDetails):
      userMovie = movieDetails[1]

      ### The user has already given us this movie ###
      if movieDetails in self.userMovies:
        return 'You mentioned that one already. Please tell me another.'
    
      ### Must include a sentiment, e.g. "I like..." ###
      while (not withoutTitle):
        withoutTitle = raw_input('How did you like "%s"?\n' % userMovie).strip()

      ### Classify sentiment ###
      sentiment = self.classifySentiment(withoutTitle)
      while (not sentiment):
        userSentimentInput = raw_input('I\'m sorry, I\'m not quite sure if you liked "%s".\nTell me more about "%s" e.g. \'I liked it\'\n' % (userMovie, userMovie))
        sentiment = self.classifySentiment(userSentimentInput)

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
                    "Would you like to hear another recommendation? (yes/no)"

        if self.is_turbo:
          ### Allow the user to ask for more recommendations ###
          print response
          response = raw_input("> ").strip()
          while (self.getYesOrNo(response)):
            recommendation = self.recommend()
            print('I suggest you watch "%s".\n' % recommendation)
            print('Would you like to hear another recommendation? (yes/no)')
            response = raw_input("> ").strip()
          return "OK. You can tell me more about your taste in movies (Or enter :quit if you're done)"

      else:
        response += ' ' + random.choice(self.requestAnotherResponses)

      return response

    def stripMovieTitle(self, input, userMovie):
      ### Split user movie into a bag of words, and replace common occurrences with input ###
      bag_of_words_title = userMovie.lower().split(' ')
      input = input.lower()
      for word in bag_of_words_title:
        if input.__contains__(word):
          input = input.replace(word, '')
      return input

    #############################################################################
    # GET MOVIE DETAILS                                                          #
    #############################################################################

    ### Returns the idx into self.titles and the movie title of 'movie' ###
    ### 'movie' must match the title in self.titles exactly.            ###
    def getMovieDetailsStarter(self, movie):
      for idx, movieDetails in enumerate(self.titles):
        title, genre = movieDetails
        if title == movie:
          return idx, title

      return None

    ### Tries to return movie details, assuming the movie title is one of ###
    ### the entries in potentialMovieTitles                               ###
    def getMovieDetailsCreative(self, potentialMovieTitles):
      self.alreadyAsked = set() # reset the set of movies we've already asked them about
      for movieTitle in potentialMovieTitles:
        movieDetails = self.getMovieDetailsCreativeHelper(movieTitle)
        if (movieDetails):
          return movieDetails
      return None

    ### Identical to getMovieDetailsStarter, but doesn't require the date ###
    ### to be entered and ignores case.                                   ###
    def getMovieDetailsCreativeHelper(self, movie):
      origMovie = movie
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
        if title in self.alreadyAsked: # We've already asked the user if this is the movie
          continue
        possibleMatch, userConfirmedMatch = self.isMatch(title, movie) 
        if (userConfirmedMatch):
          return idx, title
        elif (possibleMatch):
          moviesInSeries.append((idx, title)) # Append the title WITH date to help user disambiguate

      if len(moviesInSeries) == 1:
        return moviesInSeries[0]

      # Check for potential series
      if len(moviesInSeries) > 1:
        return self.extractMovieInSeries(moviesInSeries, origMovie) # Returns (idx, title) or None

      return None

    def removeArticle(self, title):
      titleTokens = title.split()
      if titleTokens[0].lower() in self.engArticlesLower:
        return " ".join(titleTokens[1:])
      return title

    ### EXTENSION: Responding to arbitrary input ###
    ################################################

    ### Responds to input of the form 'Can you ...?' or 'What is ...?' ###
    def checkArbitraryInput(self, input):
      if input.lower().startswith("can you"):
        input = input[len("can you"):]
        input = input.replace(" me ", " you ")
        return (random.choice(self.canYouAnswers) + "\n") % self.trimInput(input)

      if input.lower().startswith("what is"):
        input = input[len("what is"):]
        input = input.replace(" me ", " you ")
        input = input.replace(" my ", " your ")
        return (random.choice(self.whatIsAnswers) + "\n") % self.trimInput(input)

      if input.lower().startswith("what\'s"):
        input = input[len("what\'s"):]
        input = input.replace(" me ", " you ")
        input = input.replace(" my ", " your ")
        return (random.choice(self.whatIsAnswers) + "\n") % self.trimInput(input)

      return ""

    def trimInput(self, input):
      input = input.translate(None, string.punctuation) # remove punctuation
      return " ".join(input.split()) # remove extraneous whitespace

    ### EXTENSION 1: Identifying movies without quotation marks or perfect capitalization ###
    #########################################################################################

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

      potentialTitles = list(set(potentialTitles)) 
      potentialTitles = sorted(potentialTitles, key=len, reverse=True)

      return potentialTitles

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

    ### EXTENSION 2: Disambiguating movie titles for series and year ambiguities ###
    ##############################################################################

    ### Extract a single movie is in a series. Takes in a list of potential movies and returns the filtered movie ###
    def extractMovieInSeries(self, moviesInSeries, userMovie):      
      failedLastTime = False
      filter_phrase = None
      while (len(moviesInSeries) > 1):
        print ('\n' if not failedLastTime else '') + "These are the movies I know that match %s" % userMovie + \
                          (" and %s" % filter_phrase if filter_phrase else '') + ':'
        
        for i in range(len(moviesInSeries)):
          print str(i) + ".) " + str(moviesInSeries[i][1])

        request_string = "If the movie you were searching for is present in the above list, please enter either its date, part of " + \
                          "the title, or 'None' if your movie is not there\n> "

        if failedLastTime: # The user entered something that didn't match any of the movies
          request_string = "\nSorry, we don't have any films that match %s and %s. Please try again.\n" % (userMovie, filter_phrase) + request_string

        filter_phrase = raw_input(request_string)
        filter_phrase_processed = filter_phrase.lower().strip()
        if (filter_phrase_processed == 'none'): # The user was not referring to any of the movies in moviesInSeries
          return None

        newMoviesInSeries = [m for m in moviesInSeries if filter_phrase_processed in m[1].lower()]
        if (len(newMoviesInSeries) > 0):
          failedLastTime = False
          moviesInSeries = newMoviesInSeries
        else: # filter_phrase doesn't match any of the potential movies
          failedLastTime = True 

        return moviesInSeries[0]

    ### Extracts a valid date (date in movieYears) from the user and returns the index ###
    ### of that date in the movieYears array                                           ###
    def extractDate(self, date, movieYears):
      match = re.match(self.DATE_REGEX, date)
      while (not match or not match.group("date") in movieYears):
        date = raw_input("Please enter a valid date (e.g. %s)" % movieYears[0]).strip()
      return movieYears.index(date)
           
    ### EXTENSION 3: HANDLING MISPELLINGS ###
    ##################################################

    ### Checks for misspelling and series matches. Returns a (possibleMatch, userConfirmedMatch) ###
    ### tuple. If user explicitly confirms, userConfirmedMatch is set to be True, otherwise it   ###
    ### is set to False.                                                                         ###
    def isMatch(self, title, movie):
      titleWithoutDate = title.lower()[:-7] # (1995) -> 6 characters + 1 space character
      titleWithoutArticle = self.removeArticle(titleWithoutDate)

      if movie == titleWithoutDate or movie == titleWithoutArticle:
        return (True, False)

      if len(movie) <= 4:
        return (False, False)

      ### movie can be a maximum of self.MAX_EDIT_DIST away from titleWithoutDate to still be considered a match ###
      if not self.editDistanceExceeds(titleWithoutDate, movie, self.MAX_EDIT_DIST) or not self.editDistanceExceeds(titleWithoutArticle, movie, self.MAX_EDIT_DIST):
        response = raw_input("Did you mean %s? (Enter yes if this suggestion is correct)\n> " % title).strip()
        self.alreadyAsked.add(title)
        return (False, self.getYesOrNo(response))

      # movie is a prefix of titleWithoutDate and len(movie) >= 4 to prevent superfluous matches like 'I' -> 'Ice Age'
      if not titleWithoutDate.startswith(movie) and not titleWithoutArticle.startswith(movie):
        return (False, False)

      return (True, False)

    ### Returns True if the edit distance between w1 and w2 exceeds maxEditDist ###
    def editDistanceExceeds(self, w1, w2, maxEditDist):
      if abs(len(w2) - len(w1)) > maxEditDist:
        return True
      if (self.levenshtein(w1.lower(), w2.lower()) > maxEditDist):
        return True
      return False

    ### Computes the levenshtein distance between source and target ###
    def levenshtein(self, source, target):
      if len(source) < len(target):
        return self.levenshtein(target, source)

      # So now we have len(source) >= len(target).
      if len(target) == 0:
        return len(source)

      # We call tuple() to force strings to be used as sequences
      # ('c', 'a', 't', 's') - numpy uses them as values by default.
      source = np.array(tuple(source))
      target = np.array(tuple(target))

      # We use a dynamic programming algorithm, but with the
      # added optimization that we only need the last two rows
      # of the matrix.
      previous_row = np.arange(target.size + 1)
      for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
          current_row[1:],
          np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
          current_row[1:],
          current_row[0:-1] + 1)

        previous_row = current_row

      return previous_row[-1]

    def getYesOrNo(self, input):
      while not input.lower().startswith('y') and not input.lower().startswith('n'):
        input = raw_input("Please answer yes or no.\n").strip()
      return input.lower().startswith('y') # will be false if it starts with 'n'
    
    #############################################################################
    # SENTIMENT CLASSIFICATION                                                  #
    #############################################################################

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

      if countPos > countNeg:
        return 1
      if countNeg > countPos:
        return -1
      return 0

    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    ##############################################################################
    # DATA PROCESSING                                                            #
    ##############################################################################
    def read_data(self):
      """Reads the ratings matrix from file"""
      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, self.ratings = ratings()
      # Process movie titles from 'Lion King, The (1994)' -> 'The Lion King (1994)'
      self.titles = [(self.processMovieTitle(title), genre) for (title, genre) in self.titles]
      self.ratings = np.array(self.ratings)
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      self.sentiment = dict(reader)
      self.sentiment = {self.p.stem(k):v for k,v in self.sentiment.iteritems()}
      self.pearsonize()
      self.binarize() 

    ### Binarize the ratings matrix ###
    def binarize(self):
      self.ratingsBinary = np.where(self.ratings > self.BINARY_THRESH, 1, np.where(self.ratings > 0, -1, 0))

    ### Mean-center the ratings matrix ###
    def pearsonize(self):
      non_zero = np.count_nonzero(self.ratings, axis=1)
      mean = self.ratings.sum(axis=1)[:, None] / np.where(non_zero > 0, non_zero, 1)[:, None]
      self.ratingsMeanCentered = self.ratings - np.where(self.ratings > 0, mean, 0)

    ##############################################################################
    # RECOMMENDATION ENGINE                                                      #
    ##############################################################################
    
    ### Generates a list of movies based on the movies in self.userMovies using collaborative filtering ###
    def recommend(self):
      if (self.is_turbo):
        simScores = self.pearsonCollabFiltering()
      else:
        simScores = self.binaryCollabFiltering()
      bestMovieIndices = np.argsort(-simScores) # Sort in decreasing order
      for idx in bestMovieIndices:
        title, _ = self.titles[idx] # (title, genre)
        ### We do not want to return the same movie multiple times ###
        if title not in self.returnedMovies:
          self.returnedMovies.add(title)
          return title

      ### We have already recommended every single movie (This should never happen) ###
      bestMovie = self.titles[bestMovieIndices[0]][0]
      ### Reset the set of returned movies ###
      self.returnedMovies = {bestMovie}
      return bestMovie

    ### EXTENSION 4: Using non-binarized dataset ###
    ##############################################

    ### Returns a numpy array of similarity scores 'movieRatings' where the similarity score at index i ###
    ### corresponds to the movie at self.titles[i]                                                      ###
    def pearsonCollabFiltering(self):
      userMovieRatings = np.array([rating for _, rating in self.userMovies.items()]) # numpy array of user movie ratings i.e. [1, 0, 1] (ratings are not required to be binary)
      userMovieIndices = np.array([movieDetails[0] for movieDetails, _ in self.userMovies.items()]) # numpy array of indices into self.titles corresponding to userMovieRatings
      movieRatings = np.zeros((len(self.titles),))
      for titlesIdx, movieDetails in enumerate(self.titles):
        title, _ = movieDetails # (title, genre)
        if (titlesIdx, title) in self.userMovies: # Don't want to return movies they've already told us
          continue

        similarities = np.zeros((len(self.userMovies),)) # array of similarity values between movie i and the user movies
        for i, userMovieIdx in enumerate(userMovieIndices):
          sim = self.cosineSimilarity(self.ratingsMeanCentered[titlesIdx], self.ratingsMeanCentered[userMovieIdx])
          if (sim > 0): # only keep the similarity values > 0
            similarities[i] = sim

        movieRatings[titlesIdx] = np.sum(similarities * userMovieRatings)

      if (movieRatings.max() <= 0): # If we predict the user won't like the best movie in our list, use binary collaberative filtering instead
        return self.binaryCollabFiltering()
      else:
        return movieRatings

    ### Returns a numpy array of similarity scores 'ratingSums' where the similarity score at index i ###
    ### corresponds to the movie at self.titles[i]                                                    ###
    def binaryCollabFiltering(self):
      ### Use an array instead of a dict for speed/efficiency ###
      ratingSums = np.zeros((len(self.titles),)) # array of ratingSums where ratingSums[i] corresponds to the ratingSum for the movie at self.titles[i]
      for titlesIdx, movieDetails in enumerate(self.titles):
        title, _ = movieDetails # (title, genre)
        if (titlesIdx, title) in self.userMovies: # Don't want to return movies they've already told us
          continue
        for userMovieDetails in self.userMovies:
          userMovieIdx, userMovie = userMovieDetails # (index into self.titles, movie title)
          similarity = self.cosineSimilarity(self.ratingsBinary[titlesIdx], self.ratingsBinary[userMovieIdx])
          rating = self.userMovies[userMovieDetails]
          ratingSums[titlesIdx] += (similarity * rating)
      return ratingSums

    ### Returns the cosine similarity between the two vectors ###
    def cosineSimilarity(self, vec1, vec2):
      denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
      similarity = 0 if denom == 0 else vec1.dot(vec2) / denom
      return similarity

    ### Converts movie titles of the form 'Lion King, The (1994)' to 'The Lion King (1994).' ###
    ### Movie titles not of this form are left alone. Assumes the year is the title suffix.  ###
    def processMovieTitle(self, title):
      titleTokens = title.split()
      if len(titleTokens) >= 3 and titleTokens[-2].lower() in self.engArticlesLower and titleTokens[-3].endswith(','):
        titleTokens[-3] = titleTokens[-3][:-1] # remove comma
        title = ' '.join([titleTokens[-2]] + titleTokens[:-2] + [titleTokens[-1]])
 
      return title


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
      Hello! I am Seb, an all wise and all knowing movie reccomendation bot. I am here, 
      to help you decide what to do on your lonely Friday nights, when Netflix and seems 
      the only option. I have a vast and diverse catalog of films for your viewing pleasure,
      just tell me some of the things you like and we'll be on our way! 
      - I am able to recognize movie titles without quotation marks or proper capitalization (so 
      long as the first word is capitalized).
      - I am able to do fine grain sentiment analysis, so I can recognize just how much you love 
      or hate a movie. 
      - I am able to recognize misspelled titles (within reason), but I can't do much
       about your face. 
      - I use a non-binarized data set to provide the utmost precision in my reccomendations
      - I respond well to arbitrary input, so ask me anything!
      - Finally, I am able to disambiguate movie titles for series and year ambiguities
      I hope you enjoy playing with me, as much as I have enjoyed playing with myself!!
      Enjoy!!!!!
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
