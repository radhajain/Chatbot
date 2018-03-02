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
      self.BINARY_THRESH = 3.0
      self.MAX_EDIT_DIST = 2
      self.DATE_REGEX = r'\((?:\d{4})?-?(?:\d{4})?\)$'
      self.STRONG_SENTIMENT_MULTIPLIER = 10
      self.INTENSIFIER_MULTIPLIER = 5
      self.FIRST_TIME = True

      ### UTILS ###
      self.p = PorterStemmer()
      self.inputMarker = '\001\033[96m\002%s> \001\033[0m\002' % self.name

      ### TOKEN SETS ###
      self.engArticles = {'The', 'A', 'An'}
      self.engArticlesLower = {'the', 'a', 'an'}
      self.negationTokens = {"nt", "not", "no", "never"}
      self.punctuation = ".,!?"
      self.alphanum = re.compile('[^a-zA-Z0-9]')
      self.ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
      self.strongPosWords = set([self.p.stem(w) for w in ['love', 'favorite', 'amazing', 'stunning', 'great', 'heartwarming', 'hysterical']])
      self.strongNegWords = set([self.p.stem(w) for w in ['hate', 'despise', 'disgusting', 'dull', 'annoying', 'unoriginal', 'boring', 'eh', 'meh']])
      self.intensifiers = set([self.p.stem(w) for w in ['really', 'very', 'so']])

      ### DATA ###
      self.read_data() # self.titles, self.ratings, self.sentiment
      self.userMovies = {}
      self.returnedMovies = set()
      
      ### RESPONSE SETS ###
      self.veryPositiveResponses = ["Wait %s was one my favorite movies!! Happy to hear you liked it so much!",
                                  "Wow, you really liked that one, huh? I totally agree, %s was so so good.",
                                  "Someone's excited! You really enjoyed that one. I've literally recommended %s to everyone I've ever met.",
                                  "You're absolutely right, %s is SUCH a good movie. ",
                                  "You sound like you liked %s even more than I do... and I'm obsessed, so that's saying something."]

      self.veryNegativeResponses = ["Woah, you really didn't like %s! I know what you mean, it was pretty disappointing.",
                                    "Yeah, I hated %s too. Glad I'm not the only one.",
                                    "You're absolutely right, %s was terrible!"]

      self.positiveResponses = ["You liked %s. Me too!",
                                "Yeah, %s was a great movie.",
                                "I loved %s! Glad you enjoyed it too.",
                                "I'm a huge fan of %s. I'm glad you liked it.",
                                "Ooh %s is a good one.",
                                "You're absolutely right, %s was a gem.",
                                "Right on! %s is definitely a classic."]

      self.negativeResponses = ["I agree, %s was pretty bad.",
                                "I didn't like %s either.",
                                "Yeah %s wasn't very good.",
                                "I hear you -- I was disappointed by %s, too.",
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
      # Short greeting message                                                    #
      #############################################################################

      greeting_message = "Let's get going!\nTell me about a movie that you have seen, I promise I won't judge."

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return greeting_message

    def goodbye(self):
      """chatbot goodbye message"""
      #############################################################################
      # Short farewell message                                                    #
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
        return "Sorry, I don't understand. " + random.choice(self.requestAnotherResponses)

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
            return "Sorry I couldn't understand that last movie title you gave me!\nLet's try again. " + random.choice(self.requestAnotherResponses)

          return self.checkArbitraryInput(input) + ' ' + random.choice(self.returnToMovies)

        withoutTitle = self.stripMovieTitle(input)
        return self.processMovieDetails(withoutTitle, movieDetails)

    ### Removes the movie title from the input string ###
    def stripMovieTitle(self, input):
      matchTokens = self.matchString.translate(None, string.punctuation).lower().strip().split()
      input = input.translate(None, string.punctuation).lower().strip().split()
      for i in range(len(matchTokens)):
        if matchTokens[i] in input:
          input.remove(matchTokens[i])
      return ' '.join(input)

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
        withoutTitle = raw_input(self.inputMarker + "How did you like '%s'?\n" % userMovie).strip()

      ### Classify sentiment ###
      sentiment = self.classifySentiment(withoutTitle)
      while (not sentiment):
        userSentimentInput = raw_input(self.inputMarker + 'I\'m sorry, I\'m not quite sure if you liked "%s".\nTell me more about "%s" e.g. \'I liked it\'\n' % (userMovie, userMovie))
        sentiment = self.classifySentiment(userSentimentInput)

      idx, movie = movieDetails
      self.userMovies[movieDetails] = sentiment
      print self.inputMarker + self.getSentimentResponse(sentiment, movie)

      ### We have collected a sufficient number of movies ###
      if len(self.userMovies) >= self.NUM_MOVIES_MIN:
        recommendation = self.recommend()
        response = ''
        if (self.FIRST_TIME):
          self.FIRST_TIME = False
          response = "Thank you for your patience, That's enough for me to make a recommendation.\n" + \
                    'I suggest you watch "%s".\n' % recommendation
        response += "Would you like to hear another recommendation? (yes/no)"

        ### Allow the user to ask for more recommendations ###
        print response
        response = raw_input().strip()
        while (self.getYesOrNo(response)):
          recommendation = self.recommend()
          print(self.inputMarker + 'I suggest you watch "%s".' % recommendation)
          print('Would you like to hear another recommendation? (yes/no)')
          response = raw_input().strip()
        return "OK. You can tell me more about your taste in movies (Or enter :quit if you're done)"

      else:
        response = random.choice(self.requestAnotherResponses)

      return response


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
          self.matchString = movieTitle
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

      if (re.search(self.DATE_REGEX, movieWordTokens[-1])): # check if last word is a date
        movieWordTokens = movieWordTokens[:-1]

      if movieWordTokens[0] in self.engArticlesLower: # check not empty and first word is an article e.g. American in Paris, An
        movie = ' '.join(movieWordTokens[1:]) + ', ' + movieWordTokens[0]

      movie = " ".join(movieWordTokens).strip()

  
      moviesInSeries = []
      for idx, movieDetails in enumerate(self.titles):
        title, genre = movieDetails
        if title in self.alreadyAsked: # We've already asked the user if this is the movie
          continue
        possibleMatch, userConfirmedMatch = self.isMatch(title, movie, idx) 
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
        print ('\n' if not failedLastTime else '') + self.inputMarker + "These are the movies I know that match %s" % userMovie + \
                          (" and %s" % filter_phrase if filter_phrase else '') + ':'
        
        for i in range(len(moviesInSeries)):
          print str(i + 1) + ".) " + str(moviesInSeries[i][1])

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

           
    ### EXTENSION 3: HANDLING MISPELLINGS ###
    ##################################################

    ### Checks for misspelling and series matches. Returns a (possibleMatch, userConfirmedMatch) ###
    ### tuple. If user explicitly confirms, userConfirmedMatch is set to be True, otherwise it   ###
    ### is set to False.                                                                         ###
    def isMatch(self, title, movie, idx):
      titleWithoutDate = self.titlesWithoutDate[idx]
      titleWithoutArticle = self.titlesWithoutArticle[idx]

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


    def removeYear(self, title):
      return re.sub(self.DATE_REGEX, '', title).strip()

    ### Returns True if the edit distance between w1 and w2 exceeds maxEditDist ###
    def editDistanceExceeds(self, w1, w2, maxEditDist):
      if abs(len(w2) - len(w1)) > maxEditDist:
        return True
      if (self.levenshtein(w1.lower(), w2.lower()) > maxEditDist):
        return True
      return False

    ### Computes the levenshtein distance between a and b ###
    def levenshtein(self, a, b):
      if len(a) < len(b):
        return self.levenshtein(b, a)
      if len(b) == 0:
        return len(a)

      a = np.array(tuple(a))
      b = np.array(tuple(b))

      prev = np.arange(b.size + 1)
      for s in a:
        curr = prev + 1
        curr[1:] = np.minimum(curr[1:], np.add(prev[:-1], b != s))
        curr[1:] = np.minimum(curr[1:], curr[0:-1] + 1)
        prev = curr

      return prev[-1]

    def getYesOrNo(self, input):
      while not input.lower().startswith('y') and not input.lower().startswith('n'):
        input = raw_input(self.inputMarker + "Please answer yes or no.\n").strip()
      return input.lower().startswith('y') # will be false if it starts with 'n'
    
    #############################################################################
    # SENTIMENT CLASSIFICATION                                                  #
    #############################################################################

    ### Classifies the sentiment of an input string ###
    def classifySentiment(self, input):
      stemmedInputArr = self.stem(input)
      count = 0
      neg_state = 1
      multiplier = 1
      for word in stemmedInputArr:
        if word in self.punctuation:
          neg_state = 1 # reset
          multiplier = 1

        elif any(neg in word for neg in self.negationTokens):
          neg_state *= -1

        elif word in self.intensifiers:
          multiplier *= self.INTENSIFIER_MULTIPLIER

        elif word in self.sentiment:
          if word in self.strongPosWords:
            count += multiplier * (self.STRONG_SENTIMENT_MULTIPLIER if neg_state > 0 else -self.STRONG_SENTIMENT_MULTIPLIER / 4) # 'not love' isn't as negative as 'hate'
            multiplier = 1
            continue
          if word in self.strongNegWords:
            count -= multiplier * (self.STRONG_SENTIMENT_MULTIPLIER if neg_state > 0 else -self.STRONG_SENTIMENT_MULTIPLIER / 4) # 'not hate' isn't as positive as 'love'
            multiplier = 1
            continue

          if self.sentiment[word] == 'pos':
            count += multiplier * neg_state
          else:
            count -= multiplier * neg_state
          multiplier = 1
        else:
          multiplier = 1

      if (count == 0):
        return 0 # unsure, ask for clarification

      if count > 0:
        return 5.0 if count >= 5 else 4.0

      return 1.0 if count <= -5 else 2.0

    def getSentimentResponse(self, sentiment, userMovie):
      if (sentiment == 5.0):
        return random.choice(self.veryPositiveResponses) % userMovie
      if (sentiment == 4.0):
        return random.choice(self.positiveResponses) % userMovie
      if (sentiment == 2.0):
        return random.choice(self.negativeResponses) % userMovie
      if (sentiment == 1.0):
        return random.choice(self.veryNegativeResponses) % userMovie

      return random.choice(self.positiveResponses) % userMovie # should never get here


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
      self.titlesWithoutDate = [self.removeYear(t).lower() for t, _ in self.titles]
      self.titlesWithoutArticle = [self.removeArticle(t).lower() for t in self.titlesWithoutDate]
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
          rating = 1 if self.userMovies[userMovieDetails] > self.BINARY_THRESH else -1
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

      Hello!

      I am Seb, an all wise and all knowing movie recomendation bot. I am here
      to help you decide what to do when the procrastination bug hits or to pass 
      the time on lonely Friday nights. I have a vast and diverse catalog of 
      films for your viewing pleasure -- just tell me some of the things you 
      like and we'll be on our way!

      My core features include:
      1) Identifying movies without quotation marks or perfect capitalization
      2) Fine-grained sentiment extraction
      3) Spell-checking movie titles
      4) Disambiguating movie titles for series and year ambiguities
      5) Responding to arbitrary input
      6) Speaking very fluently
      7) Using non-binarized dataset

      As well as partial support for:
      8) Understanding references to things said previously

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
