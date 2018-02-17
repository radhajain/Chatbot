#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA6, CS124, Stanford, Winter 2018
# v.1.0.2
#
# Original Python code by Ignacio Cases (@cases)
#
# Refs
# https://docs.python.org/3/library/cmd.html
# https://pymotw.com/2/cmd/
#
######################################################################

import cmd
import argparse
from chatbot import *

pa6_description = 'Simple Read-Eval-Print-Loop that handles the input/output part of the conversational agent'

class REPL(cmd.Cmd):
    """Simple REPL to handle the conversation with the chatbot."""
    is_turbo = False
    chatbot = Chatbot(is_turbo)
    name = chatbot.bot_name()
    bot_prompt = '\001\033[96m\002%s> \001\033[0m\002' % name
    prompt = '> '
    greeting = chatbot.greeting()
    intro = chatbot.intro() + '\n' + bot_prompt + greeting
    debug = False

    doc_header = ''
    misc_header = ''
    undoc_header = ''
    ruler = '-'

    def set_turbo(self, is_turbo = False):
      self.chatbot.is_turbo = is_turbo

    def cmdloop(self, intro=None):
      if self.debug == True:
        print 'cmdloop(%s)' % intro
      return cmd.Cmd.cmdloop(self, intro)

    def preloop(self):
      print self.header()
      self.debug_chatbot = False
      if self.debug == True:
        print 'preloop(); Chatbot %s created and loaded' % self.chatbot

    def postloop(self):
      if self.debug == True:
        print 'postloop()'

    def parseline(self, line):
      if self.debug == True:
        print 'parseline(%s) =>' % line,
        print 'applying function chat to line \'' + line + '\'...'
      [cmd, arg] = ['','']
      return [cmd, arg, line]

    def onecmd(self, s):
      if self.debug == True:
        print 'onecmd(%s)' % s
      if s:
        return cmd.Cmd.onecmd(self, s)
      else:
        return

    def emptyline(self):
      if self.debug == True:
        print 'emptyline()'
      return cmd.Cmd.emptyline(self)

    def default(self, line):
      if self.debug == True:
        print 'default(%s)' % line
      if line == ":quit":
        return True
      else:
        response = self.chatbot.process(line)
        print self.bot_says(response)

    def precmd(self, line):
      if self.debug == True:
        print 'precmd(%s)' % line
      return cmd.Cmd.precmd(self, line)

    def postcmd(self, stop, line):
      if self.debug == True:
        print 'postcmd(%s, %s)' % (stop, line)
      if line == ':quit':
        return True
      elif (line.lower() == 'who are you?'):
        self.secret(line)
      elif line == ':debug on':
        print 'enabling debug...'
        self.debug_chatbot = True
      elif line == ':debug off':
        print 'disabling debug...'
        self.debug_chatbot = False

      # Debugging the chatbot
      if self.debug_chatbot == True:
        print self.chatbot.debug(line)

      return cmd.Cmd.postcmd(self, stop, line)

    def secret(self, line):
      story = """A long time ago, in a remote land, a young developer named Alberto Caso managed to build an ingenious and mysterious chatbot... Now it's your turn!"""
      print story

    def do_prompt(self, line):
      "Change the interactive prompt"
      self.prompt = line + ': '

    def emptyline(self):
      if self.debug == True:
        print 'emptyline()'
      return cmd.Cmd.emptyline(self)

    def postloop(self):
      goodbye = self.chatbot.goodbye()
      print self.bot_says(goodbye)

    def bot_says(self, response):
      return self.bot_prompt + response

    def header(self):
      header ="""
  Welcome to Stanford CS124
     _______  _______         ___
    |       ||   _   |       |   |
    |    _  ||  |_|  | ____  |   |___
    |   |_| ||       ||____| |    _  |
    |    ___||       |       |   | | |
    |   |    |   _   |       |   |_| |
    |___|    |__| |__|       |_______|
     _______  __   __  _______  _______  _______  _______  _______  __
    |       ||  | |  ||   _   ||       ||  _    ||       ||       ||  |
    |       ||  |_|  ||  |_|  ||_     _|| |_|   ||   _   ||_     _||  |
    |       ||       ||       |  |   |  |       ||  | |  |  |   |  |  |
    |      _||       ||       |  |   |  |  _   | |  |_|  |  |   |  |__|
    |     |_ |   _   ||   _   |  |   |  | |_|   ||       |  |   |   __
    |_______||__| |__||__| |__|  |___|  |_______||_______|  |___|  |__|


      """
      return header

def process_command_line():
  parser = argparse.ArgumentParser(description=pa6_description)
  # optional arguments
  parser.add_argument('--turbo', dest='is_turbo', type=bool, default=False, help='Enables turbo mode')
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = process_command_line()
  repl = REPL()
  repl.set_turbo(args.is_turbo) # sigh, this is hacky -- we should be able to pass it directly to the constructor or initialization method, but there is an inheritance issue
  repl.cmdloop()
