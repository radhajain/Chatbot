#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA 6, CS124, Stanford, Winter 2018
# v.1.0.2
######################################################################

import csv
import numpy as np

def ratings(src_filename='data/ratings.txt', delimiter='%', header=False, quoting=csv.QUOTE_MINIMAL):
	title_list = titles()
	user_id_set = set()
	with open(src_filename, 'r') as f:
	    content = f.readlines()
	    for line in content:
	        user_id = int(line.split(delimiter)[0])
	        if user_id not in user_id_set:
	            user_id_set.add(user_id)
	num_users = len(user_id_set)
	num_movies = len(title_list)
	mat = np.zeros((num_movies, num_users))

	reader = csv.reader(file(src_filename), delimiter=delimiter, quoting=quoting)
	for line in reader:
		mat[int(line[1])][int(line[0])] = float(line[2])
	return title_list, mat

def titles(src_filename='data/movies.txt', delimiter='%', header=False, quoting=csv.QUOTE_MINIMAL):
	reader = csv.reader(file(src_filename), delimiter=delimiter, quoting=quoting)
	title_list = []
	for line in reader:
		movieID, title, genres = int(line[0]), line[1], line[2]
		if title[0] == '"' and title[-1] == '"':
			title = title[1:-1]
		title_list.append([title, genres])
	return title_list
