import argparse
import os
import time
import bs4
import re
import json
import pandas as pd
import scrapper

from urllib.request import urlopen
from urllib.error import HTTPError
from datetime import datetime
from pymongo import MongoClient  
from bson.json_util import dumps


def test_bad_url(soup):
	ret = scrapper.get_author_image_url(soup)
	assert(ret == "Missing Image Url")

def test_get_book_id(url):
	ret = scrapper.get_book_id(url)
	assert(ret == "3735293")

def test_get_rating(soup):
	ret = scrapper.get_rating(soup)
	assert(ret == '4.32')

def test_get_author_url(soup):
	ret = scrapper.get_author_url(soup)
	assert(ret == 'https://www.goodreads.com/author/show/48622.Erich_Gamma')


if __name__ == "__main__":

	bad_url = 'https://www.google.com/'
	src = urlopen(bad_url)
	soup = bs4.BeautifulSoup(src, 'html.parser')
	test_bad_url(soup)

	#
	good_url = 'https://www.goodreads.com/book/show/3735293-clean-code'
	test_get_book_id(good_url)

	#
	url = 'https://www.goodreads.com/book/show/4099'
	src = urlopen(url)
	soup = bs4.BeautifulSoup(src, 'html.parser')
	test_get_rating(soup)

	#
	url = 'https://www.goodreads.com/book/show/85009'
	src = urlopen(url)
	soup = bs4.BeautifulSoup(src, 'html.parser')
	test_get_author_url(soup)

	print("Unit Tests Passed.")