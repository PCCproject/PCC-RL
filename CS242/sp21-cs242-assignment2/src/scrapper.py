import argparse
import os
import time
import bs4
import re
import json
import pandas as pd

from urllib.request import urlopen
from urllib.error import HTTPError
from datetime import datetime
from pymongo import MongoClient  
from bson.json_util import dumps


def get_similar_books(soup, max_num = 10):
    book_list = []

    url_similar_books = soup.find('a', {'class': 'actionLink right seeMoreLink'})['href']
    source = urlopen(url_similar_books)
    soup = bs4.BeautifulSoup(source, 'lxml')

    forbid_word = "href"
    sentence = soup.findAll('span', {'itemprop': 'name'})
    for line in sentence:  
        line = str(line)
        if forbid_word not in line:
            idx1 = line.find('>')
            #str.find(str, beg=0, end=len(string))
            idx2 = line.find('<',1)

            book_name = line[idx1+1:idx2]

            book_list += [book_name]

    #for boo in book_list:
    #    print(boo)
    return book_list


def get_related_authors(soup, author_url, max_num = 10):
    author_list = []

    url_similar_authors = author_url.replace("show", "similar")
    source = urlopen(url_similar_authors)
    soup = bs4.BeautifulSoup(source, 'lxml')

    sentence = soup.findAll('span', {'itemprop': 'name'})
    for line in sentence:  
        line = str(line)
        idx1 = line.find('>')
        idx2 = line.find('<',1)
        author_name = line[idx1+1:idx2]
        author_list += [author_name]

    #for boo in author_list:
    #    print(boo)
    return author_list



def get_author_books(soup, max_num = 10):
    book_list = []

    sentence = soup.findAll('span', {'itemprop': 'name', 'role': 'heading'})
    for line in sentence:  
        line = str(line)
        idx1 = line.find('>')
        idx2 = line.find('<',1)
        book_name = line[idx1+1:idx2]
        book_list += [book_name]

    #for boo in book_list:
    #    print(boo)
    return book_list

def get_isbn(soup):
    try:
        isbn = soup.find('meta', {'property': 'books:isbn'})['content']
        return isbn
    except:
        return "missing isbn"

'''

def get_isbn(soup):
    isbn = soup.find('meta', {'property': 'books:isbn'})['content']
    return isbn
'''    

def get_id(bookid):
    pattern = re.compile("([^.-]+)")
    return pattern.search(bookid).group()


def get_book_image_url(soup):
    partial_url = soup.find('a', {'itemprop': 'image'})['href']
    return 'https://www.goodreads.com' + partial_url


def get_author_image_url(soup):
    try:
        name = get_author_name(soup)
        partial_url = soup.find('a', {'title': name, 'rel': 'nofollow'})['href']
    except:
        return "Missing Image Url"
    return 'https://www.goodreads.com' + partial_url


def get_author_url(soup):
    name = get_author_name(soup)
    return soup.find('meta', {'property': 'books:author'})['content']


def get_author_id(author_url_str):
    idx1 = author_url_str.find('show/')
    idx2 = author_url_str.find('.',idx1)
    return author_url_str[idx1+5:idx2]


def get_author_name(soup):
    return soup.find('meta', {'property': 'og:title'})['content']


def get_rating(soup):
    return soup.find('span', {'itemprop': 'ratingValue'}).text.strip()


def get_next_book_url(book_url, visited_url_list):
    src = urlopen(book_url)
    soup = bs4.BeautifulSoup(src, 'html.parser')

    url_similar_books = soup.find('a', {'class': 'actionLink right seeMoreLink'})['href']
    source = urlopen(url_similar_books)
    soup = bs4.BeautifulSoup(source, 'lxml')

    sentence = soup.findAll('a', {'itemprop': 'url'})

    for line in sentence:  
        line = str(line)
        if ('/book/show/' in line):
            idx1 = line.find('/book')
            idx2 = line.find('itemprop')
            url = 'https://www.goodreads.com' + line[idx1:idx2-2]

            if(url in visited_url_list):
                continue

            return url

    return book_url


def get_next_author_url(author_url, visited_url_list):

    url_similar_authors = author_url.replace("show", "similar")
    source = urlopen(url_similar_authors)
    soup = bs4.BeautifulSoup(source, 'lxml')

    sentence = soup.findAll('a', {'itemprop': 'url'})
    for line in sentence:  
        line = str(line)
        if ('/author/show/' in line):
            idx1 = line.find('/author')
            idx2 = line.find('itemprop')
            url = 'https://www.goodreads.com' + line[idx1:idx2-2]
            if(url in visited_url_list):
                continue

            return url

    return author_url


def scrape_author_from_url(author_url):
    src = urlopen(author_url)
    soup = bs4.BeautifulSoup(src, 'html.parser')

    time.sleep(0.1)

    return {
            'name':                 get_author_name(soup),
            'author_url':           author_url,  
            'author_id':            get_author_id(author_url),      
            'rating':               get_rating(soup),
            'rating_count':         soup.find('span', {'itemprop': 'ratingCount'})['content'].strip(),
            'review_count':         soup.find('span', {'itemprop': 'reviewCount'})['content'].strip(),
            'image_url':            get_author_image_url(soup), 
            'related_authors':      get_related_authors(soup, author_url),
            'author_books':         get_author_books(soup)
            
            }


def scrape_author_from_book(book_id):
    url = 'https://www.goodreads.com/book/show/' + book_id
    src = urlopen(url)
    soup = bs4.BeautifulSoup(src, 'html.parser')

    author_url = get_author_url(soup)

    scrape_author_from_url(author_url)



def scrape_book(book_id):
    url = 'https://www.goodreads.com/book/show/' + book_id
    src = urlopen(url)
    soup = bs4.BeautifulSoup(src, 'html.parser')

    time.sleep(0.1)

    return {
            'book_url':             url,  
            'title':                ' '.join(soup.find('h1', {'id': 'bookTitle'}).text.split()),
            'book_id':              get_id(book_id),
            'isbn':                 get_isbn(soup),
            'author_url':           get_author_url(soup),
            'author':               ' '.join(soup.find('span', {'itemprop': 'name'}).text.split()),
            'rating':               get_rating(soup),
            'rating_count':         soup.find('meta', {'itemprop': 'ratingCount'})['content'].strip(),
            'review_count':         soup.find('meta', {'itemprop': 'reviewCount'})['content'].strip(),
            'image_url':            get_book_image_url(soup), 
            'similar_books':        get_similar_books(soup),
            }

def db_setup():
    # Set up database
    client = MongoClient('localhost', 27017)

    # Initialize a databse called db, and initialize two tables 'books' and 'authors'
    db = client['scraped_data']
    db_books = db['books']
    db_authors = db['authors']
    return client, db_books, db_authors


def db_export(collection, out_file):
    #with open('collection.json', 'w') as file:
    cursor = collection.find({})
    with open(out_file, "w") as file:
        file.write('[')
        for document in cursor:
            file.write(dumps(document))
            file.write(',')
        file.write(']')

def db_import(collection, in_file):
    #with open('data.json') as file: 
    with open(in_file) as file: 
        file_data = json.load(file) 
          
    # if JSON contains data more than one entry 
    # insert_many is used else inser_one is used 
    if isinstance(file_data, list): 
        Collection.insert_many(file_data)   
    else: 
        Collection.insert_one(file_data) 

def get_book_id(url):
    idx1 = url.find('show/')
    idx2 = url.find('-')
    if(idx2 == -1):
        idx2 = url.find('.', idx1)
    book_id = url[idx1+5:idx2]
    return book_id


# scrap data from starting url and store json data in the specified output directory
# example command: python scrapper.py --url https://www.goodreads.com/book/show/3735293-clean-code --num_books 5 --num_authors 5 --out_dir scraped_data
def main():

    start_time = datetime.now()
    script_name = os.path.basename(__file__)

    parser = argparse.ArgumentParser()

    parser.add_argument('--url', type=str, help="the start url for scrapping")
    parser.add_argument('--num_books', type=int, default=200, help="number of books to scrape")
    parser.add_argument('--num_authors', type=int, default=50, help="number of authors to scrape")
    parser.add_argument('--out_dir', default='scraped_data', type=str)

    args = parser.parse_args()

    substr_goodreads = 'goodreads.com'
    substr_book = 'goodreads.com/book/show/'

    if(substr_goodreads in args.url):
        if(substr_book in args.url):
            pass
        else:
            exit("ERROR: input url does not represent a book.")
    else:
        exit("ERROR: input url does not point to the GoodReads website.")

    if(args.num_books > 200):
        print('WARNING: number of books greater than 200!')

    if(args.num_authors > 50):
        print('WARNING: number of authors greater than 50!')

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    MAX_ITER = 1998
    max_iter_counter = 0

    ###############################    

    client, db_books, db_authors = db_setup()

    ################################

    idx1 = args.url.find('show/')
    idx2 = args.url.find('-')
    if(idx2 == -1):
        idx2 = next_url.find('.', idx1)
    book_id = args.url[idx1+5:idx2]

    next_url = 'https://www.goodreads.com/book/show/' + book_id
    visited_url_list = [args.url]

    for i in range(args.num_books):
        try:
            print('At time ' + str(datetime.now()) + ' ' + ': Scraping Book ' + book_id + '...')

            book_info = scrape_book(book_id)
            #db_books.insert(book_info)

            print('Done scraping ' + str(i+1) + ' / ' + str(args.num_books) + ' books')            

            json.dump(book_info, open(args.out_dir + '/' + 'book:' + book_id + '.json', 'w'))

            next_url = get_next_book_url(next_url, visited_url_list)

            idx1 = next_url.find('show/')
            idx2 = next_url.find('-')
            if(idx2 == -1):
                idx2 = next_url.find('.', idx1)
            book_id = next_url[idx1+5:idx2]
            visited_url_list += [next_url]

            max_iter_counter += 1
            if (max_iter_counter >= MAX_ITER):
                exit("Too much scraping in one go, stop early.")
            print('=============================')

        except HTTPError as e:
            print(e)
            exit(0)

    #############################

    idx1 = args.url.find('show/')
    idx2 = args.url.find('-')
    if(idx2 == -1):
        idx2 = next_url.find('.', idx1)
    book_id = args.url[idx1+5:idx2]
    url = 'https://www.goodreads.com/book/show/' + book_id
    src = urlopen(url)
    soup = bs4.BeautifulSoup(src, 'html.parser')
    next_author_url = get_author_url(soup)
    visited_url_list = [next_author_url]
    
    for i in range(args.num_authors):
        try:
            author_id = get_author_id(next_author_url)
            print('At time ' + str(datetime.now()) + ' ' + ': Scraping Author ' + author_id + '...')
        
            author_info = scrape_author_from_url(next_author_url)
            #db_authors.insert(author_info)
            
            print('Done scraping ' + str(i+1) + ' / ' + str(args.num_authors) + ' authors')

            json.dump(author_info, open(args.out_dir + '/' + 'author:' + author_id + '.json', 'w'))

            next_author_url = get_next_author_url(next_author_url , visited_url_list)

            visited_url_list += [next_author_url]

            max_iter_counter += 1
            if (max_iter_counter >= MAX_ITER):
                exit("Too much scraping in one go, stop early.")

            print('=============================')

        except HTTPError as e:
            print(e)
            exit(0)

    ###############################
    
    #db_export(db_books, 'books.json')
    #db_export(db_authors, 'authors.json')

    print('At time ' + str(datetime.now()) + ' ' + script_name + f': \nAll Data Successfully Scraped. \n Outputs are stored in the JSON files in the output directory.\n')

    client.close()

if __name__ == '__main__':
    main()

