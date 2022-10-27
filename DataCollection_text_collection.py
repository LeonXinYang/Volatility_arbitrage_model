import urllib
import urllib.parse
import urllib.request
import json
import ssl
import time
import random
import urllib.request
import zlib
import urllib.request
import ssl
import os
from bs4 import BeautifulSoup
import socket
"""
Stage: Stage 1 file

Document type: Web Crawler + Data collection only.

Main purpose: 1. To get latest conference call files  
              2. To get the conference call in the SP500 only
              3. To capture the audio in the file.

Need to run? No.

Dependency:
    use -> None
    be used -> DataCleaning_audio_collection.py

Methods:
    Part 1: 
    
    get_proxy_vpn(): To build up the proxy to get the vpn website request
    get_proxy(): To get the proxy connection
    build proxy(): To build up the proxy connection
    build_proxy_vpn(): To build up the vpn connection
    get_specific_json_article(num,opener,code): to download the article
    get_the_list_of_article(page,opener): get the list of article
    retrive_page_info(page): get the article number in the json file
    try_get_list_of_article(page,opener): To keep 'get_the_list_of_article' connected until it get the answer
    try_get_json_article(each_article_num,opener): To keep 'get_specific_json_article' connected until it get the answer
    main(): To get pages of files
    
    Part 2: 
    
    get_the_list_of_article_sp500(code,opener,page): get the articles number of transcript in SP500 companies
    try_get_list_of_article_sp500(code,page): To keep get_the_list_of_article_sp500 connected
    try_get_json_article_sp500(each_article_num,code): To keep get_json_article_sp500 connected
    
    Part 3:
    get_the_sp500_mp3: to download the corresponding audio file
    try_get_sp500_mp3: keep downloading the audio file
"""
my_headers = [
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
    "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)",
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
    'Opera/9.25 (Windows NT 5.1; U; en)',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
    'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
    'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
    'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
    "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7",
    "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 "
]
rest = range(1,2)
def get_proxy_vpn():
    """
    Get the VPN connection
    :return: /
    """
    ssl._create_default_https_context = ssl._create_unverified_context

    tunnel = "tps238.kdlapi.com:15818"

    username = "t14025235199951"
    password = "r8kw17ev"
    proxies = {
        "http": "http://%(user)s:%(pwd)s@%(proxy)s/" % {"user": username, "pwd": password, "proxy": tunnel},
        "https": "http://%(user)s:%(pwd)s@%(proxy)s/" % {"user": username, "pwd": password, "proxy": tunnel}
    }
    return proxies

def get_proxy():
    """
    Get the proxy connection
    :return:
    """
    #api_url = "http://dps.kdlapi.com/api/getdps/?orderid=944018809070266&num=1&pt=1&sep=1"
    api_url = "http://dev.kdlapi.com/api/getproxy/?orderid=924025140987962&num=1&protocol=2&method=1&quality=0&sep=1"

    headers = {"Accept-Encoding": "Gzip"}

    req = urllib.request.Request(url=api_url, headers=headers)

    res = urllib.request.urlopen(req)

    content_encoding = res.headers.get('Content-Encoding')
    if content_encoding and "gzip" in content_encoding:
        return zlib.decompress(res.read(), 16 + zlib.MAX_WBITS).decode('utf-8')
    else:
        return res.read().decode('utf-8')

def build_proxy():
    """
    Build up the proxy connection
    :return:
    """
    proxy = get_proxy()
    proxy_values = "%(ip)s"%{'ip':proxy}
    proxies = {"http":proxy_values, "https": proxy_values}
    urlhandle = urllib.request.ProxyHandler(proxies)
    opener = urllib.request.build_opener(urlhandle)
    return opener

def build_proxy_vpn():
    """
    Build up the proxy VPN connection
    :return:
    """
    proxies = get_proxy_vpn()
    urlhandle = urllib.request.ProxyHandler(proxies)
    opener = urllib.request.build_opener(urlhandle)
    return opener

def get_specific_json_article(num,opener,code):
    """
    Capture the json file in the website
    :param num: The file number
    :param opener: website opener
    :param code: The company code
    :return: download the file
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    url = 'https://seekingalpha.com/api/v3/articles/' + str(num)
    referer = 'https://seekingalpha.com/article/' + str(num)
    herders = {
        'User-Agent': random.choice(my_headers),
        'Referer': referer, 'Connection': 'keep-alive'}
    #opener = build_proxy()
    req = urllib.request.Request(url, headers=herders)
    response = opener.open(req,timeout=10)
    hjson = json.loads(response.read())
    address = "JsonSp500/" + str(code)+"/"+ str(num) + ".json"
    json.dump(hjson, open(address, "w"))


def get_the_list_of_article(page,opener):
    """
    Get the list of article in sepcific page
    :param page: page
    :param opener: website opener
    :return: The article number in the page
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://seekingalpha.com/author/sa-transcripts/ajax_load_regular_articles?lt=9&page=" + str(page) + "&author=true&userId=101639&sort=recent&rta=true"

    herders = {
        'User-Agent': random.choice(my_headers),
        'Referer': 'https://seekingalpha.com/author/sa-transcripts', 'Connection': 'keep-alive'}
    #opener = build_proxy()
    req = urllib.request.Request(url, headers=herders)
    response = opener.open(req,timeout=10)
    hjson = json.loads(response.read())
    output = retrive_page_info(hjson)
    address = "JsonFile/page" + str(page) + ".json"
    json.dump(hjson, open(address, "w"))
    return output;


def retrive_page_info(page):
    """
    get the article number in the json file
    :param page: the number of page
    :return: set of files number
    """
    # page = json.load(open("JsonFile/page1.json"))
    soup = BeautifulSoup(page['html_content'], 'html.parser')
    a = set()
    for each in soup.find_all("a"):
        if (each.get('href')[0:9] == "/article/") and (each.get('href').find("earning") != -1) and (
                each.get('href').find("presentation") == -1):
            a.add(each.get('href')[9:16])
            # print(each.get('href'))
    print(a)
    return a

def try_get_list_of_article(page,opener):
    """
    To keep 'get_the_list_of_article' connected until it get the answer
    """
    while True:
        try:
            return get_the_list_of_article(page,build_proxy_vpn())
        except Exception as e:
            print(e)
            print("another 403! list")


def try_get_json_article(each_article_num,opener):
    """
    To keep 'get_specific_json_article' connected until it get the answer
    """
    while True:
        try:
            return get_specific_json_article(each_article_num,build_proxy_vpn())
        except Exception as e:
            print(e)
            print("another 403! article")

def main(pages):
    """
    To get pages of files
    :param pages: number of pages to stop searching
    :return: to get files from page 1 to page pages
    """
    i = 0
    opener = build_proxy()
    for page in range(1, pages):
        output_set = try_get_list_of_article(page,opener)
        for each_article_num in output_set:
            try_get_json_article(each_article_num,opener)
            time.sleep(random.choice(rest))
            i += 1
            print("complete",i)

        time.sleep(random.choice(rest))

def get_the_list_of_article_sp500(code,opener,page):
    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://seekingalpha.com/author/sa-transcripts/ajax_load_regular_articles?author=true&userId=101639&q[]=" + str(code) + "&sort=recent&rta=true&page="+ str(page)

    herders = {
        'User-Agent': random.choice(my_headers),
        'Referer': 'https://seekingalpha.com/author/sa-transcripts', 'Connection': 'keep-alive'}
    #opener = build_proxy()
    req = urllib.request.Request(url, headers=herders)
    response = opener.open(req,timeout=10)
    hjson = json.loads(response.read())
    all_pages = hjson["page_count"]
    output = retrive_page_info(hjson)
    address = "JsonSp500/" + str(code) + "/page" + str(page) + ".json"
    if not os.path.exists("JsonSp500/"+str(code)):
        os.makedirs("JsonSp500/"+str(code))
    json.dump(hjson, open(address, "w"))
    return output,all_pages

def try_get_list_of_article_sp500(code,page):
    while True:
        try:
            return get_the_list_of_article_sp500(code,build_proxy_vpn(),page)
        except Exception as e:
            print(e)
            print("another 403! list")


def try_get_json_article_sp500(each_article_num,code):
    while True:
        try:
            return get_specific_json_article(each_article_num,build_proxy_vpn(),code)
        except Exception as e:
            print(e)
            print("another 403! article")

def main1():
    for root, dirs, files in os.walk("stock_market_data/sp500/csv"):
        for each in files:
            if not os.listdir("JsonSp500").__contains__(each[:-4]):
                print(each)
                list = set()
                list_and_page = try_get_list_of_article_sp500(each[:-4],1)
                list.update(list_and_page[0])
                print("total page: " + str(list_and_page[1]))
                if list_and_page[1] > 1:
                    for page in range(2,min(6,list_and_page[1]+1)):
                        print(page)
                        list.update(try_get_list_of_article_sp500(each[:-4],page)[0])
                        time.sleep(random.choice(rest))
                print("total:" +str(list))
                for num in list:
                    print(num)
                    try_get_json_article_sp500(num,each[:-4])
                    time.sleep(random.choice(rest))


#This method is to capture the mp3 from the conference call
def callbackfunc(blocknum, blocksize, totalsize):
  '''call back function
  @blocknum: already download
  @blocksize: size of data
  @totalsize: size of data
  '''
  percent = 100.0 * blocknum * blocksize / totalsize
  if percent > 100:
    percent = 100
  print("%.2f%%"% percent)
def get_the_sp500_mp3(opener, each, address):
    socket.setdefaulttimeout(5)
    ssl._create_default_https_context = ssl._create_unverified_context
    #url= "https://static.seekingalpha.com/cdn/s3/transcripts_audio/4300690.mp3"
    herders = {
        'User-Agent': random.choice(my_headers),
        'Referer': 'https://seekingalpha.com/author/sa-transcripts', 'Connection': 'keep-alive'}
    #urllib.request.build_opener().addheaders = herders
    #urllib.request.install_opener(opener)
    name = address[address.rfind("/")+1:]
    urllib.request.urlretrieve(address, "/Volumes/My Passport/Research Data/SP500_stopword_semantics/"+each+"/"+name,callbackfunc)
    return


def try_get_sp500_mp3(opener,each,address):
    i = 0
    t = time.time()
    delta = 0
    while i < 20 or delta < 60:
        delta = time.time() - t
        try:
            get_the_sp500_mp3(opener,each, address)
            onecheck = json.load(open("/Volumes/My Passport/Research Data/SP500Audio/CheckComplete.json"))
            if(onecheck.__contains__(each)):
                onecheck[each].append(address[address.rfind("/")+1:])
                json.dump(onecheck,open("/Volumes/My Passport/Research Data/SP500Audio/CheckComplete.json","w"))
            else:
                for eachs in onecheck:
                    if(onecheck[eachs][-1] != "completed"):
                        onecheck[eachs].append("completed")
                onecheck[each] = [address[address.rfind("/")+1:]]
                json.dump(onecheck,open("/Volumes/My Passport/Research Data/SP500Audio/CheckComplete.json","w"))
            return
        except Exception as e:
            print(e)
            print("error")
            i += 1
    name = address[address.rfind("/")+1:]
    check = json.load(open("/Volumes/My Passport/Research Data/SP500Audio/UnComplete.json"))
    check.append(each)
    check.append(name)
    json.dump(check,open("/Volumes/My Passport/Research Data/SP500Audio/UnComplete.json","w"))



