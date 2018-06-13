from bs4 import BeautifulSoup
from bs4.element import Comment
from urllib.request import Request, urlopen

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    untokenized = []
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    for i in visible_texts:
        if (i != '\n' and len(i) > 10):
            untokenized.append(i)
    return untokenized


def links_to_corpus(links):
    
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}
    
    sentences = []

    for i in links:
        """
        try:
            urllib2.urlopen(i, headers={'User-Agent': 'Mozilla/5.0'})
        except urllib2.HTTPerror as e:
            print(e)
            continue
        """
        print(i)
        
        req = Request(i, headers=hdr)
        web_byte = urlopen(req).read()
        html = web_byte.decode('utf-8')

        untokenized = text_from_html(html)
    
        words_total = []
        
        for j in untokenized:
            words = j.split()
            words_total += words
        sentences.append(words_total)
        
    return sentences

def links_to_text(links):
    corpus=links_to_corpus(links)
    corpus=[' '.join(page) for page in corpus]
    return ' '.join(corpus)








    