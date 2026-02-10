#-*- coding:utf-8 -*-
import urllib, urllib2, cookielib, time, random, codecs

'''
Lang-8 Corpus Scraper
Originally written by Julian Brooke: http://www.cs.toronto.edu/~jbrooke/Lang8.zip
Adapted by Sabrina Stehwien
24.11.2014
'''


username = ''
password = ''

def get_stuff(page, url):

    '''
    index = page.find("'subject_show'>")
    if index == -1:
        return ""
    subject = page[index + 15:page.find("<", index)]
    index = page.find("'entry_time'>", index)
    if index == -1:
        #return ""
    time = page[index + 13:page.find("<", index)].strip()
    #index = page.find("<!-- google_ad_section_start -->", index)
    #if index == -1:
    #    return ""
    '''

    index = page.find("<div id='body_show_ori'>")
    if index == -1:
        print "content not found"
        return ""
    #content = page[index + 20: page.find("<!-- google_ad_section_end -->", index)].strip()
    content = page[index + 24: page.find("</div>", index)].strip()
    content = content.replace("<br/>", " ")

    '''
    index = page.find("<div id='comments_and_corrections_field'>")
    if index != -1:
        corrections = page[index:page.find("<script type='text/javascript'>",index)]
    else:
        corrections = ""      
    index = page.find("id='author_box'>", index)
    index = page.find('"user_name">', index)
    if index == -1:
        #return ""
    author = page[index + 12:page.find("<", index)]
    '''

    index = page.find("'Native language'>", index)
    if index == -1:
        return ""
    native_lang = page[index + 18: page.find("<", index)]

    #header = "|".join([url,author,native_lang,subject,time])
    header = " ||| ".join([native_lang,url])


    #return "!@#$%" + header + "\n" + content + "\n^^^^^\n" + corrections + "\n"
    return "^^DOC^^ ||| " + header + "\n" + content + "\n"

cj = cookielib.CookieJar()
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
login_data = urllib.urlencode({'username' : username, 'password' : password, 'User-Agent' : 'Mozilla/5.0 (X11; Linux x86_64; rv:33.1) Gecko/20100101 Firefox/33.1'})
opener.open('https://lang-8.com/sessions/new', login_data)
url_list = codecs.open("lang8_urls.txt", encoding="utf-8")
fentries = codecs.open("lang8_entries.txt", "w", encoding="utf-8")
failed = codecs.open("lang8_failed_urls.txt", "w", encoding="utf-8")
for line in url_list:
    url = line.strip()
    try:
        entry_page = opener.open(url)
        stuff = get_stuff(unicode(entry_page.read(), "utf-8"), url)
        fentries.write(stuff)
        print url
        print "succeeded"
    except:
        print url
        print "failed"
        failed.write(url+"\n")
    fentries.flush()
    failed.flush()
    time.sleep(4 + random.random()*6)
fentries.close()
failed.close()
    
