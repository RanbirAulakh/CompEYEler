import requests
import os
import re
import sys
from bs4 import BeautifulSoup
from sys import platform
from urllib.request import urlretrieve

userId = sys.argv[1]
# term = sys.argv[2]

s = requests.session()
r = s.post('https://myaccess.rit.edu/myAccess5/process_login.php', data={"un":userId,"pw":"", "signin":""})
c = (r.content)

soup = BeautifulSoup(c, "html.parser")

for term in ['2121', '2125', '2131', '2135', '2141', '2145', '2151', '2155', '2161', '2165', '2171', '2175']:
	print("Term: " + term)
	ajax_url = r'https://myaccess.rit.edu/myAccess5/home_ajax_classes.php?term=' + term

	r = s.get(ajax_url)
	c = r.content
	soup = BeautifulSoup(c, "html.parser")

	for i in soup.findAll("h4"):
		print(i.text.strip().replace("View Notes", ""))

	print("===================")





