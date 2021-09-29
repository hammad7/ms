from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from rasa_sdk import Action
from rasa_sdk.events import SlotSet,FollowupAction,AllSlotsReset
import pandas as pd
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


ZomatoData = pd.read_csv('zomato.csv', encoding = "latin1")
ZomatoData = ZomatoData.drop_duplicates().reset_index(drop=True)
WeOperate = ['New Delhi', 'Gurgaon', 'Noida', 'Faridabad', 'Allahabad', 'Bhubaneshwar', 'Mangalore', 'Mumbai', 'Ranchi', 'Patna', 'Mysore', 'Aurangabad', 'Amritsar', 'Puducherry', 'Varanasi', 'Nagpur', 'Vadodara', 'Dehradun', 'Vizag', 'Agra', 'Ludhiana', 'Kanpur', 'Lucknow', 'Surat', 'Kochi', 'Indore', 'Ahmedabad', 'Coimbatore', 'Chennai', 'Guwahati', 'Jaipur', 'Hyderabad', 'Bangalore', 'Nashik', 'Pune', 'Kolkata', 'Bhopal', 'Goa', 'Chandigarh', 'Ghaziabad', 'Ooty', 'Gangtok', 'Shimla']
WeOperate = [c.lower() for c in WeOperate]

def RestaurantSearch(City,Cuisine,Price):
	print("=================="+str(Price)+City)
	TEMP = ZomatoData[(ZomatoData['Cuisines'].apply(lambda x: Cuisine.lower() in x.lower())) & (ZomatoData['City'].apply(lambda x: City.lower() in x.lower()))]
	Price.sort()
	if len(Price)==2:
		TEMP = TEMP[(TEMP['Average Cost for two'].apply(lambda x: int(x)>=int(Price[0]))) & (TEMP['Average Cost for two'].apply(lambda x: int(x)<=int(Price[1])))]
	elif len(Price)==1 and int(Price[0])<500:
		TEMP = TEMP[(TEMP['Average Cost for two'].apply(lambda x: int(x)<=int(Price[0])))]
	elif len(Price)==1 and int(Price[0])>=500:
		TEMP = TEMP[(TEMP['Average Cost for two'].apply(lambda x: int(x)>=int(Price[0])))]
	return TEMP[['Restaurant Name','Address','Average Cost for two','Aggregate rating']].sort_values(by=['Aggregate rating'], ascending=False)


def sendmail(mailid,response):
	#The mail addresses and password
	sender_address = 'alinarana071@gmail.com'
	sender_pass = ''
	#Setup the MIME
	message = MIMEMultipart()
	message['From'] = sender_address
	message['To'] = mailid
	message['Subject'] = 'Restaurants near you'
	#The body and the attachments for the mail
	message.attach(MIMEText("Hi,\n\t\t"+response + "\n\n Regards, \n Team NFL Foodies", 'plain'))
	#Create SMTP session for sending the mail
	session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
	session.starttls() #enable security
	session.login(sender_address, sender_pass) #login with mail_id and password
	text = message.as_string()
	session.sendmail(sender_address, mailid, text)
	session.quit()
	print('Mail Sent')


class ActionSearchRestaurants(Action):
	def name(self):
		return 'action_search_restaurants'

	def run(self, dispatcher, tracker, domain):
		#config={ "user_key":"f4924dc9ad672ee8c4f8c84743301af5"}
		loc = tracker.get_slot('location')
		cuisine = tracker.get_slot('cuisine')
		price = tracker.get_slot('price')
		results = RestaurantSearch(City=loc,Cuisine=cuisine,Price=price)
		response=""
		# if loc.lower() not in WeOperate:
		# 	response = "We do not operate in that area yet"
		if results.shape[0] == 0:
			response= "No restaurants found"
			dispatcher.utter_message(template="utter_ask_retry_price")
			dispatcher.utter_message(template="utter_ask_price")
			# dispatcher.utter_template("utter_email_error", tracker)
			return [SlotSet('price',None)]
		else:
			for restaurant in RestaurantSearch(loc,cuisine,price).iloc[:5].iterrows():
				restaurant = restaurant[1]
				response=response + F"Found {restaurant['Restaurant Name']} in {restaurant['Address']} rated {restaurant['Aggregate rating']} with avg cost for 2 Rs.{restaurant['Average Cost for two']} \n\n"
				
		dispatcher.utter_message("-----"+response)
		return [SlotSet('location',loc)]

class ActionSendMail(Action):
	def name(self):
		return 'action_send_mail'

	def run(self, dispatcher, tracker, domain):
		loc = tracker.get_slot('location')
		cuisine = tracker.get_slot('cuisine')
		price = tracker.get_slot('price')
		MailID = tracker.get_slot('email')
		response=""

		for restaurant in RestaurantSearch(loc,cuisine,price).iloc[:10].iterrows():
				restaurant = restaurant[1]
				response=response + F"Found {restaurant['Restaurant Name']} in {restaurant['Address']} rated {restaurant['Aggregate rating']} with avg cost for 2 Rs.{restaurant['Average Cost for two']} \n\n"

		sendmail(MailID,response)
		return [SlotSet('email',MailID)]


class ActionVerifyLocation(Action):
	def name(self):
		return 'action_verify_location'

	def run(self, dispatcher, tracker, domain):
		#config={ "user_key":"f4924dc9ad672ee8c4f8c84743301af5"}
		loc = tracker.get_slot('location')
		response=""
		if loc.lower() not in WeOperate:
			response = "We do not operate in that area yet"
			dispatcher.utter_message(template="utter_notier")
			return [SlotSet('location',None)]
		else:
			cuisine = tracker.get_slot('cuisine')
			price = tracker.get_slot('price')
			if cuisine is None:
				dispatcher.utter_message(template="utter_ask_cuisine")
			elif price is None:
				dispatcher.utter_message(template="utter_ask_price")
			else:
				return [FollowupAction("action_search_restaurants")]
				# dispatcher.utter_message(response="action_search_restaurants")
			return [SlotSet('location',loc)]


class ActionVerifyLocation(Action):
	def name(self):
		return 'action_reset_context'

	def run(self, dispatcher, tracker, domain):
		return [AllSlotsReset()]
