## complete path
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "varanasi"}
    - slot{"location": "varanasi"}
    - action_verify_location
    - slot{"location": "varanasi"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "chinese"}
    - slot{"cuisine": "chinese"}
    - utter_ask_price
* restaurant_search{"price": "300"}
    - slot{"price": ["300"]}
    - action_search_restaurants
    - slot{"location": "varanasi"}
    - utter_ask_mailer
* affirm
    - utter_ask_email
* restaurant_search{"email": "aman3572@gmail.com"}
    - slot{"email": "aman3572@gmail.com"}
    - action_send_mail
    - slot{"email": "aman3572@gmail.com"}
    - utter_mail_sent
    - utter_goodbye
    - action_reset_context
    - reset_slots

* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "Delhi"}
    - slot{"location": "Delhi"}
    - action_verify_location
    - slot{"location": "Delhi"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "American"}
    - slot{"cuisine": "American"}
    - utter_ask_price
* restaurant_search{"price": "700"}
    - slot{"price": ["300", "700"]}
    - action_search_restaurants
    - slot{"location": "Delhi"}
    - utter_ask_mailer
* negate
    - utter_goodbye
    - action_reset_context
    - reset_slots

* restaurant_search{"location": "New Delhi"}
    - slot{"location": "New Delhi"}
    - action_verify_location
    - slot{"location": "New Delhi"}
* restaurant_search{"cuisine": "chinese"}
    - slot{"cuisine": "chinese"}
    - utter_ask_price
* restaurant_search{"price": "700"}
    - slot{"price": ["300", "700"]}
    - action_search_restaurants
    - slot{"location": "New Delhi"}
    - utter_ask_mailer
* affirm
    - utter_ask_email
* restaurant_search{"email": "aman3572@gmail.com"}
    - slot{"email": "aman3572@gmail.com"}
    - action_send_mail
    - slot{"email": "aman3572@gmail.com"}
    - utter_mail_sent
    - utter_goodbye
    - action_reset_context
    - reset_slots


* greet
    - utter_ask_howcanhelp
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "mumbai"}
    - slot{"location": "mumbai"}
    - action_verify_location
    - slot{"location": "mumbai"}
* restaurant_search{"cuisine": "american"}
    - slot{"cuisine": "american"}
    - utter_ask_price
* restaurant_search{"price": "700"}
    - slot{"price": ["700"]}
    - action_search_restaurants
    - slot{"location": "mumbai"}
    - utter_ask_mailer
* restaurant_search{"email": "aman3572@gmail.com"}
    - slot{"email": "aman3572@gmail.com"}
    - action_send_mail
    - slot{"email": "aman3572@gmail.com"}
    - utter_mail_sent
    - utter_goodbye
    - action_reset_context
    - reset_slots

* restaurant_search{"cuisine": "italian", "location": "pune", "price": "700"}
    - slot{"cuisine": "italian"}
    - slot{"location": "pune"}
    - slot{"price": ["300", "700"]}
    - action_search_restaurants
    - slot{"location": "pune"}
    - utter_ask_mailer
* affirm
    - utter_ask_email
* restaurant_search{"email": "aman3572@gmail.com"}
    - slot{"email": "aman3572@gmail.com"}
    - action_send_mail
    - slot{"email": "aman3572@gmail.com"}
    - utter_mail_sent
    - utter_goodbye
    - action_reset_context
    - reset_slots


* greet
    - utter_ask_howcanhelp
* restaurant_search{"price": "300"}
    - slot{"price": ["300"]}
    - utter_ask_location
* restaurant_search{"location": "aligarh"}
    - slot{"location": "aligarh"}
    - action_verify_location
    - slot{"location": null}
* restaurant_search{"location": "indore"}
    - slot{"location": "indore"}
    - action_verify_location
    - slot{"location": "indore"}
* restaurant_search{"cuisine": "South Indian"}
    - slot{"cuisine": "South Indian"}
    - action_search_restaurants
    - slot{"location": "indore"}
    - utter_ask_mailer
* negate
    - utter_goodbye
    - action_reset_context
    - reset_slots


* restaurant_search{"cuisine": "american"}
    - slot{"cuisine": "american"}
    - utter_ask_location
* restaurant_search{"location": "ooty"}
    - slot{"location": "ooty"}
    - action_verify_location
    - slot{"location": "ooty"}
* restaurant_search{"price": "700"}
    - slot{"price": ["700"]}
    - action_search_restaurants
    - slot{"location": "ooty"}
    - utter_ask_mailer
* negate
    - utter_goodbye
    - action_reset_context
    - reset_slots


* greet
    - utter_ask_howcanhelp
* restaurant_search{"price": "300", "cuisine": "chinese", "location": "mumbai"}
    - slot{"cuisine": "chinese"}
    - slot{"location": "mumbai"}
    - slot{"price": ["300"]}
    - action_search_restaurants
    - slot{"location": "mumbai"}
    - utter_ask_mailer
* affirm
    - utter_ask_email
* restaurant_search{"email": "aman3572@gmail.com"}
    - slot{"email": "aman3572@gmail.com"}
    - action_send_mail
    - slot{"email": "aman3572@gmail.com"}
    - utter_mail_sent

* restaurant_search{"price": "700", "cuisine": "american"}
    - slot{"cuisine": "american"}
    - slot{"price": ["700"]}
    - utter_ask_location
* restaurant_search{"location": "kanpur"}
    - slot{"location": "kanpur"}
    - action_verify_location
    - followup{"name": "action_search_restaurants"}
    - action_search_restaurants
    - slot{"location": "kanpur"}
    - utter_ask_mailer
* negate
    - utter_goodbye
    - action_reset_context
    - reset_slots

* restaurant_search{"location": "vishakhapatnam"}
    - slot{"location": "vishakhapatnam"}
    - action_verify_location
    - slot{"location": null}
* restaurant_search{"location": "chennai"}
    - slot{"location": "chennai"}
    - action_verify_location
    - slot{"location": "chennai"}
* restaurant_search{"cuisine": "Italian"}
    - slot{"cuisine": "Italian"}
    - utter_ask_price
* restaurant_search{"price": "300"}
    - slot{"price": ["300"]}
    - action_search_restaurants
    - slot{"location": "chennai"}
    - utter_ask_mailer
* affirm
    - utter_ask_email
* restaurant_search{"email": "aman3572@gmail.com"}
    - slot{"email": "aman3572@gmail.com"}
    - action_send_mail
    - slot{"email": "aman3572@gmail.com"}
    - utter_mail_sent
    - utter_goodbye
    - action_reset_context
    - reset_slots

* restaurant_search{"cuisine": "american", "location": "mumbai"}
    - slot{"cuisine": "american"}
    - slot{"location": "mumbai"}
    - action_verify_location
    - slot{"location": "mumbai"}
* restaurant_search{"price": "700"}
    - slot{"price": ["700"]}
    - action_search_restaurants
    - slot{"location": "mumbai"}
    - utter_ask_mailer
* restaurant_search{"email": "aman3572@gmail.com"}
    - slot{"email": "aman3572@gmail.com"}
    - action_send_mail
    - slot{"email": "aman3572@gmail.com"}
    - utter_mail_sent
    - utter_goodbye
    - action_reset_context
    - reset_slots
* restaurant_search{"cuisine": "chinese"}
    - slot{"cuisine": "chinese"}
    - utter_ask_location
* restaurant_search{"location": "bareilly"}
    - slot{"location": "bareilly"}
    - action_verify_location
    - slot{"location": null}
* restaurant_search{"location": "ooty"}
    - slot{"location": "ooty"}
    - action_verify_location
    - slot{"location": "ooty"}
* restaurant_search{"price": "700"}
    - slot{"price": ["500", "700"]}
    - action_search_restaurants
    - slot{"location": "ooty"}
    - utter_ask_mailer
* negate
    - utter_goodbye
    - action_reset_context
    - reset_slots
* goodbye
    - utter_goodbye

* greet
    - utter_ask_howcanhelp
* goodbye
    - utter_goodbye
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "New Delhi"}
    - slot{"location": "New Delhi"}
    - action_verify_location
    - slot{"location": "New Delhi"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "North Indian"}
    - slot{"cuisine": "North Indian"}
    - utter_ask_price
* restaurant_search{"price": "10000"}
    - slot{"price": ["10000"]}
    - action_search_restaurants
    - slot{"price": null}
* restaurant_search{"price": "500"}
    - slot{"price": ["500"]}
    - action_search_restaurants
    - slot{"location": "New Delhi"}
    - utter_ask_mailer
* negate
    - utter_goodbye
    - action_reset_context
    - reset_slots

* restaurant_search{"cuisine": "chinese", "location": "chennai", "price": "100000"}
    - slot{"cuisine": "chinese"}
    - slot{"location": "chennai"}
    - slot{"price": ["100000"]}
    - action_verify_location
    - followup{"name": "action_search_restaurants"}
    - action_search_restaurants
    - slot{"price": null}
* restaurant_search{"price": "250"}
    - slot{"price": ["250"]}
    - action_search_restaurants
    - slot{"price": null}
* restaurant_search{"price": "700"}
    - slot{"price": ["700"]}
    - action_search_restaurants
    - slot{"location": "chennai"}
    - utter_ask_mailer
* affirm
    - utter_ask_email
* restaurant_search{"email": "aman3572@gmail.com"}
    - slot{"email": "aman3572@gmail.com"}
    - action_send_mail
    - slot{"email": "aman3572@gmail.com"}
    - utter_mail_sent
    - utter_goodbye
    - action_reset_context
    - reset_slots
