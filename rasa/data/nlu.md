## intent:affirm
- yes
- yep
- yeah
- indeed
- that's right
- ok
- great
- right, thank you
- correct
- great choice
- sounds really good
- thanks

## intent:negate
- no
- no need
- not required
- nopes
- nope
- no required

## intent:goodbye
- bye
- goodbye
- good bye
- stop
- end
- farewell
- Bye bye
- have a good one

## intent:greet
- hey
- howdy
- hey there
- hello
- hi
- good morning
- good evening
- dear sir
- Hey

## intent:restaurant_search
- i'm looking for a place to eat
- I want to grab lunch
- I am searching for a dinner spot
- I am looking for some restaurants in [Delhi](location).
- I am looking for some restaurants in [Bangalore](location)
- show me [chinese](cuisine) restaurants
- show me [chines]{"entity": "cuisine", "value": "chinese"} restaurants in the [New Delhi]{"entity": "location", "value": "Delhi"}
- show me a [mexican](cuisine) place in the [centre](location)
- i am looking for an [indian](cuisine) spot called olaolaolaolaolaola
- search for restaurants
- anywhere in the [west](location)
- I am looking for [asian fusion](cuisine) food
- I am looking a restaurant in [294328](location)
- in [Gurgaon](location)
- [South Indian](cuisine)
- [North Indian](cuisine)
- [Italian](cuisine)
- [Chinese]{"entity": "cuisine", "value": "chinese"}
- [chinese](cuisine)
- [Lithuania](location)
- Oh, sorry, in [Italy](location)
- in [delhi](location)
- I am looking for some restaurants in [Mumbai](location)
- I am looking for [mexican indian fusion](cuisine)
- can you book a table in [rome](location) in a [moderate]{"entity": "price", "value": "mid"} price range with [british](cuisine) food for [four]{"entity": "people", "value": "4"} people
- [central](location) [indian](cuisine) restaurant
- please help me to find restaurants in [pune](location)
- Please find me a restaurantin [bangalore](location)
- [mumbai](location)
- show me restaurants
- please find me [chinese](cuisine) restaurant in [delhi](location)
- can you find me a [chinese](cuisine) restaurant
- [delhi](location)
- please find me a restaurant in [ahmedabad](location)
- please show me a few [italian](cuisine) restaurants in [bangalore](location)
- i want food
- where can i have food
- [my email is xyz@gmail.com]{"entity": "email", "value": "xyz@gmail.com"}
- [ham@ymail.com]{"entity": "email", "value": "ham@mail.com"}
- [agra](location)
- [Mexican](cuisine)
- [sfwf@gmail.com](email)
- i want food places
- [kolkata](location)
- [pune](location)
- send me at [aman3572@gmail.com](email)
- [south indian](cuisine) food near [goa](location)
- i want to eat [mexican](cuisine) at [bihar](location)
- [new delhi](location)
- my email is [aman3572@gmail.com](email)
- show me restaurants in [kashmir](location)
- [amritsar](location)
- where can i find food
- [varanasi](location)
- less than Rs [300](price)
- reach me at [aman3572@gmail.com](email)
- i am hungry
- [American](cuisine)
- Rs. [300](price) to [700](price)
- [dilli]{"entity": "location", "value": "New Delhi"}
- show me restaurants in [dehli]{"entity": "location", "value": "New Delhi"}
- [aman3572@gmail.com](email)
- restaurants in [delhi]{"entity": "location", "value": "New Delhi"}
- show me retaurants
- [american](cuisine)
- More than [700](price)
- my id is [aman3572@gmail.com](email)
- show me [italian](cuisine) restaurants in [pune](location) between [300](price)-[700](price)
- I want food with the cost below Rs [300](price)
- [aligarh](location)
- [indore](location)
- can you help me find out [american](cuisine) restaurants
- [ooty](location)
- more than [700](price)
- share list of [cheap]{"entity": "price", "value": "300"} [chinese](cuisine) restaurants in [mumbai](location)
- mail me at [aman3572@gmail.com](email)
- show me [costy]{"entity": "price", "value": "700"} [american](cuisine) restaurants
- show me [costy]{"entity": "price", "value": "700"} [american](cuisine) restaurant
- [kanpur](location)
- restaurant in [vishakhapatnam](location)
- [chennai](location)
- <[300](price)
- show me [american](cuisine) food cafeteria in [mumbai](location)
- >[700](price)
- yes send it to [aman3572@gmail.com](email)
- can you also share [chinese](cuisine) restaurants
- [bareilly](location)
- [500](price) to [700](price)
- share retaurants
- anywhere in [india](location) like [delhi]{"entity": "location", "value": "New Delhi"}
- >[10000](price)
- [500](price)
- [chinese](cuisine) resturants in [chennai](location) with cost more then [100000](price)
- [250](price)
- [400](price)
- [700](price)
- [aman3572@gmail.com](email)

## synonym:300
- cheap

## synonym:4
- four

## synonym:700
- costy

## synonym:Delhi
- New Delhi

## synonym:Mumbai
- Bombay

## synonym:New Delhi
- dilli
- dehli
- delhi
- Delhi
- Dilli
- Dehli

## synonym:bangalore
- Bengaluru

## synonym:chinese
- chines
- Chinese
- Chines

## synonym:ham@mail.com
- ham@ymail.com

## synonym:mid
- moderate

## synonym:vegetarian
- veggie
- vegg

## synonym:xyz@gmail.com
- my email is xyz@gmail.com

## regex:greet
- hey[^\s]*

## regex:pincode
- [0-9]{6}
