import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# select id,creationDate,modificationDate from collegereviews.CollegeReview_MainTable where status in ("published") limit 9000000;
# select * from collegereviews.CollegeReview_Tracking where action in ("reviewEdited","autoModerated","reviewAdded","reviewDescriptionUpdated","reviewTitleUpdated","facultyDescriptionUpdated","infraDescriptionUpdated","placementDescriptionUpdated") order by reviewId,addedOn limit 90000000;

publishedReviews = pd.read_csv("/media/mohd/New Volume/hackathon21/cr_mt.csv")
reviewHistory = pd.read_csv("/media/mohd/New Volume/hackathon21/cr_t.csv")

## 85/10/5 split
publishedReviews_train, publishedReviews_temp = train_test_split(publishedReviews, train_size = 0.85, test_size = 0.15, random_state = 100)
publishedReviews_val, publishedReviews_test = train_test_split(publishedReviews_temp, train_size = 0.67, test_size = 0.33, random_state = 100)


publishedReviewsIds = set(publishedReviews_train["id"].tolist())
# publishedReviewsIds = set(publishedReviews_val["id"].tolist())
# publishedReviewsIds = set(publishedReviews_test["id"].tolist())

## get train 
## date sorted
trackingHist = reviewHistory[reviewHistory["reviewId"].isin( publishedReviewsIds)]

REV_START_LOG = ("reviewAdded","autoModerated")
REV_EDIT_LOG = "reviewEdited" ### dont add as traingn
REV_AUTOMOD_LOG = ("autoModerated")
REV_DESCUPD_LOG,REV_TITLUPD_LOG,REV_FACUUPD_LOG,REV_INFAUPD_LOG,REV_PLACUPD_LOG = "reviewDescriptionUpdated","reviewTitleUpdated","facultyDescriptionUpdated","infraDescriptionUpdated","placementDescriptionUpdated"
REV_VALID_SERIES = set([REV_AUTOMOD_LOG,REV_DESCUPD_LOG,REV_TITLUPD_LOG,REV_FACUUPD_LOG,REV_INFAUPD_LOG,REV_PLACUPD_LOG])

log_col_map = dict()
log_col_map[REV_DESCUPD_LOG] = "reviewDescription"
log_col_map[REV_TITLUPD_LOG] = "reviewTitle"
log_col_map[REV_FACUUPD_LOG] = "facultyDescription"
log_col_map[REV_INFAUPD_LOG] = "infraDescription"
log_col_map[REV_PLACUPD_LOG] = "placementDescription"

def getReviewLogDict(reviewLog):
	resp = {}
	try:
		resp = json.loads(reviewLog)
	except Exception as e:
		print("json____decode" ,e, reviewLog)
	return resp

def getJsonDiff(old,new):
	data = []
	if log_col_map[REV_DESCUPD_LOG] in old and log_col_map[REV_DESCUPD_LOG] in new and old[log_col_map[REV_DESCUPD_LOG]]!=new[log_col_map[REV_DESCUPD_LOG]]:
		data.append([old[log_col_map[REV_DESCUPD_LOG]],new[log_col_map[REV_DESCUPD_LOG]]])
	if log_col_map[REV_TITLUPD_LOG] in old and log_col_map[REV_TITLUPD_LOG] in new and old[log_col_map[REV_TITLUPD_LOG]]!=new[log_col_map[REV_TITLUPD_LOG]]:
		data.append([old[log_col_map[REV_TITLUPD_LOG]],new[log_col_map[REV_TITLUPD_LOG]]])
	if log_col_map[REV_FACUUPD_LOG] in old and log_col_map[REV_FACUUPD_LOG] in new and old[log_col_map[REV_FACUUPD_LOG]]!=new[log_col_map[REV_FACUUPD_LOG]]:
		data.append([old[log_col_map[REV_FACUUPD_LOG]],new[log_col_map[REV_FACUUPD_LOG]]])
	if log_col_map[REV_INFAUPD_LOG] in old and log_col_map[REV_INFAUPD_LOG] in new and old[log_col_map[REV_INFAUPD_LOG]]!=new[log_col_map[REV_INFAUPD_LOG]]:
		data.append([old[log_col_map[REV_INFAUPD_LOG]],new[log_col_map[REV_INFAUPD_LOG]]])
	if log_col_map[REV_PLACUPD_LOG] in old and log_col_map[REV_PLACUPD_LOG] in new and old[log_col_map[REV_PLACUPD_LOG]]!=new[log_col_map[REV_PLACUPD_LOG]]:
		data.append([old[log_col_map[REV_PLACUPD_LOG]],new[log_col_map[REV_PLACUPD_LOG]]])
	return data

CONSIDER_AUTOMOD_DATA = True
CONSIDER_DESCUPD_LOG = False
CONSIDER_TITLUPD_LOG = False
CONSIDER_FACUUPD_LOG = True
CONSIDER_INFAUPD_LOG = True
CONSIDER_PLACUPD_LOG = True
AUTO_MOD_CNT = 0

VERBOSE = False

train_data = []
INVALID_DATA_SOLO = []
INVALID_DATA_SERIES = []
INVALID_DATA = []
INVALID_DATA_FIRST_ACTION_NOT_EDIT_ADD = []
INVALID_DATA_EDIT = []
INVALID_DATA_ADD_DEL = []
INVALID_DATA_LAST_NOT_PRESENT = []
IDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT = []
reviewspicked = []
isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT_GLOBALFLAG = True #################### ONLY 15% data

for reviewId in tqdm(publishedReviewsIds):
	try:	
		reviewHist = trackingHist[trackingHist["reviewId"]==reviewId]
		if REV_EDIT_LOG in reviewHist["action"].tolist(): #### No edited reviews FOR NOW, 
			INVALID_DATA_EDIT.append(reviewId)
		elif len(reviewHist) == 1:
			INVALID_DATA_SOLO.append(reviewId)
		elif len(reviewHist) > 1 :
			###
			lastAction = reviewHist[["action"]].iloc[0][0]
			if lastAction not in REV_START_LOG:
				INVALID_DATA_FIRST_ACTION_NOT_EDIT_ADD.append(reviewId)
				continue
			lastReviewState = getReviewLogDict(reviewHist[["data"]].iloc[0][0])
			reviewHist = reviewHist.iloc[1:,:]  ## remove first entry
			actionSeries = reviewHist["action"]
			actionSeriesSet = set(actionSeries.tolist())
			if (REV_VALID_SERIES==actionSeriesSet):
				INVALID_DATA_SERIES.append(reviewId)
				continue
			isREV_DESCUPD_LOG = False
			isREV_TITLUPD_LOG = False
			isREV_FACUUPD_LOG = False
			isREV_INFAUPD_LOG = False
			isREV_PLACUPD_LOG = False
			isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT = False 
			train_data_review = []
			idxDESC = actionSeries.where(actionSeries==REV_DESCUPD_LOG).last_valid_index()
			if CONSIDER_DESCUPD_LOG and idxDESC is not None:
				if log_col_map[REV_DESCUPD_LOG] in lastReviewState:
					logText = reviewHist.loc[[idxDESC],"data"].iloc[0]
					if lastReviewState[log_col_map[REV_DESCUPD_LOG]].strip()!=logText.strip():
						train_data_review.append([lastReviewState[log_col_map[REV_DESCUPD_LOG]].strip(),logText.strip()])
						isREV_DESCUPD_LOG  =True
					else:
						_=1
						#TODO No change
				else:
					INVALID_DATA_LAST_NOT_PRESENT.append(reviewId)
					isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT = True
					if VERBOSE: print(reviewId,"desc")
			idxTITL = actionSeries.where(actionSeries==REV_TITLUPD_LOG).last_valid_index()
			if CONSIDER_TITLUPD_LOG and idxTITL is not None:
				if log_col_map[REV_TITLUPD_LOG] in lastReviewState:
					logText = reviewHist.loc[[idxTITL],"data"].iloc[0]
					if lastReviewState[log_col_map[REV_TITLUPD_LOG]].strip()!=logText.strip():
						train_data_review.append([lastReviewState[log_col_map[REV_TITLUPD_LOG]].strip(),logText.strip()])
						isREV_TITLUPD_LOG  =True
					else:
						_=1
						#TODO No change
				else:
					INVALID_DATA_LAST_NOT_PRESENT.append(reviewId)
					isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT = True
					if VERBOSE: print(reviewId,"titl")
			idxFACU = actionSeries.where(actionSeries==REV_FACUUPD_LOG).last_valid_index()
			if CONSIDER_FACUUPD_LOG and  idxFACU is not None:
				if log_col_map[REV_FACUUPD_LOG] in lastReviewState:
					logText = reviewHist.loc[[idxFACU],"data"].iloc[0]
					if lastReviewState[log_col_map[REV_FACUUPD_LOG]].strip()!=logText.strip():
						train_data_review.append([lastReviewState[log_col_map[REV_FACUUPD_LOG]].strip(),logText.strip()])
						isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT = True
					else:
						_=1
						#TODO No change
					isREV_FACUUPD_LOG  =True
				else:
					INVALID_DATA_LAST_NOT_PRESENT.append(reviewId)
					if VERBOSE: print("FACU")
			idxINFA = actionSeries.where(actionSeries==REV_INFAUPD_LOG).last_valid_index()
			if CONSIDER_INFAUPD_LOG and  idxINFA is not None:
				if log_col_map[REV_INFAUPD_LOG] in lastReviewState:
					logText = reviewHist.loc[[idxINFA],"data"].iloc[0]
					if lastReviewState[log_col_map[REV_INFAUPD_LOG]].strip()!=logText.strip():
						train_data_review.append([lastReviewState[log_col_map[REV_INFAUPD_LOG]].strip(),logText.strip()])
						isREV_INFAUPD_LOG  =True
					else:
						_=1
						#TODO No change
				else:
					INVALID_DATA_LAST_NOT_PRESENT.append(reviewId)
					isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT = True
					if VERBOSE: print("INFA")
			idxPLAC = actionSeries.where(actionSeries==REV_PLACUPD_LOG).last_valid_index()
			if CONSIDER_PLACUPD_LOG and  idxPLAC is not None:
				if log_col_map[REV_PLACUPD_LOG] in lastReviewState:
					logText = reviewHist.loc[[idxPLAC],"data"].iloc[0]
					if lastReviewState[log_col_map[REV_PLACUPD_LOG]].strip()!=logText.strip():
						train_data_review.append([lastReviewState[log_col_map[REV_PLACUPD_LOG]].strip(),logText.strip()])
						isREV_PLACUPD_LOG  =True
					else:
						_=1
						#TODO No change
				else:
					INVALID_DATA_LAST_NOT_PRESENT.append(reviewId)
					isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT = True
					if VERBOSE: print("PLAC")
			if REV_AUTOMOD_LOG in actionSeriesSet and CONSIDER_AUTOMOD_DATA:
				## automoderate data
				idxAUTO = actionSeries.where(actionSeries==REV_AUTOMOD_LOG).last_valid_index()
				if idxAUTO is not None:
					currentReviewState = getReviewLogDict(reviewHist.loc[[idxAUTO],"data"].iloc[0])
					if CONSIDER_DESCUPD_LOG and not isREV_DESCUPD_LOG: 
						colName = log_col_map[REV_DESCUPD_LOG]
						if colName in lastReviewState and colName in currentReviewState and lastReviewState[colName].strip()!=currentReviewState[colName].strip():
							train_data_review.append([lastReviewState[colName].strip(),currentReviewState[colName].strip()])
						if colName not in lastReviewState and colName in currentReviewState or colName in lastReviewState and colName not in currentReviewState:
							INVALID_DATA_ADD_DEL.append(reviewId)
							isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT = True
							if VERBOSE: print(REV_DESCUPD_LOG)  ###
					if CONSIDER_TITLUPD_LOG and not isREV_TITLUPD_LOG: 
						colName = log_col_map[REV_TITLUPD_LOG]
						if colName in lastReviewState and colName in currentReviewState and lastReviewState[colName].strip()!=currentReviewState[colName].strip():
							train_data_review.append([lastReviewState[colName].strip(),currentReviewState[colName].strip()])
						if colName not in lastReviewState and colName in currentReviewState or colName in lastReviewState and colName not in currentReviewState:
							INVALID_DATA_ADD_DEL.append(reviewId)
							isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT = True
							if VERBOSE: print(REV_TITLUPD_LOG)
					if CONSIDER_FACUUPD_LOG and not isREV_FACUUPD_LOG: 
						colName = log_col_map[REV_FACUUPD_LOG]
						if colName in lastReviewState and colName in currentReviewState and lastReviewState[colName].strip()!=currentReviewState[colName].strip():
							train_data_review.append([lastReviewState[colName].strip(),currentReviewState[colName].strip()])
						if colName not in lastReviewState and colName in currentReviewState or colName in lastReviewState and colName not in currentReviewState:
							isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT = True
							INVALID_DATA_ADD_DEL.append(reviewId)
							if VERBOSE: print(REV_FACUUPD_LOG)
					if CONSIDER_INFAUPD_LOG and not isREV_INFAUPD_LOG: 
						colName = log_col_map[REV_INFAUPD_LOG]
						if colName in lastReviewState and colName in currentReviewState and lastReviewState[colName].strip()!=currentReviewState[colName].strip():
							train_data_review.append([lastReviewState[colName].strip(),currentReviewState[colName].strip()])
						if colName not in lastReviewState and colName in currentReviewState or colName in lastReviewState and colName not in currentReviewState:
							INVALID_DATA_ADD_DEL.append(reviewId)
							isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT = True
							if VERBOSE: print(REV_INFAUPD_LOG)
					if CONSIDER_PLACUPD_LOG and not isREV_PLACUPD_LOG: 
						colName = log_col_map[REV_PLACUPD_LOG]
						if colName in lastReviewState and colName in currentReviewState and lastReviewState[colName].strip()!=currentReviewState[colName].strip():
							train_data_review.append([lastReviewState[colName].strip(),currentReviewState[colName].strip()])
						if colName not in lastReviewState and colName in currentReviewState or colName in lastReviewState and colName not in currentReviewState:
							INVALID_DATA_ADD_DEL.append(reviewId)
							isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT = True
							if VERBOSE: print(REV_PLACUPD_LOG)
			if isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT:
				IDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT.append(reviewId)
			if not isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT_GLOBALFLAG:
				train_data+=train_data_review
				reviewspicked.append(reviewId)
			else:
				if not isIDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT:
					train_data+=train_data_review
					reviewspicked.append(reviewId)
			# currentReviewState = getReviewLogDict(reviewHist[["data"]].iloc[0][0])
			# for index, reviewLog in enumerate(reviewHist.itertuples()):
			# 	if index > 0:
			# 		currentAction = reviewHist[["action"]].iloc[index][0]
			# 		if currentAction in REV_START_LOG:
			# 			currentReviewState = getReviewLogDict(reviewHist[["data"]].iloc[index][0])
			# 		elif currentAction == REV_AUTOMOD_LOG:
			# 			if CONSIDER_AUTOMOD_DATA:
			# 				train_data+= getJsonDiff(currentReviewState,getReviewLogDict(reviewHist[["data"]].iloc[index][0]))
			# 				AUTO_MOD_CNT = AUTO_MOD_CNT +1
			# 			currentReviewState = getReviewLogDict(reviewHist[["data"]].iloc[index][0])
			# 		elif currentAction == REV_DESCUPD_LOG:
			# 			currentReviewState[log_col_map[REV_DESCUPD_LOG]] = reviewHist[["data"]].iloc[index][0]
			# 			train_data.append([currentReviewState[log_col_map[REV_DESCUPD_LOG]] , reviewHist[["data"]].iloc[index][0]])
			# 		elif currentAction == REV_TITLUPD_LOG:
			# 			currentReviewState[log_col_map[REV_TITLUPD_LOG]] = reviewHist[["data"]].iloc[index][0]
			# 			train_data.append([currentReviewState[log_col_map[REV_TITLUPD_LOG]] , reviewHist[["data"]].iloc[index][0]])
			# 		elif currentAction == REV_FACUUPD_LOG:
			# 			currentReviewState[log_col_map[REV_FACUUPD_LOG]] = reviewHist[["data"]].iloc[index][0]
			# 			train_data.append([currentReviewState[log_col_map[REV_FACUUPD_LOG]] , reviewHist[["data"]].iloc[index][0]])
			# 		elif currentAction == REV_INFAUPD_LOG:
			# 			currentReviewState[log_col_map[REV_INFAUPD_LOG]] = reviewHist[["data"]].iloc[index][0]
			# 			train_data.append([currentReviewState[log_col_map[REV_INFAUPD_LOG]] , reviewHist[["data"]].iloc[index][0]])
			# 		elif currentAction == REV_PLACUPD_LOG:
			# 			currentReviewState[log_col_map[REV_PLACUPD_LOG]] = reviewHist[["data"]].iloc[index][0]
			# 			train_data.append([currentReviewState[log_col_map[REV_PLACUPD_LOG]] , reviewHist[["data"]].iloc[index][0]])
			# 		else:
			# 			INVALID_DATA.append(reviewId,lastAction,currentAction)
			# 		lastAction = currentAction
		# break ###################
	except Exception as e:
		print(e)



print(len(INVALID_DATA_EDIT))
print(len(INVALID_DATA_SOLO))
print(len(INVALID_DATA))
print(len(INVALID_DATA_FIRST_ACTION_NOT_EDIT_ADD))
print(AUTO_MOD_CNT)
print(len(publishedReviewsIds))
print(len(train_data))
print(train_data[0:10])
print(len(INVALID_DATA_ADD_DEL))
print(len(INVALID_DATA_SERIES))
print(len(INVALID_DATA_LAST_NOT_PRESENT))
print(len(IDS_NOT_CONSIDERED_DUE_TO_CONTENT_SHIFT))
print(len(reviewspicked))

data = pd.DataFrame(train_data, columns=["source","target"])
# data.to_csv("/media/mohd/New Volume/hackathon21/train.csv")
# data.to_csv("/media/mohd/New Volume/hackathon21/val.csv")
# data.to_csv("/media/mohd/New Volume/hackathon21/test.csv")

# REMOVE title-desc

### Emrge tile desc - -INVALID_DATA_ADD_DEL  - PRBLEM

# {"placementDescription":"Mostly tcs like companies visit campus but the placements are decent. Almost 90% students get placed in the companies like TCS, Infosys, Capgemini etc. College Life is good many events keep happening around the year. The marking system is strict it is difficult to get good CGPA.","infraDescription":"Infrastructure is not very good but the environment is very good mess food is good. Hostel room is small and 4 students live together. The campus is very clean and is full of trees. There are many canteens all over the campus. Facilities provided are less as compared to other private universities. ","facultyDescription":"Teachers are good and course curriculum is also good. Teachers are highly qualified the classes are equipped with good infrastructure and the labs are also good. The course curriculum is good it provides all the skills that are required to get jobs. ","reviewTitle":"Good college, large campus, nice placements excellent environment, large library. "}


# College life is good, and many events keep happening around the year.

# It is a good college with a large campus, good placements excellent environment and a large library.


### max char length***
# avg -300
from collections import Counter
lenSet = list()
for row in train_data:
	lenSet.append(len(row[0]))
	lenSet.append(len(row[1]))

lenSet.sort(reverse=True)
print(lenSet[0:100])
print(Counter(lenSet).most_common(100))


## max_seq_length - 60 avg
lenSet = list()
for row in train_data:
	lenSet.append(len(row[0].split()))
	lenSet.append(len(row[1].split()))

lenSet.sort(reverse=True)
print(lenSet[0:100])
print(Counter(lenSet).most_common(100))


