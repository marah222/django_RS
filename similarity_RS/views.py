from django.shortcuts import render
from django.http import HttpResponse
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from . import restaurants as rest
from . import hotels
from django.views.decorators.csrf import csrf_exempt


# Create your views


# retaurn top 10 similar users
@csrf_exempt
def getSimilarRestaurants(request, location_id):

    if request.method == 'POST':
        try:

            data = json.loads(str(request.body, encoding='utf-8'))
            ## mapping location-id->data
            processed_data = rest.dict_key_location_id(data)

            mouther_df = pd.DataFrame(data['data']);
            rate_price_df = rest.preprocess_price_rating(mouther_df[['location_id', 'rating', 'price']])

            # print(df)
            # """## Add TF-IDF Features """
            tfidf_df = rest.TF_IDF_preprocessing(mouther_df[['location_id', 'description', 'cuisine']])
            print('*'*100)

            cleanData = pd.merge(rate_price_df, tfidf_df, on='location_id')
            restaurants_similarities = cosine_similarity(cleanData)

            rest_similarity_df = pd.DataFrame(restaurants_similarities, index=cleanData.index, columns=cleanData.index)

            result_IDs = rest_similarity_df[location_id].sort_values(ascending=False)

            result = {}
            result['data'] = []

            for loc, sim in result_IDs[0:11].items():
                # print(loc,sim)
                result['data'].append(processed_data[loc])
            return HttpResponse(json.dumps(result), 200)

        except:
            return HttpResponse('Error', 500)

@csrf_exempt
def getSimilarHotels(request, location_id):
    if request.method == 'POST':
        try:

            data = json.loads(str(request.body, encoding='utf-8'))
            ## mapping location-id->data
            processed_data = {}
            for item in data['data']:
                temploc = item['hotel']['hotelId']
                processed_data[temploc] = item;
                item['price'] = item['offers'][0]['price']['total']
            mouther_df = pd.json_normalize(data['data']);

            rate_price_df = hotels.preprocess_price_rating(mouther_df[['hotel.hotelId', 'hotel.rating', 'price']])

            # print(df)
            # """## Add TF-IDF Features """
            tfidf_df = hotels.TF_IDF_preprocessing(
                mouther_df[['hotel.hotelId', 'hotel.description.text', 'hotel.amenities']])
            cleanData = pd.merge(rate_price_df, tfidf_df, on='hotel.hotelId')
            Hotels_similarities = cosine_similarity(cleanData)
            Hotels_similarity_df = pd.DataFrame(Hotels_similarities, index=cleanData.index, columns=cleanData.index)

            result_IDs = Hotels_similarity_df[location_id].sort_values(ascending=False)

            result = {}
            result['data'] = []

            for loc, sim in result_IDs[0:11].items():
                # print(loc,sim)
                result['data'].append(processed_data[loc])

            # return HttpResponse('YES')
            return HttpResponse(json.dumps(result), 200)

        except:
            return HttpResponse('Error', 500)



