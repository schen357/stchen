#!/usr/bin/env python
import argparse
import copy
import requests
import calendar

import urllib
from urlparse import parse_qs
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Hardcoded list of campgrounds I'm willing to sleep at
PARKS = {
    '70925': 'UPPER PINES',
    '70928': 'LOWER PINES',
    '70927': 'NORTH PINES',
#    '73635': 'STANISLAUS',
#    '70926': 'TUOLOMNE MEADOWS'
# only interested in pines campgrounds
}

# Sets the search location to yosemite
LOCATION_PAYLOAD = {
    'currentMaximumWindow': '12',
    'locationCriteria': 'yosemite',
    'interest': '',
    'locationPosition': '',
    'selectedLocationCriteria': '',
    'resetAllFilters':    'false',
    'filtersFormSubmitted': 'false',
    'glocIndex':    '0',
    'googleLocations':  'Yosemite National Park, Yosemite Village, CA 95389, USA|-119.53832940000001|37.8651011||LOCALITY'
}

# Sets the search type to camping
CAMPING_PAYLOAD = {
    'resetAllFilters':  'false',
    'filtersFormSubmitted': 'true',
    'sortBy':   'RELEVANCE',
    'category': 'camping',
    'selectedState':    '',
    'selectedActivity': '',
    'selectedAgency':   '',
    'interest': 'camping',
    'usingCampingForm': 'true'
}

# Runs the actual search
PAYLOAD = {
    'resetAllFilters':   'false',
    'filtersFormSubmitted': 'true',
    'sortBy':   'RELEVANCE',
    'category': 'camping',
    'availability': 'all',
    'interest': 'camping',
    'usingCampingForm': 'false'
}


BASE_URL = "https://www.recreation.gov"
UNIF_SEARCH = "/unifSearch.do"
UNIF_RESULTS = "/unifSearchResults.do"

def findCampSites(args):
    start_end_dates = generateDates(args)

    for i in range(0, len(start_end_dates)-1):
        start_date = start_end_dates[0][i]
        end_date = start_end_dates[1][i]

        payload = generatePayload(start_date, end_date)

        content_raw = sendRequest(payload)
        html = BeautifulSoup(content_raw, 'html.parser')
        sites = getSiteList(html, start_date, end_date)
    return sites

# keep for if start_date and end_date are params
def getNextDay(date):
    date_object = datetime.strptime(date, "%Y-%m-%d")
    next_day = date_object + timedelta(days=1)
    return datetime.strftime(next_day, "%Y-%m-%d")

def formatDate(date):
    if (type(date) == str):
        date_object = datetime.strptime(date, "%Y-%m-%d")
        date_formatted = datetime.strftime(date_object, "%a %b %d %Y")
    else:
        date_formatted = datetime.strftime(date, "%a %b %d %Y")
    return date_formatted

def generateDates(args):
    if args['start_date']:
        start_dates = [args['start_date']]
        end_dates = [args['end_date']]
    elif args['year'] and args['month'] and args['day_of_week']:
        monthWeeks = calendar.Calendar().monthdatescalendar(args['year'], args['month'])
        dayOfWeek = list(calendar.day_name).index(args['day_of_week'])

        start_dates = []
        end_dates = []
        # create set of start dates for month given day of week (e.g. all Fridays)
        for i in range(len(monthWeeks)):
            if monthWeeks[i][dayOfWeek] > datetime.today().date():
                start_dates.append(monthWeeks[i][dayOfWeek])
                end_dates.append(monthWeeks[i][dayOfWeek] + timedelta(days=args['num_nights']))

    start_end_dates = [start_dates, end_dates]

    return start_end_dates

def generatePayload(start_date, end_date):
    payload = copy.copy(PAYLOAD)

    payload['arrivalDate'] = formatDate(start_date)
    payload['departureDate'] = formatDate(end_date)

    return payload

def getSiteList(html, start_date, end_date):
    sites = html.findAll('div', {"class": "check_avail_panel"})
    results = []

    start_date_cln = formatDate(start_date)
    end_date_cln = formatDate(end_date)

    for site in sites:
        if site.find('a', {'class': 'book_now'}):
            get_url = site.find('a', {'class': 'book_now'})['href']
            # Strip down to get query parameters
            get_query = get_url[get_url.find("?") + 1:] if get_url.find("?") >= 0 else get_url
            if get_query:
                get_params = parse_qs(get_query)
                siteId = get_params['parkId']
                if siteId and siteId[0] in PARKS:
                    results.append("%s, Booking Url: %s&arrivalDate=%s&departureDate=%s" % (PARKS[siteId[0]], BASE_URL + get_url, start_date_cln, end_date_cln))
    return results

def sendRequest(payload):
    with requests.Session() as s:

        s.get(BASE_URL + UNIF_RESULTS, verify=False) # Sets session cookie
        s.post(BASE_URL + UNIF_SEARCH, LOCATION_PAYLOAD, verify=False) # Sets location to yosemite
        s.post(BASE_URL + UNIF_SEARCH, CAMPING_PAYLOAD, verify=False) # Sets search type to camping

        resp = s.post(BASE_URL + UNIF_SEARCH, payload, verify=False) # Runs search on specified dates
        if (resp.status_code != 200):
            raise Exception("failedRequest","ERROR, %d code received from %s".format(resp.status_code, BASE_URL + SEARCH_PATH))
        else:
            return resp.text

# SC updating to make command input a month and day(s) of week so that script can check any weekend of a month for instance
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=int, help="Month [MM] with no leading 0s e.g. Jan = 1")
    parser.add_argument("--year", type=int, help="Year [YYYY]")
    parser.add_argument("--day_of_week", type=str, help="First night of week e.g. Friday")
    parser.add_argument("--num_nights", type=int, help="Consecutive number of nights desired e.g. 2, default = 1")
    parser.add_argument("--start_date", type=str, help="Start date [YYYY-MM-DD]")
    parser.add_argument("--end_date", type=str, help="End date [YYYY-MM-DD]")

    args = parser.parse_args()
    arg_dict = vars(args)
    if (arg_dict['start_date']) and ('end_date' not in arg_dict or not arg_dict['end_date']):
        arg_dict['end_date'] = getNextDay(arg_dict['start_date']) # if no end date, use next day from start date
    if 'num_nights' not in arg_dict or not arg_dict['num_nights']:
        arg_dict['num_nights'] = 1 # set to 1 night if no num nights given

    sites = findCampSites(arg_dict)
    if sites:
        for site in sites:
            print site
