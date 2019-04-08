import json
import requests

from person2vec.utils import wiki_extract
from person2vec.utils import wikidata_api_grabber


ENTRIES = None


def set_up():
    global ENTRIES
    if ENTRIES is None:
        WIKIDATA_TITLE_URL = "https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&titles=%s&format=json"
        headers = {
            'User-Agent': 'ML project for describing famous people',
            'From': 'aaabbbcccz@gmail.com'
        }
        r = requests.get(
            WIKIDATA_TITLE_URL % ('Hillary_Clinton'),
            headers=headers,
            verify=False)
        test_input = json.loads(r.text)
        entities_entries = test_input['entities']
        ENTRIES = entities_entries[list(entities_entries.keys())[0]]
    return ENTRIES


def test_get_instance_of():
    test_entity = set_up()
    assert wiki_extract.get_instance_of(test_entity) == 'human'


def test_get_title():
    test_entity = set_up()
    assert wiki_extract.get_title(test_entity) == 'Hillary Clinton'


def test_get_description():
    test_entity = set_up()
    assert wiki_extract.get_description(
        test_entity
    ) == 'American politician, senator, Secretary of State, First Lady'


def test_get_gender():
    test_entity = set_up()
    assert wiki_extract.get_gender(test_entity) == 'female'


def test_get_person_attributes():
    test_entity = set_up()
    person_attributes = wikidata_api_grabber._get_person_attributes(test_entity)
    del person_attributes['claims']
    assert person_attributes == {
        'description':
        'American politician, senator, Secretary of State, First Lady',
        'gender':
        'female',
        'occupation':
        'politician',
        'birth_date':
        '+1947-10-26T00:00:00Z',
        'political_party':
        'democrat'
    }


def test_get_occupation():
    test_entity = set_up()
    assert wiki_extract.get_occupation(test_entity) == 'politician'
