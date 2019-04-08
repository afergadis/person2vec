import json
import requests

from person2vec.utils import wiki_extract
from person2vec.utils import wikidata_api_grabber


def set_up():
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
    return entities_entries[list(entities_entries.keys())[0]]


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
    ) == 'American politician, senator, and U.S. Secretary of State'


def test_get_gender():
    test_entity = set_up()
    assert wiki_extract.get_gender(test_entity) == 'female'


def test_get_person_attributes():
    test_entity = set_up()
    assert wikidata_api_grabber._get_person_attributes(test_entity) == {
        "description":
        "American politician, senator, and U.S. Secretary of State",
        "gender":
        "female",
        "occupation":
        "politician"
    }


def test_get_occupation():
    test_entity = set_up()
    assert wiki_extract.get_occupation(test_entity) == 'politician'

