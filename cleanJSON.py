import json
import nltk
from pprint import pprint

with open('E:/work/ML/SampleRASA/starter-pack-rasa-nlu/data/trainingdata.json') as f:
    data = json.load(f)

#pprint(data)
entities = []
dummy_entities = []
tokens = []
for k,v in data.items():
    for k1,v1 in v.items():
        for item in v1:
            tokens.append((nltk.word_tokenize(item['text'])))
            text = item['text']
            inner_entity = []
            inner_entity_dummy = []
            for i in (item['entities']):
                inner_entity.append((i['start'],i['end'],i['entity']))
                inner_entity_dummy.append((text.find(i['value'].strip()),text.find(i['value'].strip())+len(i['value']),i['entity']))
            entities.append(inner_entity)
            dummy_entities.append(inner_entity_dummy)

#new_entities = sorted(entities, key=lambda x: x[0])
for i in range(0,len(entities)):
    print(entities[i])
    print(dummy_entities[i])
    print(tokens[i])
    #print(sorted(each_entity, key=lambda x: x[0]))
    print("----------------------------------------------------------------------------------------------------------")
