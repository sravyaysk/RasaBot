from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Interpreter


training_data = load_data("E:/work/ML/SampleRASA/starter-pack-rasa-nlu/data/trainingdata.json")
trainer = Trainer(config.load("E:/work/ML/SampleRASA/starter-pack-rasa-nlu/nlu_config.yml"))
#interpreter = trainer.train(training_data)
trainer.train(training_data)
model_directory = trainer.persist("E:/work/ML/SampleRASA/starter-pack-rasa-nlu/models/", project_name="nlu")
interpreter = Interpreter.load(model_directory)
output = interpreter.parse(u"Deductible:USD 10,000 each and every Claim. Including costs and Expenses Deductible:")
#parsing = interpreter.parse('hello')
print(output)
# assert parsing['intent']['name'] == 'greet'
# assert model_directory
