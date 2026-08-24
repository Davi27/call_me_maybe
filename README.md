Contract Table

argument1:
path: data/input/functions_definition.json
contains: the functions available to be selected by the model
required by the subject: to be formated in JSON
unknown: if missing return the program

argument2:
path: data/input/function_calling_tests.json
contains: strings to be used as prompts
required by the subject: to be formated in JSON
unknown: if missing return the program

argument3:
path: data/output/function_calls.json
contains: the file where the output of the program will be placed
required by the subject: to be formated in JSON
unknown: if missing tue program will have no output