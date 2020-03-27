import nltk
import numpy as np
import warnings

#pre-process char level
#load the text
text = open('data/shakespeare.txt', 'r').readlines()

#clean numbers and lines
new_text = ''
for each_line in list(text):
    each_line = each_line.lower()
    if len(each_line) < 5:
        continue
    if '   ' in each_line:
        continue
    if each_line[0] == ' ':
        each_line = each_line[2:]
    
    new_text += each_line
text = new_text

# Create a list of the unique characters in the text
chars = list(set(text))
num_chars = len(chars)

# Create a character -> integer mapping 
char_to_int = {ch:i for i, ch in enumerate(chars)}

# Create a integer -> character mapping
int_to_char = {i:ch for i, ch in enumerate(chars)}

# Function to generate sample text 
def generate(temperature=1.0, seed=None):

    cur_seed = seed
    #set a hard number of lines corresponding to sonnet structure.
    num_lines = 14
    line = 1
    while line < num_lines:
        x = np.zeros((1, win_size, num_chars))

        for t, char in enumerate(seed):
            x[0, t, char_to_int[char]] = 1.

        # Obtain output prob distribution
        probs = model.predict(x, verbose=0)[0]

        # Temperature sampling
        a = np.log(probs)/temperature
        d = np.exp(a)/np.sum(np.exp(a))
        choices = range(len(probs))

        next_idx = np.random.choice(choices, p=d)
        next_char = int_to_char[next_idx]

        cur_seed += next_char
        seed = seed[1 : ] + next_char
        
        if next_char == '\n':
            line += 1

    return cur_seed 

#build LSTM via Keras
import random
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

# Window size
win_size = 40
# Stride size 
stride = 3

# Create RNN architecture: 128 LSTM units 
model = Sequential()
model.add(LSTM(300, return_sequences=False, input_shape=(win_size, num_chars)))
model.add(Dense(num_chars))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Inputs to the RNN 
in_sequence = []
output = []
for i in range(0, len(text) - win_size, stride):
    in_sequence.append(text[i : i + win_size])
    output.append(text[i + win_size])

# Create one-hot encoded inputs and outputs 
X = np.zeros((len(in_sequence), win_size, num_chars))
y = np.zeros((len(in_sequence), num_chars))

for i, example in enumerate(in_sequence):
    for t, char in enumerate(example):
        X[i, t, char_to_int[char]] = 1
    y[i, char_to_int[output[i]]] = 1

# +
#training
# Set training parameters 
tot_epochs = 20
batch = 128
temperature = 0.2

#parameter tuning
epoch_list =[5,10,20]
#tuning number of epochs
for epoch in range(tot_epochs):
    model.fit(X, y, batch_size=batch, nb_epoch=1)
    if epoch in epoch_list:
        print('epoch',epoch)
        print(generate(temperature=0.2, seed="shall i compare thee to a summer's day?\n"))
    
#tuning batch size
for batch in [64,256,128]:
    model.fit(X, y, batch_size=batch, nb_epoch=tot_epochs)
    print('batch',batch)
    print(generate(temperature=1, seed="shall i compare thee to a summer's day?\n"))
    
#varing the temperature for generating poems
for temperature in [1.5,0.75,0.25]:
    print('temperature',temperature)
    print(generate(temperature=temperature, seed="shall i compare thee to a summer's day?\n"))


'''
tuning results
epoch 5
shall i compare thee to a summer's day?
that the sing to the soul the sore the seall the sore the sill dove and the self all the stell for the self love the seare the self all dove to the self love the self all the self to the self all the sore,
and the sear stell that sing the soul the fort the self the stell growe the sore the self love to the sore,
and the sore the store to the store the self all the sould the sing the sore to the sore the sould to the sore,
and the stell the sill the sore the seast the seat the self in the self and thes beat the song to the fort the sell the soll the seave the self love and the self the seave the sore the sore to the self the self and the self and and the seave thee the seate the sore the self in the self all the self all the sore the sore the sing the fort the self and the self and the sore the seave the sore,
and the sull the sore the soul doth the self the self the self in the self to the self love the self in the self the self the sould the sore the sould the self has deat do the sore the self and the sore,
and the seast the sull stoul the stell the love the self doug to the sull the self and the self love the stell douth thee the self the soul the sull for the seed beath the self love the seall do the sore,
and the sill the sull the the store that the self and the self all the stell dove the seat,
the sore the sull the sing the seave the sill doth beting the self the self love the sill doth the soll the self love to the self the seef soll the seef of the sore,
and the seall the self the sing the self and the self and the sore,
the for the sear my hour the sing to the sill growe the for the self the seall the self love the self love thee thou stell doud the love to the the sore the sore to the sore the self deat,
and the self and and to the sore the sore the sould the sore the seave the will doth the seall the seave the sore the sear stell the stell for the sore the stell the self the stell the store the self the self all the seef fart the self and the stell dous deate,
and the sore the self dove the sing the self the sull dove,
the stall that the sull the soul that the self the self the self and the self love the sill doth love the self in the self me to ded,


epoch 10
shall i compare thee to a summer's day?
thy self this for the stare the stall death thee the storn,
which i love the strengle the stare the stall,
and the for the stare the storn the love,
the stall my dear the parse the stornd there,
thou art the starn my self this shall beat,
the star for the seast thou shall thou beart the steed,
which i the stall doth that the still doth to dear,
the start the star for the sure to self beart,
that the stall with my dear the sore to thy self self,
thy self thy self thy self that the stor,
the coms in the love heart the stare the strenged thee the steed,
that the stall doth thou beauty to do the love,
the start that the stall that the stored,

batch 64
shall i compare thee to a summer's day?

hat theugh i sooke pent workn will stalled time,
and corfand whet i am cornile the ever.
when i dreap cankery by my love shall heart
,
and therefore with this chilied, the great,
to gaich mo becaid, and in she love thee might,
so loot under, as thou ure you, blestrexs's resed.
flllfone quice what stall well bein gran:
but not in the from timest all mis thee,
whe can juncer ad my have year dug
to bland bid and make them my deame,
the oubjuy lingrince and night by right.

batch 256
shall i compare thee to a summer's day?
show a asty ad every blaksn and lies ard eve,
then heart blan wirh oot with men, will pood,
or he hur then mine eye lave that my bad.
and giffund fimm ald thy grest me forder,
and to heart that i might then grown,
when it full can deciry i ame for my wind,
sich your loags hath paist thou mest being,
for it love, will in their fie doci appest,
as peture linghist of my varse thon my.
?o llike to me so five as thich i among:
as fall that time, choloot do thy goveng.
i with mo enole wort his didect my ulfersuse,
so thou alteroon, where and this mostre,

batch 128
shall i compare thee to a summer's day?

o mer is feepis groan doth glastast love,
and ever you sunfer's aught non then my noed,
to chume all that i vide my beauty car,
my dost love this chill again, arthe of my thees,
and wist exeeps amend, wrich dis reppatth,
again thes mort, on my strenche himowry write,
and nith is haste, villing on the propsed,
and reck) i bland wir plowis benuty nembl,
which car bling having thy elvidy fair,
or ampety am the wistor's forreculo,
and sur to mine eye ala the gountantors delight,
i nith excessed appepl awl frull surh muding,

'''


