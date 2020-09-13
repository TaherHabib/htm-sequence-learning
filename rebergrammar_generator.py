import numpy as np


chars='ATSXPVZ'
emb_chars = 'TP'

graph = [[(1,5),('T','P')] , [(1,2),('S','X')], \
           [(3,5),('V','X')], [(6,),('Z')], \
           [(4,2,3),('T','P','S')], [(4,),('V')] ]


# TO GENERATE SEQUENCES OF SRG
def generateSequences(minLength=5):
    """
    Returns a tuple with
    first entry: as array of chars generated from SRG
    second entry: as array of next possible transitions from each char in the first entry
    """
    while True:
        inchars = ['A']
        node = 0
        outchars = []    
        while node != 6:
            transitions = graph[node]
            i = np.random.randint(0, len(transitions[0]))
            inchars.append(transitions[1][i])
            outchars.append(transitions[1])
            node = transitions[0][i]
        if len(inchars) > minLength:  
            return inchars, outchars
        
        
# TO GENERATE ONE-HOT ENCODINGS OF THE OUTPUT ARRAYS OF 'generateSequences()' function
def get_one_srg(minLength=5):
    inchars, outchars = generateSequences(minLength)
#     print(inchars)
    inseq = []
    outseq= []
    for i,o in zip(inchars, outchars): 
        inpt = np.zeros(7)
        inpt[chars.find(i)] = 1.     
        outpt = np.zeros(7)
        for oo in o:
            outpt[chars.find(oo)] = 1.
        inseq.append(inpt)
        outseq.append(outpt)
    return inseq, outseq


# TO CONVERT BACK INTO SYMBOLS FROM THE ONE-HOT ENCODINGS
def OnehotToWord(sequence):
    """
    converts a sequence (one-hot) in a reber string
    """
    reberString = ''
    for s in sequence:
        index = np.where(s==1.0)[0][0]
        reberString += chars[index]
    reberString+='Z'
    return reberString


def get_n_srg(n, minLength=5):
    examples = []
    for i in range(n):
        examples.append(get_one_srg(minLength))
    return examples



# ____________________________________For ERG strings__________________________________________#


def get_char_onehot(char):
    char_oh = np.zeros(7)
    if chars.find(char) == -1:
        print('Character NOT in Grammar')
        return
    else:
        char_oh[chars.find(char)] = 1.
    return char_oh 


def get_one_erg(minLength=5):
    
    simple_in, simple_out = get_one_srg()
    emb_in = simple_in[:]
    emb_out = simple_out[:]
    
    emb_char = emb_chars[np.random.randint(0, len(emb_chars))]

    emb_in[1:1] = [get_char_onehot(emb_char)]
    emb_in.insert(len(emb_in), get_char_onehot(emb_char))
    print('Embedded INPUT string:', OnehotToWord(emb_in))
    
    emb_out[1:1] = [simple_out[0]]
    emb_out.insert(len(emb_out)-1, get_char_onehot(emb_char))

    return emb_in, emb_out


def get_n_erg(n, minLength=5):
    examples = []
    for i in range(n):
        examples.append(get_one_erg(minLength))
    return examples


def in_grammar(word):
    if word[0] != 'A':
        return False
    node = 0    
    for c in word[1:]:
        transitions = graph[node]
        try:
            node = transitions[0][transitions[1].index(c)]
        except ValueError: # using exceptions for flow control in python is common
            return False
    return True        