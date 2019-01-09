# coding: utf-8

import nltk


def extract_key_words(sentence, diagram_word):

    sent_tokenize = nltk.data.load('tokenizers/punkt/portuguese.pickle')

    sentences = sent_tokenize.tokenize(sentence.lower())
    diagram_words = sent_tokenize.tokenize(diagram_word.lower())

    test_diagram = diagram_words[0].split(" ");
    test_sentence = sentences[0].split(" ");

    result = {'noun': {}, 'article': {}, 'adverb': {}, 'verb': {}, 'pronoun': {}, 'preposition': {},  'adjective': {}}
    result_only_diagram = {'noun': {}, 'article': {}, 'adverb': {}, 'verb': {},'pronoun': {}, 'preposition': {}, 'adjective': {}}
    result_only_sentence = {'noun': {}, 'article': {}, 'adverb': {}, 'verb': {},'pronoun': {}, 'preposition': {}, 'adjective': {}}
    result_intersection = {'noun': {}, 'article': {}, 'adverb': {}, 'verb': {},'pronoun': {}, 'preposition': {}, 'adjective': {}}


    with open('punctuation.txt', 'r') as f:
        punctuation = f.read().split('\n')

    with open('article.txt', 'r', encoding = 'utf8') as f:
        article = f.read().split('\n')

    with open('verb.txt', 'r', encoding = 'utf8') as f:
        verb = f.read().split('\n')

    with open('adjective.txt', 'r', encoding = 'utf8') as f:
        adjective = f.read().split('\n')

    with open('adverb.txt', 'r', encoding = 'utf8') as f:
        adverb = f.read().split('\n')

    with open('pronoun.txt', 'r', encoding = 'utf8') as f:
        pronoun = f.read().split('\n')

    with open('preposition.txt', 'r', encoding = 'utf8') as f:
        preposition = f.read().split('\n')

    for sentence in sentences:

        for word in nltk.word_tokenize(sentence):

            word = word.lower()

            tmp_class = None

            if word in punctuation:
                print('Ignoring {}'.format(word))

            elif word in article:
                tmp_class = 'article'

            elif word in adjective:
                tmp_class = 'adjective'

            elif word in adverb:
                tmp_class = 'adverb'

            elif word in verb:
                tmp_class = 'verb'

            elif word in pronoun:
                tmp_class = 'pronoun'

            elif word in preposition:
                tmp_class = 'preposition'

            else:
                tmp_class = 'noun'

            if tmp_class is not None:
                update_dict(result, tmp_class, word)

                if word in test_diagram:
                    update_dict_intersect(result_intersection, tmp_class, word)
                else:
                    update_dict_only(result_only_sentence,tmp_class,word)

    for diagram_word in diagram_words:

        for word in nltk.word_tokenize(diagram_word):

            word = word.lower()

            tmp_class = None

            if word in punctuation:
                print('Ignoring {}'.format(word))

            elif word in article:
                tmp_class = 'article'

            elif word in adjective:
                tmp_class = 'adjective'

            elif word in adverb:
                tmp_class = 'adverb'

            elif word in verb:
                tmp_class = 'verb'

            elif word in pronoun:
                tmp_class = 'pronoun'

            elif word in preposition:
                tmp_class = 'preposition'

            else:
                tmp_class = 'noun'

            if tmp_class is not None:
                update_dict(result, tmp_class, word)
                if word not in test_sentence:
                    update_dict_only(result_only_diagram, tmp_class, word)



    return result_only_sentence,result_only_diagram,result_intersection


def update_dict(dictionary, key, word):
    dictionary[key][word] = 1 if word not in dictionary[key] else dictionary[key][word] + 1


def update_dict_intersect(dictionary, key, word):
    dictionary[key][word] = 1 if word not in dictionary[key] else dictionary[key][word] + 1

def update_dict_only(dictionary, key, word):
    dictionary[key][word] = 1 if word not in dictionary[key] else dictionary[key][word] + 1