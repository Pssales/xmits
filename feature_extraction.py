# coding: utf-8
import feature

class FeatureExtractor():

    def feature_extractor(sentence,diagram):

        results = feature.extract_key_words(sentence,diagram)
        #results = feature.extract_key_words("","")
        print(results[0])
        print(results[1])
        print(results[2])
        conteudo=""
        for i in range(0, 3):
            conteudo += str(len(results[i]['adverb']))+","
            conteudo += str(len(results[i]['adjective']))+","
            conteudo += str(len(results[i]['article']))+","
            conteudo += str(len(results[i]['noun']))+","
            conteudo += str(len(results[i]['verb']))+","
            conteudo += str(len(results[i]['preposition']))+","
            conteudo += str(len(results[i]['pronoun']))+","

        return conteudo
